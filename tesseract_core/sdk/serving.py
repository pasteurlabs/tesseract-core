# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a served Tesseract is, and what everything serving one needs.

The interface a running Tesseract presents -- :class:`ServedTesseract`, and the
handful of questions worth asking one that docker-py has no method for -- plus
choosing a port to serve on, waiting for the server to answer, and telling a lost
race for a port apart from a Tesseract that is genuinely broken.

None of it is particular to how the Tesseract is run, so none of it belongs in
the module that knows how to run one. That cuts both ways: nothing here may
import a transport, which is why the startup path catches broadly rather than
naming one transport's exceptions.

Implemented by :mod:`tesseract_core.sdk.docker_client` for containers and by
:mod:`tesseract_core.sdk.local_client` for subprocesses. Both import this to
register their answers; neither is imported back.
"""

import logging
import random
import socket
import time
from collections.abc import Mapping, Sequence
from contextlib import closing
from functools import singledispatch
from pathlib import Path
from typing import Any, Protocol

import requests

from .exceptions import UserError

logger = logging.getLogger("tesseract")

# How long to wait for a freshly started Tesseract to answer /health.
DEFAULT_STARTUP_TIMEOUT = 30.0


class ServedTesseract(Protocol):
    """A Tesseract that has been started and can be reached, inspected and stopped.

    Satisfied by :class:`~tesseract_core.sdk.docker_client.Container` and by
    :class:`~tesseract_core.sdk.local_client.TesseractProcess`, so callers that
    only need to talk to a Tesseract, read its output or shut it down need not
    know which of the two they hold.

    Structural, so neither has to declare allegiance to it: a `Container` is a
    mirror of docker-py's, and fits this by having the methods docker-py gave it
    rather than by inheriting anything from us.

    Deliberately narrow: it promises nothing about separating stdout from stderr,
    since a Tesseract served as a bare process writes both to one file.
    """

    host_ip: str | None
    host_port: str | None

    @property
    def url(self) -> str:
        """Base URL the Tesseract is serving on."""

    def reload(self) -> None:
        """Read the Tesseract's state again."""

    def remove(self, v: bool = False, link: bool = False, force: bool = False) -> None:
        """Dispose of the Tesseract, leaving nothing of it behind.

        docker-py's ``Container.remove`` signature exactly, since that is what a
        `Container` has and this is structural -- including refusing one that is
        still running unless ``force`` is set. ``v`` and ``link`` are Docker's
        and mean nothing to a Tesseract served any other way.
        """

    def wait(self, timeout: float | None = None) -> dict:
        """Wait for the Tesseract to stop, and report the status it stopped with.

        Shaped after docker-py's ``Container.wait``, down to returning a dict
        keyed by ``StatusCode``. Waits for as long as the Tesseract runs unless
        ``timeout`` says otherwise, so ask only about one you expect to have
        stopped -- and before disposing of it, since a Tesseract that is gone can
        no longer be asked.
        """

    def logs(self) -> bytes:
        """Everything the Tesseract has written so far."""


@singledispatch
def diagnose_exit(served: ServedTesseract, logs: str) -> str:
    """Anything this Tesseract can add about why it stopped running.

    The code it exited with and what it wrote are reported by whoever noticed.
    This is for what remains: a cause the transport can name and the logs cannot.
    Takes the logs as evidence, not to repeat them.

    Dispatched rather than a method, so a `Container` keeps to the shape of
    docker-py, which has nothing like this. Implementations are registered beside
    the class they are for, which is also why nothing above the clients has to
    know they exist.

    Raises:
        NotImplementedError: if nothing is registered for this kind of Tesseract.
    """
    del logs
    raise NotImplementedError(
        f"No diagnose_exit is registered for {type(served).__name__}. Register one "
        "with `@diagnose_exit.register` in the module that defines the class, "
        "returning an empty string if there is nothing to add."
    )


@singledispatch
def is_running(served: ServedTesseract) -> bool:
    """Whether this Tesseract is running now, asking again rather than recalling.

    Dispatched for the same reason as :func:`diagnose_exit`: docker-py has no
    such method, and asks you to compare `status` yourself after a `reload`.

    Raises:
        NotImplementedError: if nothing is registered for this kind of Tesseract.
    """
    raise NotImplementedError(
        f"No is_running is registered for {type(served).__name__}. Register one "
        "with `@is_running.register` in the module that defines the class."
    )


class PortInUseError(RuntimeError):
    """Container failed to start because its port was already bound.

    Signals that a fresh port should be picked and startup retried. Only raised
    when we chose the port ourselves; a user-supplied port is never retried.

    A port collision surfaces in one of two ways depending on network mode:
    - port-mapping mode: the Docker daemon fails to publish the host port and
      ``containers.run`` raises ``ContainerError`` ("port is already allocated").
    - host networking: the container binds the host port directly, so the
      failure appears in the container logs as uvicorn's "address already in
      use" and is detected in ``wait_for_health_or_dispose``.
    """


# Substrings container runtimes use to report a host port already being taken.
PORT_CONFLICT_MARKERS = (
    "address already in use",
    "port is already allocated",
    # Podman machine's wording when its port-forward proxy holds the port
    "proxy already running",
    # Windows' wording for the same condition (WSAEADDRINUSE / WinError 10048)
    "only one usage of each socket address",
)


def get_free_port(
    within_range: tuple[int, int] = (49152, 65535),
    exclude: Sequence[int] = (),
) -> int:
    """Find a random free port to use for HTTP."""
    start, end = within_range
    if start < 0 or end > 65535 or start > end:
        raise ValueError("Invalid port range, must be between 0 and 65535")

    # Try random ports in the given range
    portlist = list(range(start, end))
    random.shuffle(portlist)
    for port in portlist:
        if port in exclude:
            continue
        # Check if the port is free
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                # Port is already in use
                continue
            else:
                return port
    raise RuntimeError(f"No free ports found in range {start}-{end}")


def validate_output_format(
    output_format: str | None, output_path: str | Path | None
) -> None:
    """Reject an output format the given output path cannot support.

    ``json+binref`` writes array buffers to .bin sidecar files instead of
    inlining them, so it needs somewhere durable to put them. Without an output
    path they land wherever the Tesseract happens to be running -- inside a
    container that is about to be torn down, or loose in a subprocess's working
    directory -- and the paths in the response lead nowhere the caller picked.

    Shared so that the check fires the same way, with the same error type,
    however the Tesseract is served: a caller that switches between the two has
    no reason to catch two different exceptions.

    Raises:
        UserError: if the combination cannot produce output the caller can read.
    """
    if output_format == "json+binref" and output_path is None:
        raise UserError(
            "The 'json+binref' output format writes array buffers to .bin files, "
            "which are lost unless an output path is set. Specify one with "
            "--output-path (or output_path=...)."
        )


def runtime_config_env(runtime_config: Mapping[str, Any] | None) -> dict[str, str]:
    """Convert runtime configuration to the variables the Tesseract runtime reads.

    Shared so that ``runtime_config`` means the same thing however a Tesseract is
    served; booleans in particular have to be spelled the way the config parser
    expects.
    """

    def encode(value: Any) -> str:
        return ("1" if value else "0") if isinstance(value, bool) else str(value)

    return {
        f"TESSERACT_{key.upper()}": encode(value)
        for key, value in (runtime_config or {}).items()
    }


def is_port_conflict(stderr: str) -> bool:
    """Whether runtime stderr/logs indicate a host port collision."""
    # Collapse whitespace before matching. An uncaught error in the runtime is
    # rendered by rich into a fixed-width box, which wraps long lines -- and
    # debugpy's "Address already in use" is long enough to be split across two,
    # so a naive substring search silently misses a genuine conflict.
    lowered = " ".join(stderr.split()).lower()
    return any(marker in lowered for marker in PORT_CONFLICT_MARKERS)


def retry_or_raise_port_conflict(
    port: str, auto_port: bool, attempt: int, max_attempts: int
) -> None:
    """Decide whether a port collision should be retried.

    Returns normally if the caller should retry with a fresh port; raises
    otherwise. A user-supplied fixed port is never retried (we must not
    silently move the Tesseract elsewhere), and auto-selected ports raise once
    the attempt budget is exhausted.
    """
    if not auto_port:
        # User asked for this exact port; surface the collision as-is.
        raise PortInUseError(f"Port {port} was already in use")
    if attempt + 1 >= max_attempts:
        raise RuntimeError(
            f"Failed to find a free port after {max_attempts} attempts"
        ) from None
    logger.info(f"Port {port} was taken, retrying with a new port...")


# How long to give a single /health request before assuming it will not answer.
# A published port whose target is unreachable is accepted by the proxy and then
# dropped, so without this a poll can block indefinitely.
_HEALTH_REQUEST_TIMEOUT = 5.0
_HEALTH_POLL_INTERVAL = 0.1


def wait_for_health_or_dispose(
    served: ServedTesseract, url: str, timeout: float = DEFAULT_STARTUP_TIMEOUT
) -> None:
    """Wait for a Tesseract to serve /health, and dispose of it if it never does.

    Takes ``url`` rather than using ``served.url``: a container that published its
    port on every interface is not reached at the address it reports binding to.

    Raises:
        PortInUseError: if it failed because its port was taken, which the caller
            may want to retry on a fresh one.
        TimeoutError: if it never answered in time.
        RuntimeError: if it stopped running before it could.
    """
    deadline = time.monotonic() + timeout
    timed_out = False

    while True:
        try:
            response = requests.get(f"{url}/health", timeout=_HEALTH_REQUEST_TIMEOUT)
        except requests.exceptions.RequestException:
            pass
        else:
            if response.status_code == 200:
                return

        # /health did not answer, so we check for a dead Tesseract first, timeouts second
        if not is_running(served):
            break
        if time.monotonic() > deadline:
            timed_out = True
            break

        time.sleep(_HEALTH_POLL_INTERVAL)

    # Read the logs before disposing of what wrote them. Neither reading nor
    # disposing may raise: they are how we report the failure, not the failure
    # itself, and an error here would replace it with a less useful one.
    #
    # Broad on purpose, and narrow in scope to match: each transport fails in
    # its own vocabulary -- a container reports `APIError`, a subprocess whose
    # log file is gone reports `FileNotFoundError` -- and naming them here would
    # put one transport's spelling in the module that exists to not have any.
    # Only the call itself is guarded, so a mistake of ours still raises.
    try:
        raw_logs = served.logs()
    except Exception as ex:
        logger.warning(f"Failed to get logs for {served}: {ex}")
        raw_logs = b""
    logs = raw_logs.decode(errors="replace")

    # Only worth asking about one that stopped, and only before it is removed: a
    # Tesseract that is merely slow would block `wait` for as long as it runs.
    exit_code = None
    if not timed_out:
        try:
            status = served.wait(timeout=_HEALTH_REQUEST_TIMEOUT)
        except Exception as ex:
            logger.warning(f"Failed to read the exit code of {served}: {ex}")
        else:
            exit_code = status["StatusCode"]

    # Everything the Tesseract knew has now been read, so it can go -- in a
    # `finally`, because every path out of here raises and none of them should
    # leave it behind. The port-collision one especially: it is retried, so a
    # container per attempt would pile up.
    try:
        # A port collision is racy and worth retrying with a fresh port;
        # distinguish it from genuine startup failures so those still fail fast.
        if is_port_conflict(logs):
            raise PortInUseError(f"Port {served.host_port} was already in use")

        if timed_out:
            headline = f"{served} did not respond to a health check in time."
            diagnosis = (
                "If it is simply slow to initialize (e.g. loading a large model), "
                "increase `startup_timeout`."
            )
        else:
            exited = "" if exit_code is None else f" (exit code {exit_code})"
            headline = f"{served} stopped running during startup{exited}."
            diagnosis = diagnose_exit(served, logs)
        output = (
            f"Output from the Tesseract:\n{logs.strip()}"
            if logs.strip()
            else "The Tesseract produced no output."
        )
        paragraphs = [headline, diagnosis, output]
        message = "\n\n".join(p for p in paragraphs if p)
        raise TimeoutError(message) if timed_out else RuntimeError(message)
    finally:
        try:
            # Forced: it may still be running, and an unforced remove would refuse.
            served.remove(force=True)
        except Exception as ex:
            # Broadest of the three: an exception raised in a `finally` replaces
            # the one in flight, so anything at all escaping here would destroy
            # the failure we came to report.
            logger.warning(f"Failed to remove {served}: {ex}")
