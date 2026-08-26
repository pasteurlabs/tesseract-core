# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Primitives shared by everything that serves a Tesseract over HTTP.

Choosing a port to serve on, telling the Tesseract how it is configured, waiting
for it to answer, and telling a lost race for a port apart from a Tesseract that
is genuinely broken. None of it is particular to how the Tesseract is run, so none
of it belongs in a module that knows how to run one.
"""

import logging
import random
import socket
import time
from collections.abc import Mapping, Sequence
from contextlib import closing
from typing import Any

import requests

from .docker_client import APIError
from .served_client import ServedTesseract

logger = logging.getLogger("tesseract")

# How long to wait for a freshly started Tesseract to answer /health. Generous,
# because the cost of being wrong is asymmetric: a Tesseract that crashes is
# reported the moment it stops running, so this only bounds one that hangs.
DEFAULT_STARTUP_TIMEOUT = 60.0


class _PortInUseError(RuntimeError):
    """Container failed to start because its port was already bound.

    Signals that a fresh port should be picked and startup retried. Only raised
    when we chose the port ourselves; a user-supplied port is never retried.

    A port collision surfaces in one of two ways depending on network mode:
    - port-mapping mode: the Docker daemon fails to publish the host port and
      ``containers.run`` raises ``ContainerError`` ("port is already allocated").
    - host networking: the container binds the host port directly, so the
      failure appears in the container logs as uvicorn's "address already in
      use" and is detected in ``_wait_for_health``.
    """


# Substrings container runtimes use to report a host port already being taken.
_PORT_CONFLICT_MARKERS = (
    "address already in use",
    "port is already allocated",
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


def _is_port_conflict(stderr: str) -> bool:
    """Whether runtime stderr/logs indicate a host port collision."""
    # Collapse whitespace before matching. An uncaught error in the runtime is
    # rendered by rich into a fixed-width box, which wraps long lines -- and
    # debugpy's "Address already in use" is long enough to be split across two,
    # so a naive substring search silently misses a genuine conflict.
    lowered = " ".join(stderr.split()).lower()
    return any(marker in lowered for marker in _PORT_CONFLICT_MARKERS)


def _retry_or_raise_port_conflict(
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
        raise _PortInUseError(f"Port {port} was already in use")
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


def _wait_for_health(served: ServedTesseract, url: str, timeout: float) -> None:
    """Wait for a Tesseract to serve /health, and dispose of it if it never does.

    Takes ``url`` rather than using ``served.url``: a container that published its
    port on every interface is not reached at the address it reports binding to.

    Raises:
        _PortInUseError: if it failed because its port was taken, which the caller
            may want to retry on a fresh one.
        TimeoutError: if it never answered in time.
        RuntimeError: if it stopped running before it could.
    """
    deadline = time.monotonic() + timeout

    while True:
        try:
            response = requests.get(f"{url}/health", timeout=_HEALTH_REQUEST_TIMEOUT)
        except requests.exceptions.RequestException:
            pass
        else:
            if response.status_code == 200:
                return

        # Liveness is only asked once /health has failed to answer, so a healthy
        # Tesseract never pays for it -- worth having, as for a container it is a
        # `docker inspect` every poll. Checked before the deadline: "it exited"
        # tells whoever asked more than "it did not answer in time", and a
        # Tesseract that has exited is never going to answer.
        if not served.is_running():
            timed_out = False
            break
        if time.monotonic() > deadline:
            timed_out = True
            break

        time.sleep(_HEALTH_POLL_INTERVAL)

    # Read the logs before disposing of what wrote them. Neither reading nor
    # disposing may raise: they are how we report the failure, not the failure
    # itself, and an error here would replace it with a less useful one. Both
    # implementations raise `APIError` and nothing else, but a handle holding an
    # operating system resource is entitled to an `OSError`.
    try:
        logs = served.logs().decode(errors="replace")
    except (APIError, OSError) as ex:
        logger.warning(f"Failed to get logs for {served}: {ex}")
        logs = ""

    try:
        served.teardown()
    except (APIError, OSError) as ex:
        logger.warning(f"Failed to tear down {served}: {ex}")

    # A port collision is racy and worth retrying with a fresh port; distinguish
    # it from genuine startup failures so those still fail fast.
    if _is_port_conflict(logs):
        raise _PortInUseError(f"Port {served.host_port} was already in use")

    # Only what the Tesseract itself can add differs between transports; naming
    # it, saying which of the two ways it failed, and quoting its output do not.
    if timed_out:
        headline = f"{served} did not respond to a health check in time."
        # Advice neither transport can improve on: both take the same argument.
        diagnosis = (
            "If it is simply slow to initialize (e.g. loading a large model), "
            "increase `startup_timeout`."
        )
    else:
        exit_code = served.exit_code()
        exited = "" if exit_code is None else f" (exit code {exit_code})"
        headline = f"{served} stopped running during startup{exited}."
        diagnosis = served.diagnose_exit(logs)
    output = (
        f"Output from the Tesseract:\n{logs.strip()}"
        if logs.strip()
        else "The Tesseract produced no output."
    )
    paragraphs = [headline, diagnosis, output]
    message = "\n\n".join(p for p in paragraphs if p)
    raise TimeoutError(message) if timed_out else RuntimeError(message)
