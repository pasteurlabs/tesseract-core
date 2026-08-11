# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve Tesseracts as local subprocesses, without containerization.

This is the non-containerized counterpart to :mod:`tesseract_core.sdk.engine`:
instead of running ``tesseract-runtime serve`` inside a Docker container, it runs
it as a child process of the current interpreter. The Tesseract is still reached
over HTTP, so the client side is identical to the containerized case.

Compared to importing ``tesseract_api.py`` in-process, this buys process
isolation (the Tesseract gets its own interpreter, its own global state, and its
own signal handlers) at the cost of HTTP round-trips. It does *not* provide any
of the other isolation a container gives you: the child inherits the parent's
environment, working directory, filesystem access, and user.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import requests

from .engine import (
    _is_port_conflict,
    _PortInUseError,
    _retry_or_raise_port_conflict,
    get_free_port,
    runtime_config_env,
)

logger = logging.getLogger("tesseract")

# Number of times to retry startup with a fresh port when the one we picked was
# taken between selection and bind. Mirrors the containerized serve path.
_MAX_PORT_ATTEMPTS = 3

# How long to wait for a freshly spawned Tesseract to answer /health.
DEFAULT_STARTUP_TIMEOUT = 60.0

# How long to give a child process to exit on SIGTERM before escalating.
_TERMINATE_TIMEOUT = 10.0


@dataclass
class ServedTesseract:
    """A ``tesseract-runtime serve`` process running on the local host."""

    process: subprocess.Popen
    host_ip: str
    port: int
    log_path: Path
    python_executable: str

    @property
    def url(self) -> str:
        """Base URL the Tesseract is serving on."""
        return f"http://{self.host_ip}:{self.port}"

    def is_running(self) -> bool:
        """Whether the child process is still alive."""
        return self.process.poll() is None


def _runtime_env(
    api_path: Path,
    *,
    input_path: str | Path | None,
    output_path: str | Path | None,
    output_format: str | None,
    runtime_config: dict[str, Any] | None,
    environment: dict[str, str] | None,
    foreign_interpreter: bool,
) -> dict[str, str]:
    """Build the child's environment.

    Runtime configuration is passed as ``TESSERACT_*`` environment variables
    rather than by mutating this process's runtime config, which is what makes
    it possible to run several differently-configured Tesseracts side by side.
    """
    env = dict(os.environ)

    if foreign_interpreter:
        # Our own interpreter's import paths are meaningless (at best) and
        # actively harmful (at worst) to a different one: they would put this
        # environment's site-packages ahead of the Tesseract's own, defeating
        # the point of running it elsewhere. Note that importing a
        # tesseract_api.py in-process sets PYTHONPATH as a side effect, so this
        # is not a hypothetical.
        for var in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(var, None)

    # Applied after the scrub above, so an explicit request always wins.
    if environment:
        env.update(environment)

    env["TESSERACT_API_PATH"] = str(api_path)

    def mirror(key: str, value: str) -> None:
        """Write a setting under both names the runtime accepts.

        Unlike a container, the child inherits this process's environment, so a
        setting exported by the caller is already there. Writing only
        ``TESSERACT_*`` would leave an inherited ``TESSERACT_RUNTIME_*`` in place,
        and typer resolves that as a CLI option, which wins -- so an exported
        TESSERACT_RUNTIME_DEBUGPY_PORT would put every Tesseract on the same debug
        port and the second would fail to start.
        """
        env[key] = value
        env[key.replace("TESSERACT_", "TESSERACT_RUNTIME_", 1)] = value

    for key, value in runtime_config_env(runtime_config).items():
        mirror(key, value)

    if input_path is not None:
        mirror("TESSERACT_INPUT_PATH", str(Path(input_path).resolve()))
    if output_path is not None:
        mirror("TESSERACT_OUTPUT_PATH", str(Path(output_path).resolve()))
    if output_format is not None:
        mirror("TESSERACT_OUTPUT_FORMAT", output_format)

    # Without this the child's logs arrive in chunks, which makes streaming them
    # useless and startup failures look like hangs.
    env["PYTHONUNBUFFERED"] = "1"

    return env


def _popen_kwargs() -> dict[str, Any]:
    """Platform-specific options to isolate the child in its own process group.

    Two reasons to do this: a Ctrl-C in the parent's terminal must not race us to
    the child (we want to shut it down in an orderly way ourselves), and on
    teardown we need to be able to kill uvicorn's worker processes along with the
    parent it spawned them from.
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _read_logs(log_path: Path) -> str:
    """Read a child's captured output, tolerating a not-yet-created file."""
    try:
        return log_path.read_text(errors="replace")
    except OSError:
        return ""


def _describe_startup_failure(
    served: ServedTesseract, timed_out: bool, api_path: Path
) -> str:
    """Build an actionable error message for a Tesseract that never came up."""
    logs = _read_logs(served.log_path).strip()
    returncode = served.process.poll()

    if timed_out:
        headline = f"Tesseract at {api_path} did not respond to a health check in time."
        hint = (
            "If it is simply slow to initialize (e.g. loading a large model), "
            "increase `startup_timeout`."
        )
    else:
        headline = (
            f"Tesseract at {api_path} exited during startup (exit code {returncode})."
        )
        if "No module named 'tesseract_core'" in logs:
            hint = (
                f"The environment running it ({served.python_executable}) does "
                "not have Tesseract installed. Install it there with "
                "`uv pip install tesseract-core[runtime]`."
            )
        else:
            hint = (
                "This usually means `tesseract_api.py` raised at import time, or "
                "a dependency it needs is missing from the environment running "
                f"it ({served.python_executable})."
            )

    parts = [headline, hint]
    if logs:
        parts.append(f"Output from the Tesseract process:\n{logs}")
    else:
        parts.append("The Tesseract process produced no output.")
    return "\n\n".join(parts)


def _wait_for_health(served: ServedTesseract, api_path: Path, timeout: float) -> None:
    """Poll a subprocess Tesseract's /health until it answers 200.

    Raises:
        _PortInUseError: if the child failed because its port was taken.
        RuntimeError: if the child died or never became healthy in time.
    """
    deadline = time.monotonic() + timeout

    while True:
        # Check liveness before the request so a crashed child fails fast
        # instead of waiting out the full timeout.
        died = not served.is_running()

        try:
            response = requests.get(f"{served.url}/health", timeout=5)
        except requests.exceptions.RequestException:
            pass
        else:
            if response.status_code == 200:
                return

        timed_out = time.monotonic() > deadline
        if died or timed_out:
            logs = _read_logs(served.log_path)
            # A port collision is racy and worth retrying with a fresh port;
            # distinguish it from genuine startup failures so those fail fast.
            if _is_port_conflict(logs):
                terminate(served)
                raise _PortInUseError(f"Port {served.port} was already in use")

            message = _describe_startup_failure(served, timed_out, api_path)
            terminate(served)
            raise RuntimeError(message)

        time.sleep(0.05)


def serve(
    api_path: str | Path,
    *,
    host_ip: str = "127.0.0.1",
    port: int | str | None = None,
    num_workers: int = 1,
    environment: dict[str, str] | None = None,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    output_format: Literal["json", "json+base64", "json+binref"] | None = None,
    runtime_config: dict[str, Any] | None = None,
    python_executable: str | Path | None = None,
    skip_health_check: bool = False,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
) -> ServedTesseract:
    """Serve a ``tesseract_api.py`` in a dedicated subprocess.

    Args:
        api_path: Path to the ``tesseract_api.py`` to serve.
        host_ip: IP address to bind to.
        port: Port to bind to. If None, a free port is picked automatically
            (and re-picked if it gets taken before the server binds it).
        num_workers: Number of uvicorn worker processes.
        environment: Extra environment variables for the child process. These are
            layered on top of the parent's environment, not a replacement for it.
        input_path: Value for ``TESSERACT_INPUT_PATH``.
        output_path: Value for ``TESSERACT_OUTPUT_PATH``.
        output_format: Value for ``TESSERACT_OUTPUT_FORMAT``.
        runtime_config: Runtime configuration options, converted to
            ``TESSERACT_*`` environment variables just as in the containerized
            path.
        python_executable: Interpreter used to run the Tesseract. Defaults to the
            one running this process; pointing it at another environment's
            ``python`` is what allows a Tesseract to have dependencies that
            conflict with the caller's.
        skip_health_check: If True, return as soon as the process is spawned
            without waiting for it to answer /health. The caller is then
            responsible for establishing readiness.
        startup_timeout: How long to wait for the health check, in seconds.

    Returns:
        The served Tesseract, to be passed to :func:`teardown` when done.
    """
    api_path = Path(api_path).resolve()
    if not api_path.is_file():
        raise FileNotFoundError(f"Tesseract API path {api_path} is not a file.")

    if python_executable is None:
        python_executable = sys.executable
    python_executable = str(python_executable)

    foreign_interpreter = os.path.realpath(python_executable) != os.path.realpath(
        sys.executable
    )
    if foreign_interpreter and not os.path.isfile(python_executable):
        raise FileNotFoundError(
            f"Python interpreter {python_executable} does not exist."
        )

    if output_format == "json+binref" and output_path is None:
        raise ValueError(
            "output_path is required when using the 'json+binref' output format."
        )

    auto_port = port is None

    for attempt in range(_MAX_PORT_ATTEMPTS):
        chosen_port = int(get_free_port()) if auto_port else int(port)

        # Debug mode always starts a debugger, so give each Tesseract its own
        # port: host processes share a network namespace, unlike containers, so
        # the default would collide on the second Tesseract. The host it binds
        # is left to the runtime, which already defaults to loopback. Opt out of
        # debug mode entirely with `runtime_config={"debug": False}`.
        attempt_config = dict(runtime_config or {})
        if attempt_config.get("debug"):
            attempt_config.setdefault(
                "debugpy_port", get_free_port(exclude=(chosen_port,))
            )

        env = _runtime_env(
            api_path,
            input_path=input_path,
            output_path=output_path,
            output_format=output_format,
            runtime_config=attempt_config,
            environment=environment,
            foreign_interpreter=foreign_interpreter,
        )

        if attempt_config.get("debug"):
            # The runtime reports this too, but into the captured log file, which
            # nobody sees unless they go looking. Read the host back out of the
            # environment, since it may be the runtime's default rather than
            # something we set.
            debugpy_host = env.get("TESSERACT_DEBUGPY_HOST", "127.0.0.1")
            logger.info(
                "Debug mode enabled. Attach a debugger to "
                f"{debugpy_host}:{attempt_config['debugpy_port']}"
            )

        # A dedicated file rather than a pipe: nothing has to keep draining it to
        # stop a chatty Tesseract from filling the pipe buffer and blocking, and
        # it gives `server_logs()` something to read after teardown.
        log_fd, log_name = tempfile.mkstemp(prefix="tesseract_serve_", suffix=".log")
        log_path = Path(log_name)

        command = [
            python_executable,
            "-m",
            "tesseract_core.runtime",
            "serve",
            "--host",
            host_ip,
            "--port",
            str(chosen_port),
            "--num-workers",
            str(num_workers),
        ]

        logger.debug("Serving Tesseract %s on port %s", api_path, chosen_port)

        try:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                **_popen_kwargs(),
            )
        finally:
            # The child holds its own duplicate of the descriptor.
            os.close(log_fd)

        served = ServedTesseract(
            process=process,
            host_ip=host_ip,
            port=chosen_port,
            log_path=log_path,
            python_executable=python_executable,
        )

        if skip_health_check:
            return served

        try:
            _wait_for_health(served, api_path, startup_timeout)
        except _PortInUseError:
            # Retry as long as at least one of the ports was ours to pick. The
            # logs say a port was taken but not which, and the debug port is
            # chosen automatically even when the caller pins the API port, so
            # keying the decision on the API port alone would refuse to retry a
            # collision we caused and can trivially resolve.
            debugpy_port = attempt_config.get("debugpy_port")
            we_chose_debugpy_port = debugpy_port is not None and "debugpy_port" not in (
                runtime_config or {}
            )
            retriable = auto_port or we_chose_debugpy_port
            conflicting = (
                f"{chosen_port} or {debugpy_port}"
                if debugpy_port is not None
                else str(chosen_port)
            )
            _retry_or_raise_port_conflict(
                conflicting, retriable, attempt, _MAX_PORT_ATTEMPTS
            )
            continue

        return served

    raise RuntimeError(
        f"Failed to find a free port after {_MAX_PORT_ATTEMPTS} attempts"
    )


def _stop_process(process: subprocess.Popen, *, force: bool) -> None:
    """Ask a child and everything it spawned to exit, or force it to.

    ``signal.SIGKILL`` does not exist on Windows, so it must not be named outside
    the POSIX branch -- not even to compare against.
    """
    if os.name == "nt":
        # No process groups in the POSIX sense; these map to the Windows APIs
        # for asking a process to stop and for terminating it outright.
        process.kill() if force else process.terminate()
        return

    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        # The group, so uvicorn's workers go down with the parent that spawned them.
        os.killpg(os.getpgid(process.pid), sig)
    except ProcessLookupError:
        # Group is gone; fall back to the process in case it outlived it.
        process.send_signal(sig)


def terminate(served: ServedTesseract) -> None:
    """Stop a served Tesseract, forcing it if it does not exit in time.

    Safe to call more than once, and on a process that has already exited.
    """
    process = served.process
    if process.poll() is not None:
        return

    try:
        _stop_process(process, force=False)
        try:
            process.wait(timeout=_TERMINATE_TIMEOUT)
            return
        except subprocess.TimeoutExpired:
            logger.warning(
                "Tesseract process %s did not exit within %ss, killing it",
                process.pid,
                _TERMINATE_TIMEOUT,
            )
        _stop_process(process, force=True)
        process.wait(timeout=_TERMINATE_TIMEOUT)
    except (ProcessLookupError, PermissionError):
        # Already gone, or not ours to signal anymore (pid reuse).
        pass
    except subprocess.TimeoutExpired:
        logger.warning("Tesseract process %s could not be killed", process.pid)


def teardown(served: ServedTesseract, keep_logs: bool = False) -> str:
    """Stop a served Tesseract and return its captured logs.

    Args:
        served: The Tesseract to stop.
        keep_logs: If True, leave the log file on disk instead of removing it.

    Returns:
        Everything the Tesseract wrote to stdout/stderr during its lifetime.
    """
    terminate(served)
    logs_text = _read_logs(served.log_path)
    if not keep_logs:
        try:
            served.log_path.unlink(missing_ok=True)
        except OSError:
            # Windows refuses to delete a file another process still holds open,
            # and a just-killed child may not have released it yet. Harmless to
            # leave: it is in a temp directory.
            logger.debug("Could not remove log file %s", served.log_path)
    return logs_text


def logs(served: ServedTesseract) -> str:
    """Read the logs a served Tesseract has produced so far."""
    return _read_logs(served.log_path)
