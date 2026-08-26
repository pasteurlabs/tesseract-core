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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from .local_client import TesseractProcess, popen_kwargs
from .serving import (
    DEFAULT_STARTUP_TIMEOUT,
    PortInUseError,
    get_free_port,
    retry_or_raise_port_conflict,
    runtime_config_env,
    wait_for_health_or_dispose,
)

logger = logging.getLogger("tesseract")

# Number of times to retry startup with a fresh port when the one we picked was
# taken between selection and bind. Mirrors the containerized serve path.
_MAX_PORT_ATTEMPTS = 3


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
) -> TesseractProcess:
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
                **popen_kwargs(),
            )
        finally:
            # The child holds its own duplicate of the descriptor.
            os.close(log_fd)

        served = TesseractProcess(
            process=process,
            host_ip=host_ip,
            port=chosen_port,
            log_path=log_path,
            python_executable=python_executable,
            api_path=api_path,
        )

        if skip_health_check:
            return served

        try:
            wait_for_health_or_dispose(served, served.url, startup_timeout)
        except PortInUseError:
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
            retry_or_raise_port_conflict(
                conflicting, retriable, attempt, _MAX_PORT_ATTEMPTS
            )
            continue

        return served

    raise RuntimeError(
        f"Failed to find a free port after {_MAX_PORT_ATTEMPTS} attempts"
    )
