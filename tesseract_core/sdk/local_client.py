# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serving a Tesseract as a local subprocess, without containerization.

The process counterpart to :mod:`tesseract_core.sdk.docker_client`: what a
started Tesseract is, how to talk to it, read it and dispose of it -- and, since
there is only the one way to start one, how to do that too. Instead of running
``tesseract-runtime serve`` inside a container, :func:`serve` runs it as a child
of the current interpreter. The Tesseract is still reached over HTTP, so the
client side is identical to the containerized case.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .serving import (
    DEFAULT_STARTUP_TIMEOUT,
    PortInUseError,
    diagnose_exit,
    get_free_port,
    is_running,
    retry_or_raise_port_conflict,
    runtime_config_env,
    validate_output_format,
    wait_for_health_or_dispose,
)

logger = logging.getLogger("tesseract")

# Names the read end of a pipe the parent holds open, so a served Tesseract can
# tell when it has been orphaned. Read by `tesseract-runtime serve`.
PARENT_PIPE_ENV_VAR = "TESSERACT_PARENT_PIPE_FD"

# How long to give a child process to exit on SIGTERM before escalating.
_TERMINATE_TIMEOUT = 10.0


def parent_watch_pipe() -> tuple[int | None, int | None]:
    """A pipe a child can watch to notice it has been orphaned.

    Returns the read end to hand over and the write end to hold, or a pair of
    Nones where that is not possible. `subprocess` will not pass a descriptor to
    a child on Windows, so there a Tesseract can still outlive its parent, as it
    always could.
    """
    if os.name != "posix":
        return None, None
    read_fd, write_fd = os.pipe()
    os.set_inheritable(read_fd, True)
    return read_fd, write_fd


def popen_kwargs() -> dict[str, Any]:
    """Platform-specific options to isolate a child in its own process group.

    Two reasons to do this: a Ctrl-C in the parent's terminal must not race us to
    the child (we want to shut it down in an orderly way ourselves), and on
    removal we need to be able to kill uvicorn's worker processes along with the
    parent it spawned them from. Paired with :func:`_stop_process`, which relies
    on the group existing.
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


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


@dataclass
class TesseractProcess:
    """A ``tesseract-runtime serve`` process running on the local host.

    The process counterpart to :class:`~tesseract_core.sdk.docker_client.Container`.
    """

    process: subprocess.Popen
    host_ip: str
    port: int
    log_path: Path
    python_executable: str
    api_path: Path
    # Our end of the pipe the child watches. Held open for as long as the child
    # should live; closing it is what tells the child we are gone.
    parent_pipe_write_fd: int | None = None

    @property
    def url(self) -> str:
        """Base URL the Tesseract is serving on."""
        return f"http://{self.host_ip}:{self.host_port}"

    def __str__(self) -> str:
        """Name this Tesseract in a message meant for a person."""
        return f"Tesseract at {self.api_path}"

    @property
    def host_port(self) -> str:
        """Port the Tesseract can be reached on."""
        return str(self.port)

    def reload(self) -> None:
        """Nothing to do: this handle holds the process, not a copy of its state."""

    def wait(self, timeout: float | None = None) -> dict:
        """Wait for the child to exit, and report the code it exited with.

        Raises:
            TimeoutError: if it is still running when ``timeout`` expires, which
                is what a container that outlasts a `docker wait` reports too.
        """
        try:
            return {"StatusCode": self.process.wait(timeout=timeout)}
        except subprocess.TimeoutExpired as ex:
            raise TimeoutError(f"{self} was still running after {timeout}s") from ex

    def remove(self, v: bool = False, link: bool = False, force: bool = False) -> None:
        """Stop the process and remove the file its output was captured in.

        ``v`` and ``link`` are accepted and ignored: they are Docker's notion of
        volumes and links, which a process does not have. They are here so that
        this and a container can be disposed of by the same call.

        Safe to call more than once, and on one that has already exited. Refuses
        a running process unless ``force`` is set, as removing a running
        container does -- the caller has to mean it either way.
        """
        if is_running(self) and not force:
            raise RuntimeError(
                f"{self} is still running. Pass force=True to stop and remove it."
            )
        self._stop()
        if self.parent_pipe_write_fd is not None:
            try:
                os.close(self.parent_pipe_write_fd)
            except OSError:
                pass
            self.parent_pipe_write_fd = None
        try:
            self.log_path.unlink(missing_ok=True)
        except OSError:
            # Windows refuses to delete a file another process still holds open,
            # and a just-killed child may not have released it yet. Harmless to
            # leave: it is in a temp directory.
            logger.debug("Could not remove log file %s", self.log_path)

    def logs(self) -> bytes:
        """Everything the process has written to stdout and stderr so far.

        Bytes, matching what a container's logs give us, so callers holding the
        interface need not care which they have. The file is created before the
        process is, so a read that fails is a real one, not an empty Tesseract.
        """
        return self.log_path.read_bytes()

    def _stop(self) -> None:
        """Terminate the child, escalating to a kill if it outlasts the grace period."""
        if self.process.poll() is not None:
            return

        try:
            _stop_process(self.process, force=False)
            try:
                self.process.wait(timeout=_TERMINATE_TIMEOUT)
                return
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Tesseract process %s did not exit within %ss, killing it",
                    self.process.pid,
                    _TERMINATE_TIMEOUT,
                )
            _stop_process(self.process, force=True)
            self.process.wait(timeout=_TERMINATE_TIMEOUT)
        except (ProcessLookupError, PermissionError):
            # Already gone, or not ours to signal anymore (pid reuse).
            pass
        except subprocess.TimeoutExpired:
            logger.warning("Tesseract process %s could not be killed", self.process.pid)


@diagnose_exit.register
def _(served: TesseractProcess, logs: str) -> str:
    """Name the interpreter that ran it, which nothing else can see."""
    if "No module named 'tesseract_core'" in logs:
        return (
            f"The environment running it ({served.python_executable}) does not "
            "have Tesseract installed. Install it there with "
            "`uv pip install tesseract-core[runtime]`."
        )
    return (
        "This usually means `tesseract_api.py` raised at import time, or a "
        "dependency it needs is missing from the environment running it "
        f"({served.python_executable})."
    )


@is_running.register
def _(served: TesseractProcess) -> bool:
    """Whether the child process is still alive."""
    return served.process.poll() is None


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

    validate_output_format(output_format, output_path)

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
            # something we set. Under both names, in typer's order of precedence:
            # we never write this one, so an inherited TESSERACT_RUNTIME_ value is
            # what the runtime will bind, and reporting the TESSERACT_ one (or the
            # default) would send the user's debugger to the wrong address.
            debugpy_host = (
                env.get("TESSERACT_RUNTIME_DEBUGPY_HOST")
                or env.get("TESSERACT_DEBUGPY_HOST")
                or "127.0.0.1"
            )
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

        # A pipe whose read end the child watches and whose write end we hold, so
        # that a Tesseract outlives us only for as long as it takes to notice. It
        # sees EOF however we go -- returning, crashing, or being killed outright
        # -- which is the case cleanup in `remove` cannot cover.
        #
        # POSIX only: `pass_fds` is not supported on Windows, where handing a
        # descriptor to a child means inheriting handles wholesale. A Windows
        # Tesseract can still be orphaned, as it could before this.
        watch_read, watch_write = parent_watch_pipe()
        pass_fds = (watch_read,) if watch_read is not None else ()
        watch_env = (
            {PARENT_PIPE_ENV_VAR: str(watch_read)} if watch_read is not None else {}
        )

        try:
            process = subprocess.Popen(
                command,
                env={**env, **watch_env},
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                pass_fds=pass_fds,
                **popen_kwargs(),
            )
        except BaseException:
            if watch_write is not None:
                os.close(watch_write)
            raise
        finally:
            # Both are the child's now: the descriptor it logs to, and the end of
            # the pipe it watches.
            os.close(log_fd)
            if watch_read is not None:
                os.close(watch_read)

        served = TesseractProcess(
            process=process,
            host_ip=host_ip,
            port=chosen_port,
            log_path=log_path,
            python_executable=python_executable,
            api_path=api_path,
            parent_pipe_write_fd=watch_write,
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
