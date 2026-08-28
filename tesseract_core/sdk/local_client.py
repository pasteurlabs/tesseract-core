# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A handle on a Tesseract served as a local subprocess.

The process counterpart to :mod:`tesseract_core.sdk.docker_client`: what a
started Tesseract is, and how to talk to it, read it and dispose of it. Deciding
to start one is :mod:`tesseract_core.sdk.local_engine`'s job, so nothing here
imports from either engine.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .served_client import diagnose_exit, is_running

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
