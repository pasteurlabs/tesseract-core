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

from .served_client import ServedTesseract

logger = logging.getLogger("tesseract")

# How long to give a child process to exit on SIGTERM before escalating.
_TERMINATE_TIMEOUT = 10.0


def popen_kwargs() -> dict[str, Any]:
    """Platform-specific options to isolate a child in its own process group.

    Two reasons to do this: a Ctrl-C in the parent's terminal must not race us to
    the child (we want to shut it down in an orderly way ourselves), and on
    teardown we need to be able to kill uvicorn's worker processes along with the
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
class TesseractProcess(ServedTesseract):
    """A ``tesseract-runtime serve`` process running on the local host.

    The process counterpart to :class:`~tesseract_core.sdk.docker_client.Container`.
    """

    process: subprocess.Popen
    host_ip: str
    port: int
    log_path: Path
    python_executable: str
    api_path: Path

    def __repr__(self) -> str:
        return f"Tesseract at {self.api_path}"

    @property
    def host_port(self) -> str:
        """Port the Tesseract can be reached on."""
        return str(self.port)

    def reload(self) -> None:
        """Nothing to do: this handle holds the process, not a copy of its state."""

    def exit_code(self) -> int | None:
        """The code the child exited with, if it has."""
        return self.process.poll()

    def is_running(self) -> bool:
        """Whether the child process is still alive."""
        return self.process.poll() is None

    def teardown(self) -> None:
        """Stop the process, forcing it if it does not go quietly, and clean up.

        Safe to call more than once, and on a process that has already exited.
        """
        self._stop()
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

    def diagnose_exit(self, logs: str) -> str:
        """Name the interpreter that ran it, which nothing else can see."""
        if "No module named 'tesseract_core'" in logs:
            return (
                f"The environment running it ({self.python_executable}) does not "
                "have Tesseract installed. Install it there with "
                "`uv pip install tesseract-core[runtime]`."
            )
        return (
            "This usually means `tesseract_api.py` raised at import time, or a "
            "dependency it needs is missing from the environment running it "
            f"({self.python_executable})."
        )

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
