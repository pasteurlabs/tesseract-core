# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import threading
from typing import Any, Callable


# NOTE: This is duplicated in `tesseract_core/sdk/logs.py`.
# Make sure to propagate changes to both files.
class LogPipe(threading.Thread):
    """Custom IO pipe to support live logging from subprocess.run or OS-level file descriptor.

    Runs a thread that logs everything read from the pipe to the given sinks.
    Can be used as a context manager for automatic cleanup.
    """

    daemon = True

    def __init__(self, *sinks: Callable) -> None:
        """Initialize the LogPipe with the given logging level."""
        super().__init__()
        self._sinks = sinks
        self._fd_read, self._fd_write = os.pipe()
        self._pipe_reader = os.fdopen(self._fd_read)
        self._captured_lines = []

        self._lock = threading.Lock()
        self._closed = False

    def __enter__(self) -> int:
        """Start the thread and return the write file descriptor of the pipe."""
        self.start()
        return self.fileno()

    def __exit__(self, *args: Any) -> None:
        """Close the pipe and join the thread."""
        os.close(self._fd_write)
        # Use a timeout so something weird happening in the logging thread doesn't
        # cause this to hang indefinitely
        self.join(timeout=1)
        # Do not close reader before thread is joined since there may be pending data
        # This also closes the fd_read pipe
        self.close_pipe()

    def close_pipe(self) -> None:
        with self._lock:
            if not self._closed:
                os.close(self._fd_read)
                self._closed = True

    def fileno(self) -> int:
        """Return the write file descriptor of the pipe."""
        return self._fd_write

    def run(self) -> None:
        """Run the thread, logging everything."""
        for line in iter(self._pipe_reader.readline, ""):
            if line.endswith("\n"):
                line = line[:-1]
            self._captured_lines.append(line)
            for sink in self._sinks:
                sink(line)
        self.close_pipe()

    @property
    def captured_lines(self) -> list[str]:
        """Return all lines captured so far."""
        return self._captured_lines
