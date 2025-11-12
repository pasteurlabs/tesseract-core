# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import time
from typing import Any, Callable


# NOTE: This is duplicated in `tesseract_core/sdk/logs.py`.
# Make sure to propagate changes to both files.
class TeePipe(threading.Thread):
    """Custom I/O pipe to support live logging to multiple sinks.

    Runs a thread that logs everything written to the pipe to the given sinks.
    Can be used as a context manager for automatic cleanup.
    """

    daemon = True

    def __init__(self, *sinks: Callable) -> None:
        """Initialize the TeePipe by creating file descriptors."""
        super().__init__()
        self._sinks = sinks
        self._fd_read, self._fd_write = os.pipe()
        self._pipe_reader = os.fdopen(
            self._fd_read, mode="r", closefd=False, buffering=1024
        )
        self._captured_lines = []
        self._last_line_time = time.time()

    def __enter__(self) -> int:
        """Start the thread and return the write file descriptor of the pipe."""
        self.start()
        return self.fileno()

    def stop(self) -> None:
        """Close the pipe and join the thread."""
        # Wait for ongoing reads to complete
        grace = 0.1
        while (time.time() - self._last_line_time) < grace:
            time.sleep(grace / 10)
        os.close(self._fd_read)
        # Use timeout and daemon=True to avoid hanging indefinitely if something goes wrong
        self.join(timeout=1)
        self._pipe_reader.close()
        os.close(self._fd_write)

    def __exit__(self, *args: Any) -> None:
        """Close the pipe and join the thread."""
        self.stop()

    def fileno(self) -> int:
        """Return the write file descriptor of the pipe."""
        return self._fd_write

    def run(self) -> None:
        """Run the thread, logging everything."""
        line_buffer = []
        while True:
            try:
                data = self._pipe_reader.readline(1024)
            except OSError:
                # Pipe closed
                break
            if data == "":
                # EOF reached
                break

            self._last_line_time = time.time()
            if data.endswith("\n"):
                data = data[:-1]
                flush = True
            else:
                flush = False

            line_buffer.append(data)
            if flush:
                line = "".join(line_buffer)
                line_buffer.clear()
                self._captured_lines.append(line)
                for sink in self._sinks:
                    sink(line)

        # Flush incomplete lines at the end of the stream
        if line_buffer:
            line = "".join(line_buffer)
            self._captured_lines.append(line)
            for sink in self._sinks:
                sink(line)

    @property
    def captured_lines(self) -> list[str]:
        """Return all lines captured so far."""
        return self._captured_lines
