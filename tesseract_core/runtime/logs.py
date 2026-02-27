# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import threading
from collections.abc import Callable
from typing import Any


# NOTE: This is duplicated in `tesseract_core/sdk/logs.py`.
# Make sure to propagate changes to both files.
class TeePipe(threading.Thread):
    """Custom I/O construct to support live logging from a single file descriptor to multiple sinks.

    Runs a thread that records everything written to the file descriptor. Can be used as a
    context manager for automatic cleanup.

    Example:
        >>> with TeePipe(print, logger.info) as pipe_fd:
        ...     fd = os.fdopen(pipe_fd, "w")
        ...     print("Hello, World!", file=fd, flush=True)
        Hello, World!
        2025-06-10 12:00:00,000 - INFO - Hello, World!
    """

    daemon = True

    def __init__(self, *sinks: Callable) -> None:
        """Initialize the TeePipe by creating file descriptors."""
        super().__init__()
        self._sinks = sinks
        self._fd_read, self._fd_write = os.pipe()
        self._captured_lines: list[str] = []
        self._fd_write_closed = False

    def __enter__(self) -> int:
        """Start the thread and return the write file descriptor of the pipe."""
        self.start()
        return self.fileno()

    def stop(self) -> None:
        """Close the pipe and wait for the reader thread to drain all data.

        This method is safe to call even if the write fd has already been closed
        (e.g., via dup2 replacing it). Closing the write end signals EOF to the
        reader, which will then drain any remaining data from the pipe buffer
        before exiting.
        """
        # Close the write end if not already closed. This signals EOF to reader.
        # After this, no more data can be written to the pipe, and the reader
        # will eventually see EOF after draining the kernel pipe buffer.
        if not self._fd_write_closed:
            try:
                os.close(self._fd_write)
            except OSError:
                pass  # Already closed
            self._fd_write_closed = True

        # Wait for reader thread to finish. It will exit after hitting EOF
        # and draining all buffered data. No timeout needed since EOF is
        # guaranteed once write end is closed.
        self.join()

        # Now safe to close read end - reader thread has exited
        try:
            os.close(self._fd_read)
        except OSError:
            pass  # Already closed

    def __exit__(self, *args: Any) -> None:
        """Close the pipe and join the thread."""
        self.stop()

    def fileno(self) -> int:
        """Return the write file descriptor of the pipe."""
        return self._fd_write

    def run(self) -> None:
        """Run the thread, pushing every full line of text to the sinks."""
        line_buffer: list[bytes] = []
        while True:
            try:
                data = os.read(self._fd_read, 4096)
            except OSError:
                # Read fd was closed externally
                break

            if data == b"":
                # EOF reached - write end is closed and buffer is drained
                break

            lines = data.split(b"\n")

            # Log complete lines
            for i, line in enumerate(lines[:-1]):
                if i == 0:
                    line = b"".join([*line_buffer, line])
                    line_buffer = []
                decoded = line.decode(errors="ignore")
                self._captured_lines.append(decoded)
                for sink in self._sinks:
                    sink(decoded)

            # Accumulate incomplete line
            line_buffer.append(lines[-1])

        # Flush incomplete lines at the end of the stream
        remaining = b"".join(line_buffer)
        if remaining:
            decoded = remaining.decode(errors="ignore")
            self._captured_lines.append(decoded)
            for sink in self._sinks:
                sink(decoded)

    @property
    def captured_lines(self) -> list[str]:
        """Return all lines captured so far."""
        return self._captured_lines
