# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
import threading
from collections.abc import Callable
from typing import Any

# Simple logging setup for the runtime module
_logger: logging.Logger | None = None


def is_tracing_enabled() -> bool:
    """Check if tracing mode is enabled."""
    from .config import get_config

    return get_config().tracing


def get_logger() -> logging.Logger:
    """Get the runtime logger, initializing it if needed.

    Returns a logger that outputs timestamped messages to stdout.
    Log level is DEBUG when tracing is enabled, INFO otherwise.
    """
    global _logger
    if _logger is not None:
        return _logger

    from .config import get_config

    _logger = logging.getLogger("tesseract_runtime")
    _logger.setLevel(logging.DEBUG)  # Allow all levels, handler controls output

    # Only add handler if not already configured
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        level = logging.DEBUG if get_config().tracing else logging.INFO
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)

    return _logger


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
        self._captured_lines = []
        self._last_time = time.time()
        self._is_blocking = threading.Event()
        self._grace_period = 0.1

    def __enter__(self) -> int:
        """Start the thread and return the write file descriptor of the pipe."""
        self.start()
        return self.fileno()

    def stop(self) -> None:
        """Close the pipe and join the thread."""
        # Wait for ongoing streams to dry up
        # We only continue once the reader has spent some time blocked on reading
        while True:
            self._is_blocking.wait(timeout=1)
            if (time.time() - self._last_time) >= self._grace_period:
                break
            time.sleep(self._grace_period / 10)

        # This will signal EOF to the reader thread
        os.close(self._fd_write)
        os.close(self._fd_read)

        # Use timeout and daemon=True to avoid hanging indefinitely if something goes wrong
        self.join(timeout=1)

    def __exit__(self, *args: Any) -> None:
        """Close the pipe and join the thread."""
        self.stop()

    def fileno(self) -> int:
        """Return the write file descriptor of the pipe."""
        return self._fd_write

    def run(self) -> None:
        """Run the thread, pushing every full line of text to the sinks."""
        line_buffer = []
        while True:
            self._last_time = time.time()
            self._is_blocking.set()
            try:
                data = os.read(self._fd_read, 1024)
                self._is_blocking.clear()
            except OSError:
                # Pipe closed
                break

            if data == b"":
                # EOF reached
                break

            lines = data.split(b"\n")

            # Log complete lines
            for i, line in enumerate(lines[:-1]):
                if i == 0:
                    line = b"".join([*line_buffer, line])
                    line_buffer = []
                line = line.decode(errors="ignore")
                self._captured_lines.append(line)
                for sink in self._sinks:
                    sink(line)

            # Accumulate incomplete line
            line_buffer.append(lines[-1])

        # Flush incomplete lines at the end of the stream
        line = b"".join(line_buffer)
        if line:
            line = line.decode(errors="ignore")
            self._captured_lines.append(line)
            for sink in self._sinks:
                sink(line)

    @property
    def captured_lines(self) -> list[str]:
        """Return all lines captured so far."""
        return self._captured_lines
