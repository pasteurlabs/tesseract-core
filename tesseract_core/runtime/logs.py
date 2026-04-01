# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def is_tracing_enabled() -> bool:
    """Check if tracing mode is enabled."""
    from .config import get_config

    return get_config().tracing


def get_logger() -> logging.Logger:
    """Get the runtime logger, initializing it if needed.

    Returns a logger that outputs timestamped messages to stdout.
    Log level is DEBUG when tracing is enabled, INFO otherwise.
    """
    from .config import get_config

    logger = logging.getLogger("tesseract_runtime")
    logger.setLevel(logging.DEBUG)  # Allow all levels, handler controls output

    level = logging.DEBUG if get_config().tracing else logging.INFO

    # Only add handler if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        handler = logger.handlers[0]

    handler.setLevel(level)
    return logger


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
        # Close the write end. This signals EOF to reader.
        # After this, no more data can be written to the pipe, and the reader
        # will eventually see EOF after draining the kernel pipe buffer.
        try:
            os.close(self._fd_write)
        except OSError:
            pass  # Already closed

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


# NOTE: This is duplicated in `tesseract_core/sdk/logs.py`.
# Make sure to propagate changes to both files.
class LogStreamer(threading.Thread):
    """Tail a log file and stream new lines to a sink.

    Runs as a daemon thread, polling the file for new content.  Used to
    provide live log streaming from Tesseract runs.

    Unlike :class:`TeePipe`, this reads from a regular file rather than a
    pipe, so writers (including foreign-language threads) can never block.

    Example:
        >>> streamer = LogStreamer(Path("/tmp/tesseract.log"), print)
        >>> streamer.start()
        >>> # ... code writes to the log file ...
        >>> streamer.stop()  # Drains remaining content and stops
    """

    daemon = True
    _default_poll_interval = 0.001

    def __init__(
        self,
        path: Path | str,
        sink: Callable[[str], Any],
    ) -> None:
        """Initialize the LogStreamer.

        Args:
            path: Path to the log file to tail.
            sink: Callable that receives each log line.
        """
        super().__init__()
        self._path = Path(path)
        self._sink = sink
        self._current_poll_interval = self._default_poll_interval
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to drain remaining content."""
        self._stop_event.set()
        self.join()

    def run(self) -> None:
        """Poll the log file and send new lines to the sink."""
        line_buffer = ""

        # Wait for file to appear (with polling)
        while not self._stop_event.is_set() and not self._path.exists():
            time.sleep(self._current_poll_interval)

        if not self._path.exists():
            return

        with open(self._path, encoding="utf-8", errors="replace") as f:
            while True:
                new_content = f.read()

                if new_content:
                    line_buffer += new_content

                    while "\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n", 1)
                        self._sink(line)

                    self._current_poll_interval = self._default_poll_interval
                else:
                    self._current_poll_interval = min(
                        self._current_poll_interval * 2, 0.1
                    )

                if self._stop_event.is_set():
                    # Drain any remaining content
                    new_content = f.read()
                    if new_content:
                        line_buffer += new_content

                    while "\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n", 1)
                        self._sink(line)

                    if line_buffer:
                        self._sink(line_buffer)

                    break

                time.sleep(self._current_poll_interval)
