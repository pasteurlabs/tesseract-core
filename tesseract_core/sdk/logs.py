# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
import threading
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.traceback import Traceback

DEFAULT_CONSOLE = Console(stderr=True)

LEVEL_PREFIX = {
    "DEBUG": " [dim]\\[+][/] ",
    "INFO": " [dim]\\[[/][blue]i[/][dim]][/] ",
    "WARNING": " \\[[yellow]![/]] ",
    "ERROR": " [red]\\[-][/] ",
    "CRITICAL": " [red reverse]\\[x][/] ",
}


# NOTE: This is duplicated in `tesseract_core/runtime/logs.py`.
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


class LogStreamer(threading.Thread):
    """Tail a log file and stream new lines to a sink.

    Runs as a daemon thread, polling the file for new content. Used to provide
    live log streaming from Tesseract runs.

    Example:
        >>> streamer = LogStreamer(Path("/tmp/run_123/logs/tesseract.log"), print)
        >>> streamer.start()
        >>> # ... run some code that writes to the log file ...
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
            poll_interval: How often to poll for new content (seconds).
        """
        super().__init__()
        self._path = Path(path)
        self._sink = sink
        self._current_poll_interval = self._default_poll_interval
        self._stop_event = threading.Event()
        self._file_pos = 0

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
            # Stopped before file appeared
            return

        with open(self._path, encoding="utf-8", errors="replace") as f:
            while True:
                # Read any new content
                new_content = f.read()

                if new_content:
                    line_buffer += new_content

                    # Process complete lines
                    while "\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n", 1)
                        self._sink(line)

                    self._current_poll_interval = (
                        self._default_poll_interval
                    )  # Reset to default after first read
                else:
                    self._current_poll_interval = min(
                        self._current_poll_interval * 2, 0.1
                    )  # Exponential backoff up to 100ms

                # Check if we should stop
                if self._stop_event.is_set():
                    # Drain any remaining content
                    new_content = f.read()
                    if new_content:
                        line_buffer += new_content

                    # Flush remaining lines
                    while "\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n", 1)
                        self._sink(line)

                    # Flush any trailing content without newline
                    if line_buffer:
                        self._sink(line_buffer)

                    break

                time.sleep(self._current_poll_interval)


class RichLogger(logging.Handler):
    """A logging handler that uses rich to render logs and exceptions.

    This is a pared-down version of `rich.logging.RichHandler` that only applies styling without
    additional features like word wrapping.
    """

    def __init__(
        self,
        console: Console,
        level: int | str = logging.NOTSET,
        rich_tracebacks: bool = True,
        tracebacks_suppress: Iterable[str | ModuleType] = (),
    ) -> None:
        super().__init__(level=level)
        self._console = console
        self._rich_tracebacks = rich_tracebacks
        self._tracebacks_suppress = tracebacks_suppress

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        _exc_info = None
        if self._rich_tracebacks:
            _exc_info = record.exc_info
            # Prevent printing the traceback twice
            record.exc_info = None

        log_line = self.format(record)
        self._console.print(log_line, markup=True, soft_wrap=True)

        if self._rich_tracebacks and _exc_info:
            self._console.print(
                Traceback.from_exception(*_exc_info, suppress=self._tracebacks_suppress)
            )


def set_logger(
    level: str, catch_warnings: bool = False, rich_format: bool | None = None
) -> logging.Logger:
    """Initialize loggers."""
    level = level.upper()

    package_logger = logging.getLogger("tesseract")
    package_logger.setLevel("DEBUG")  # send everything to handlers

    if rich_format is None:
        rich_format = DEFAULT_CONSOLE.is_terminal

    if rich_format:
        ch = RichLogger(
            console=DEFAULT_CONSOLE,
            level=level,
            rich_tracebacks=True,
            tracebacks_suppress=[typer],
        )

        class PrefixFormatter(logging.Formatter):
            def format(self, record: Any, *args: Any) -> Any:
                record.levelprefix = LEVEL_PREFIX.get(record.levelname, "")
                record.msg = escape(record.msg)
                return super().format(record, *args)

        fmt = "{levelprefix!s}{message!s}"
        ch_fmt = PrefixFormatter(fmt, style="{")
    else:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        fmt = "{asctime} [{levelname}] {message}"
        ch_fmt = logging.Formatter(fmt, style="{")

    ch.setFormatter(ch_fmt)
    package_logger.handlers = [ch]

    if catch_warnings:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.handlers = [ch]
        warnings_logger.setLevel(level)

        def custom_formatwarning(
            message: str,
            category: Any,
            filename: str,
            lineno: int,
            line: str | None = None,
        ) -> str:
            if rich_format:
                out = f"[yellow]{category.__name__}[/]: {message}"
            else:
                out = f"{category.__name__}: {message}"
            return out

        warnings.formatwarning = custom_formatwarning

    return package_logger


def set_loglevel(level: str) -> None:
    """Update the log level of all loggers."""
    level = level.upper()
    package_logger = logging.getLogger("tesseract")
    for handler in package_logger.handlers:
        handler.setLevel(level)
