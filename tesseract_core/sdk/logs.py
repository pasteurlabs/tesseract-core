# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
import threading
import time
import warnings
from collections.abc import Iterable
from types import ModuleType
from typing import Any, Callable

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

        # This will signal EOF to the reader thread
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
