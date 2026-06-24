# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
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
class LogStreamer(threading.Thread):
    """Tail a log file and stream new lines to a sink.

    Runs as a daemon thread, polling the file for new content.  Used to
    provide live log streaming from Tesseract runs.

    Reads from a regular file rather than a pipe, so writers (including
    foreign-language threads) can never block.

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
