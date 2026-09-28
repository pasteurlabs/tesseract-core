# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import logging
import sys

import pytest
from rich.console import Console

from tesseract_core.sdk.logs import RichLogger, set_logger


def test_rich_logger_traceback_not_truncated_when_piped():
    """Tracebacks should not be hard-wrapped when output is piped to a file."""
    buf = io.StringIO()
    # Simulate piped output: force_terminal=False gives width=80 (Rich default)
    console = Console(file=buf, force_terminal=False)
    assert not console.is_terminal

    handler = RichLogger(console=console, level=logging.DEBUG, rich_tracebacks=True)
    handler.setFormatter(logging.Formatter("{message}", style="{"))

    long_msg = "x" * 200

    try:
        raise RuntimeError(long_msg)
    except RuntimeError:
        exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="tesseract",
            level=logging.ERROR,
            pathname=__file__,
            lineno=0,
            msg="an error",
            args=(),
            exc_info=exc_info,
        )
        handler.emit(record)

    output = buf.getvalue()
    # The full error message must appear on a single line, not wrapped at 80 chars
    assert f"RuntimeError: {long_msg}" in output
    matching_lines = [
        line for line in output.splitlines() if f"RuntimeError: {long_msg}" in line
    ]
    assert len(matching_lines) == 1, (
        f"Expected the full error message on one line, but got:\n{output}"
    )


@pytest.mark.parametrize("msg", [{}, {"key": "value"}, [1, 2, 3], 42])
def test_logger_accepts_nonstring_messages(msg):
    logger = set_logger("INFO", rich_format=True)
    # Should not raise — vanilla logging accepts any object as msg
    logger.info(msg)
