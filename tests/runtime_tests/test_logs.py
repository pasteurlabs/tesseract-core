# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import random
import string
import time

from tesseract_core.runtime.logs import TeePipe


def test_teepipe(caplog):
    # Verify that logging in a separate thread works as intended
    logger = logging.getLogger("tesseract")
    logger.setLevel("DEBUG")

    ch = logging.StreamHandler()
    ch.setLevel("INFO")
    ch.setFormatter(logging.Formatter("{asctime} [{levelname}] {message}", style="{"))
    logger.handlers = [ch]

    caplog.set_level(logging.INFO, logger="tesseract")

    logged_lines = []
    for _ in range(100):
        # Make sure to include a few really long lines without breaks
        if random.random() < 0.1:
            msg_length = random.randint(1000, 10_000)
            alphabet = string.ascii_letters + "\U0001f92f"
        else:
            msg_length = 2 ** random.randint(2, 12)
            alphabet = string.printable + "\U0001f92f"
        msg = "".join(random.choices(alphabet, k=msg_length))
        logged_lines.append(msg)

    teepipe = TeePipe(logger.info)
    # Extend grace period to avoid flakes in tests when runners are slow
    teepipe._grace_period = 1
    with teepipe:
        fd = os.fdopen(teepipe.fileno(), "w", closefd=False)
        for line in logged_lines:
            print(line, file=fd)
            time.sleep(random.random() / 100)
        fd.close()

    expected_lines = []
    for line in logged_lines:
        sublines = line.split("\n")
        expected_lines.extend(sublines)

    assert teepipe.captured_lines == expected_lines
    assert caplog.record_tuples == [
        ("tesseract", logging.INFO, line) for line in expected_lines
    ]


def test_teepipe_drains_buffer():
    # Verify that TeePipe drains all buffered data before stop() returns.
    # This simulates the real usage pattern where writes complete, then stop() is called.
    logged_lines = []
    for _ in range(100):
        # Make sure to include a few really long lines without breaks
        if random.random() < 0.1:
            msg_length = random.randint(1000, 10_000)
            alphabet = string.ascii_letters + "\U0001f92f"
        else:
            msg_length = 2 ** random.randint(2, 12)
            alphabet = string.printable + "\U0001f92f"
        msg = "".join(random.choices(alphabet, k=msg_length))
        logged_lines.append(msg)

    teepipe = TeePipe()
    teepipe.start()
    fd = os.fdopen(teepipe.fileno(), "w", closefd=False)

    # Write all data
    for line in logged_lines:
        print(line, file=fd, flush=True)

    print("end without newline", end="", file=fd, flush=True)

    expected_lines = []
    for line in logged_lines:
        sublines = line.split("\n")
        expected_lines.extend(sublines)
    expected_lines.append("end without newline")

    # Now stop - this should drain all buffered data
    teepipe.stop()

    assert len(teepipe.captured_lines) == len(expected_lines)
    assert teepipe.captured_lines == expected_lines
