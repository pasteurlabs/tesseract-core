# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI tests that do not require a running Docker daemon.

(Those go in endtoend_tests/test_endtoend.py.)
"""

import pytest
from typer.testing import CliRunner

from tesseract_core.sdk.cli import app as cli


@pytest.fixture
def cli_runner():
    return CliRunner(mix_stderr=False)


def test_suggestion_on_misspelled_command(cli_runner):
    result = cli_runner.invoke(cli, ["innit"], catch_exceptions=False)
    assert result.exit_code == 2, result.stdout
    assert "No such command 'innit'." in result.stderr
    assert "Did you mean 'init'?" in result.stderr

    result = cli_runner.invoke(cli, ["wellbloodygreatinnit"], catch_exceptions=False)
    assert result.exit_code == 2, result.stdout
    assert "No such command 'wellbloodygreatinnit'." in result.stderr
    assert "Did you mean" not in result.stderr


def test_version(cli_runner):
    from tesseract_core import __version__

    result = cli_runner.invoke(cli, ["--version"])
    assert result.exit_code == 0, result.stdout
    assert __version__ in result.stdout
