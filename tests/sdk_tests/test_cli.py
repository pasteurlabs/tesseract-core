# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI tests that do not require a running Docker daemon.

(Those go in endtoend_tests/test_endtoend.py.)
"""

import os
import subprocess

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


def test_bad_docker_executable_env_var():
    env = os.environ.copy()
    env.update({"TESSERACT_DOCKER_EXECUTABLE": "not-a-docker"})

    result = subprocess.run(
        ["tesseract", "ps"],
        env=env,
        check=False,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "Executable `not-a-docker` not found" in result.stderr.decode()
