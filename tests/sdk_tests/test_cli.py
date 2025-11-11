# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI tests that do not require a running Docker daemon.

(Those go in endtoend_tests/test_endtoend.py.)
"""

import os
import subprocess

import pytest

from tesseract_core.sdk.cli import app as cli


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


@pytest.mark.parametrize(
    "arg_to_override",
    [
        "name",
        "build_config.custom_build_steps",
        "build_config.base_image",
        "build_config.package_data",
    ],
)
def test_config_override(
    arg_to_override, cli_runner, mocker, dummy_tesseract_location, mocked_docker
):
    mocked_build = mocker.patch("tesseract_core.sdk.engine.build_tesseract")

    def _run_with_override(key, value):
        return cli_runner.invoke(
            cli,
            [
                "build",
                str(dummy_tesseract_location),
                "--config-override",
                f"{key}={value}",
                "--generate-only",
            ],
            catch_exceptions=False,
        )

    if arg_to_override == "name":
        argpairs = (
            (
                "my-tesseract",
                {("name",): "my-tesseract"},
            ),
        )
    elif arg_to_override == "build_config.custom_build_steps":
        argpairs = (
            (
                "[RUN foo='bar']",
                {("build_config", "custom_build_steps"): ["RUN foo='bar'"]},
            ),
            (
                '[RUN echo "hello world"]',
                {("build_config", "custom_build_steps"): ['RUN echo "hello world"']},
            ),
        )
    elif arg_to_override == "build_config.base_image":
        argpairs = (
            (
                "ubuntu:latest",
                {("build_config", "base_image"): "ubuntu:latest"},
            ),
        )
    elif arg_to_override == "build_config.package_data":
        argpairs = (
            (
                '["data/file.txt:/app/data/file.txt"]',
                {
                    ("build_config", "package_data"): [
                        "data/file.txt:/app/data/file.txt"
                    ]
                },
            ),
        )
    else:
        raise ValueError(f"Unknown arg_to_override: {arg_to_override}")

    for value, expected in argpairs:
        result = _run_with_override(arg_to_override, value)
        assert result.exit_code == 0, result.stderr
        assert mocked_build.call_args[1]["config_override"] == expected
