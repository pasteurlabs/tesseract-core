# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI tests that do not require a running Docker daemon.

(Those go in endtoend_tests/.)
"""

import os
import subprocess
import sys

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


@pytest.mark.parametrize("cmd", ["serve", "run"])
def test_cuda_ipc_not_a_cli_output_format(cli_runner, cmd):
    """The experimental json+cuda_ipc format is not exposed on the CLI.

    It is neither listed in --help nor accepted as a value; requesting it fails
    as an invalid choice.
    """
    help_result = cli_runner.invoke(cli, [cmd, "--help"])
    assert "json+cuda_ipc" not in help_result.stdout

    command = (
        ["serve", "some-image", "--output-format", "json+cuda_ipc"]
        if cmd == "serve"
        else ["run", "some-image", "apply", "--output-format", "json+cuda_ipc", "{}"]
    )
    result = cli_runner.invoke(cli, command)
    assert result.exit_code == 2, result.stdout


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
        "build_config.requirements.python_version",
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
                {("build_config", "custom_build_steps"): "[RUN foo='bar']"},
            ),
            (
                '[RUN echo "hello world"]',
                {("build_config", "custom_build_steps"): '[RUN echo "hello world"]'},
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
                    ("build_config", "package_data"): (
                        '["data/file.txt:/app/data/file.txt"]'
                    )
                },
            ),
        )
    elif arg_to_override == "build_config.requirements.python_version":
        # Nested keypath into a sub-model (the uv-pip provider settings). Values
        # reach build_tesseract as raw strings and are coerced against the target
        # field's type there (see engine._coerce_config_override), so the quoted
        # value is passed through verbatim.
        argpairs = (
            (
                "'3.12'",
                {("build_config", "requirements", "python_version"): "'3.12'"},
            ),
        )
    else:
        raise ValueError(f"Unknown arg_to_override: {arg_to_override}")

    for value, expected in argpairs:
        result = _run_with_override(arg_to_override, value)
        assert result.exit_code == 0, result.stderr
        assert mocked_build.call_args[1]["config_override"] == expected


@pytest.mark.parametrize("module", ["tesseract_core", "tesseract_core.runtime"])
def test_runnable_as_a_module(module, dummy_tesseract_package):
    """Both packages must be reachable by interpreter path alone.

    The console scripts need their directory on PATH, and `tesseract` collides
    with the OCR engine of the same name; `python -m` sidesteps both. The runtime
    also relies on this to serve a Tesseract from another environment, where only
    the interpreter is known.
    """
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        # The runtime builds its commands from a Tesseract, so it needs one to
        # have something to describe.
        env={
            **os.environ,
            # The runtime builds its commands from a Tesseract, so it needs one
            # to have something to describe.
            "TESSERACT_API_PATH": str(dummy_tesseract_package / "tesseract_api.py"),
            # As elsewhere in the suite: no colour codes chopping up the text we
            # match on, and wide enough that rich does not wrap it either.
            "TERM": "dumb",
            "COLUMNS": "1000",
        },
    )

    assert result.returncode == 0, result.stderr
    # The two CLIs print help to different streams, and which one is not the point
    # here -- that the module was found and ran under its own name is.
    assert f"python -m {module}" in result.stdout + result.stderr
    # Importing a package should not drag in its own __main__; runpy warns if it
    # has, and a warning on every invocation would land in captured logs.
    assert "RuntimeWarning" not in result.stderr
