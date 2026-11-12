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
from tesseract_core.sdk.exceptions import UserError


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


def test_init_refuses_existing_files_as_user_error(cli_runner, tmp_path):
    args = ["init", "--name", "foo", "--target-dir", str(tmp_path)]
    assert cli_runner.invoke(cli, args).exit_code == 0

    result = cli_runner.invoke(cli, args)
    assert isinstance(result.exception, UserError), result.exception
    assert "tesseract_api.py already exists" in str(result.exception)


@pytest.mark.parametrize("cmd", ["serve", "run"])
def test_cuda_ipc_not_a_cli_output_format(cli_runner, cmd):
    """cuda_ipc is a GPU transport, not a host output format, and never on the CLI.

    ``json+cuda_ipc`` is not a valid ``--output-format``: it is neither listed in
    --help nor accepted as a value; requesting it fails as an invalid choice.
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


def test_serve_accepts_valid_gpu_transport(cli_runner, mocked_docker, mocker):
    # NO_COLOR ensures Rich isn't wrapping output symbols
    help_result = cli_runner.invoke(cli, ["serve", "--help"], env={"NO_COLOR": "1"})
    assert "GPU transport" in help_result.stdout

    mocker.patch("tesseract_core.sdk.cli._display_container_meta", return_value={})

    command = ["serve", "some-image", "--gpu-transport", "none"]
    result = cli_runner.invoke(cli, command)
    assert result.exit_code == 0

    command = ["serve", "some-image", "--gpu-transport", "wrong"]
    result = cli_runner.invoke(cli, command)
    assert result.exit_code == 2


def test_serve_passes_repeated_gpu_transports(cli_runner, mocker):
    serve = mocker.patch(
        "tesseract_core.sdk.engine.serve", return_value=("name", mocker.Mock())
    )
    mocker.patch("tesseract_core.sdk.cli._display_container_meta", return_value={})

    command = ["serve", "some-image", "--gpu-transport", "cuda_ipc"]
    command += ["--gpu-transport", "nixl"]
    result = cli_runner.invoke(cli, command)
    assert result.exit_code == 0, result.stdout
    assert serve.call_args.kwargs["gpu_transport"] == ["cuda_ipc", "nixl"]

    result = cli_runner.invoke(cli, ["serve", "some-image"])
    assert result.exit_code == 0, result.stdout
    assert serve.call_args.kwargs["gpu_transport"] is None


@pytest.mark.parametrize("port", ["8000:8081", "8000-8080-9000"])
def test_serve_rejects_malformed_port(cli_runner, port):
    result = cli_runner.invoke(cli, ["serve", "some-image", "-p", port])
    assert result.exit_code == 2, result.stdout
    assert "Invalid value for port" in result.stderr


@pytest.mark.parametrize("user", [None, "1000"])
def test_run_test_spec_with_null_cli_config(
    cli_runner, mocked_docker, mocker, tmp_path, user
):
    from tesseract_core.runtime.testing.regression import TestCliConfig, TestSpec

    spec = TestSpec(
        endpoint="apply",
        payload={"inputs": {}},
        expected_outputs={},
        cli_config=None if user is None else TestCliConfig(user=user),
    )
    spec_path = tmp_path / "test_apply.json"
    spec_path.write_text(spec.model_dump_json())
    mocked_run = mocker.patch(
        "tesseract_core.sdk.engine.run_tesseract", return_value=("{}", "")
    )

    result = cli_runner.invoke(
        cli,
        ["run", "some-image", "test", f"@{spec_path}"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr
    assert mocked_run.call_args.kwargs["user"] == user
    assert mocked_run.call_args.kwargs["volumes"] == []
    assert mocked_run.call_args.kwargs["input_path"] is None


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
        encoding="utf-8",
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
        check=False,
    )

    assert result.returncode == 0, result.stderr
    # The two CLIs print help to different streams, and which one is not the point
    # here -- that the module was found and ran under its own name is.
    assert f"python -m {module}" in result.stdout + result.stderr
    # Importing a package should not drag in its own __main__; runpy warns if it
    # has, and a warning on every invocation would land in captured logs.
    assert "RuntimeWarning" not in result.stderr
