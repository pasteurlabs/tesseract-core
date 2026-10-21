# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI tests that do not require a running Docker daemon.

(Those go in endtoend_tests/.)
"""

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import requests

import tesseract_core.sdk.cli as cli_module
from tesseract_core.sdk import engine, local_client
from tesseract_core.sdk.cli import app as cli
from tesseract_core.sdk.exceptions import UserError

EXAMPLES = Path(__file__).parents[2] / "examples"

DUMMY_PAYLOAD = json.dumps({"inputs": {"a": [1.0, 2.0], "b": [3.0, 4.0]}})


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


@pytest.fixture
def docker_is_off_limits(monkeypatch):
    """Fail the test if anything asks for Docker."""

    def check_docker():
        raise AssertionError("Docker was required")

    monkeypatch.setattr(engine, "check_docker", check_docker)


def _cli_env(**extra: str) -> dict[str, str]:
    """Environment for a `tesseract` subprocess, with output we can match on."""
    return {**os.environ, "TERM": "dumb", "COLUMNS": "1000", **extra}


def _without_docker() -> dict[str, str]:
    """Point the CLI at a Docker that always fails, so needing it is an error."""
    false = shutil.which("false")
    if false is None:
        pytest.skip("needs a `false` executable to stand in for a broken Docker")
    return {"TESSERACT_DOCKER_EXECUTABLE": false}


@pytest.mark.parametrize(
    "argument, expected",
    [
        ("tess", "tess/tesseract_api.py"),
        ("./tess", "tess/tesseract_api.py"),
        ("tess/tesseract_api.py", "tess/tesseract_api.py"),
        ("tess/other_api.py", "tess/other_api.py"),
        ("no_api", None),
        ("data.txt", None),
        ("missing", None),
        ("missing.py", None),
        ("ghcr.io/org/tess:1.0", None),
    ],
)
def test_source_path_or_image_name(tmp_path, monkeypatch, argument, expected):
    """A path counts as a source only if it is a Python file or a directory with a tesseract_api.py."""
    (tmp_path / "tess").mkdir()
    (tmp_path / "tess" / "tesseract_api.py").touch()
    (tmp_path / "tess" / "other_api.py").touch()
    (tmp_path / "no_api").mkdir()
    (tmp_path / "data.txt").touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "_image_exists", lambda name: False)

    result = cli_module._get_source_path(argument)

    if expected is None:
        assert result is None
    else:
        assert result == (tmp_path / expected).resolve()


def test_source_path_wins_over_an_image_of_the_same_name(tmp_path, monkeypatch, caplog):
    (tmp_path / "tess").mkdir()
    (tmp_path / "tess" / "tesseract_api.py").touch()
    monkeypatch.chdir(tmp_path)
    asked = []
    monkeypatch.setattr(
        cli_module, "_image_exists", lambda name: asked.append(name) or True
    )

    with caplog.at_level(logging.WARNING, logger="tesseract"):
        result = cli_module._get_source_path("tess")
    assert result == (tmp_path / "tess" / "tesseract_api.py").resolve()
    assert asked == ["tess"]
    assert "both a Tesseract source directory and an image" in caplog.text
    assert "'tess:latest' to use the image" in caplog.text

    # An explicit path cannot be an image name, so there is nothing to ask.
    caplog.clear()
    asked.clear()
    assert cli_module._get_source_path("./tess") == result
    assert asked == []
    assert "both a Tesseract source directory and an image" not in caplog.text


def test_image_exists_without_docker(monkeypatch):
    """Probing for an image never fails, whatever state Docker is in."""

    def missing_docker(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(cli_module.docker_client.images, "get", missing_docker)
    assert not cli_module._image_exists("helloworld")


RUN_CONTAINER_OPTIONS = [
    ("--volume", ["--volume", "/tmp:/data"]),
    ("--gpus", ["--gpus", "all"]),
    ("--network", ["--network", "host"]),
    ("--user", ["--user", "1000"]),
    ("--memory", ["--memory", "1g"]),
    ("--docker-args", ["--docker-args", "--cpus=1"]),
]

SERVE_CONTAINER_OPTIONS = [
    ("--volume", ["--volume", "/tmp:/data"]),
    ("--network", ["--network", "host"]),
    ("--network-alias", ["--network", "net", "--network-alias", "tess"]),
    ("--gpus", ["--gpus", "all"]),
    ("--user", ["--user", "1000"]),
    ("--memory", ["--memory", "1g"]),
    ("--runtime-args", ["--runtime-args", "--cpus=1"]),
]


@pytest.mark.parametrize(
    "command, option, extra_args",
    [("run", option, args) for option, args in RUN_CONTAINER_OPTIONS]
    + [("serve", option, args) for option, args in SERVE_CONTAINER_OPTIONS],
)
def test_container_options_are_refused_for_a_source_path(
    cli_runner,
    dummy_tesseract_package,
    docker_is_off_limits,
    monkeypatch,
    command,
    option,
    extra_args,
):
    def must_not_start(*args, **kwargs):
        raise AssertionError("started a Tesseract despite a container-only option")

    monkeypatch.setattr(local_client, "run", must_not_start)
    monkeypatch.setattr(local_client, "serve", must_not_start)

    argv = [command, str(dummy_tesseract_package)]
    if command == "run":
        argv += ["apply", DUMMY_PAYLOAD]
    result = cli_runner.invoke(cli, [*argv, *extra_args])

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, UserError)
    assert f"{option} cannot be used with a Tesseract run from source" in str(
        result.exception
    )


@pytest.mark.parametrize("command", ["run", "serve"])
def test_python_executable_is_refused_for_an_image(
    cli_runner, docker_is_off_limits, command
):
    argv = [command, "some-image"]
    if command == "run":
        argv += ["apply", "{}"]
    result = cli_runner.invoke(cli, [*argv, "--python-executable", sys.executable])

    assert result.exit_code == 1, result.output
    assert "--python-executable only applies to a Tesseract run from source" in str(
        result.exception
    )


@pytest.mark.parametrize("command", ["run", "serve"])
def test_an_image_still_needs_docker(cli_runner, tmp_path, monkeypatch, command):
    """Anything that is not a source path is an image, and needs Docker as before."""

    def no_docker():
        raise UserError("Could not reach Docker daemon, check if it is running.")

    monkeypatch.setattr(engine, "check_docker", no_docker)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "not-a-tesseract").mkdir()

    for name, hinted in (("some-image", False), ("not-a-tesseract", True)):
        argv = [command, name]
        if command == "run":
            argv += ["apply", "{}"]
        result = cli_runner.invoke(cli, argv)

        assert result.exit_code == 1, result.output
        assert "Could not reach Docker daemon" in str(result.exception)
        assert ("is not a Tesseract source" in str(result.exception)) is hinted


@pytest.mark.parametrize("command", ["run", "serve"])
def test_an_image_name_still_goes_to_a_container(
    cli_runner, mocked_docker, mocker, tmp_path, monkeypatch, command
):
    monkeypatch.chdir(tmp_path)
    if command == "run":
        started = mocker.patch.object(
            engine, "run_tesseract", return_value=('{"ok": true}', "")
        )
        argv = ["run", "vectoradd", "apply", "{}"]
    else:
        started = mocker.patch.object(
            engine, "serve", return_value=("vectoradd-1", mocker.Mock())
        )
        mocker.patch.object(cli_module, "_display_container_meta", return_value={})
        argv = ["serve", "vectoradd"]

    result = cli_runner.invoke(cli, argv)

    assert result.exit_code == 0, result.output
    assert started.call_args.args[0] == "vectoradd"


def test_run_resolves_the_interpreter_like_from_source(
    cli_runner, dummy_tesseract_package, docker_is_off_limits, monkeypatch
):
    """Without --python-executable, the CLI uses the environment `from_source` would."""
    resolved = []

    def resolve(api_path):
        resolved.append(api_path)
        return Path(sys.executable)

    monkeypatch.setattr(local_client, "resolve_python_executable", resolve)

    result = cli_runner.invoke(
        cli, ["run", str(dummy_tesseract_package), "apply", DUMMY_PAYLOAD]
    )

    assert result.exit_code == 0, result.output
    assert resolved == [(dummy_tesseract_package / "tesseract_api.py").resolve()]
    assert json.loads(result.stdout)["result"]["data"]["buffer"] == [6.0, 10.0]


def test_run_from_source_without_docker(dummy_tesseract_package):
    result = subprocess.run(
        [
            "tesseract",
            "run",
            str(dummy_tesseract_package),
            "apply",
            DUMMY_PAYLOAD,
            "--python-executable",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_cli_env(**_without_docker()),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["result"]["data"]["buffer"] == [6.0, 10.0]


def test_run_from_source_builds_its_environment(dummy_tesseract_package):
    """Without --python-executable, the environment is built next to the API, once."""
    if shutil.which("uv") is None:
        pytest.skip("uv is required to build an environment")

    api_path = dummy_tesseract_package / "tesseract_api.py"
    result = subprocess.run(
        ["tesseract", "run", str(api_path), "apply", DUMMY_PAYLOAD],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_cli_env(),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["result"]["data"]["buffer"] == [6.0, 10.0]
    assert (dummy_tesseract_package / ".tesseract-venv").is_dir()


def test_run_from_source_writes_to_the_output_path(dummy_tesseract_package, tmp_path):
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            "tesseract",
            "run",
            str(dummy_tesseract_package),
            "apply",
            DUMMY_PAYLOAD,
            "--output-path",
            str(output_dir),
            "--output-format",
            "json+binref",
            "--output-file",
            "result.json",
            "--python-executable",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_cli_env(),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    written = json.loads((output_dir / "result.json").read_text())
    buffer = written["result"]["data"]["buffer"].rsplit(":", maxsplit=1)[0]
    assert (output_dir / buffer).is_file()


@pytest.mark.parametrize("passes", [True, False])
def test_check_gradients_from_source(dummy_tesseract_package, passes):
    """The dummy's Jacobian is right and its VJP and JVP are not; the exit code says which ran."""
    runtime_args = ["--runtime-args", "--endpoints jacobian"] if passes else []
    result = subprocess.run(
        [
            "tesseract",
            "run",
            str(dummy_tesseract_package),
            "check-gradients",
            DUMMY_PAYLOAD,
            *runtime_args,
            "--python-executable",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_cli_env(),
        check=False,
    )

    if passes:
        assert result.returncode == 0, result.stderr
        assert "Gradient check for jacobian passed" in result.stderr
    else:
        assert result.returncode == 1, result.stderr
        assert "Gradient check for vector_jacobian_product failed" in result.stderr
        assert "Error running Tesseract." in result.stderr


def test_unimplemented_command_from_source(tmp_path):
    shutil.copytree(EXAMPLES / "helloworld", tmp_path / "helloworld")
    result = subprocess.run(
        [
            "tesseract",
            "run",
            str(tmp_path / "helloworld"),
            "jacobian",
            "{}",
            "--python-executable",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_cli_env(),
        check=False,
    )

    assert result.returncode == 1
    assert "Unimplemented command 'jacobian'" in result.stderr


def test_run_help_from_source(dummy_tesseract_package):
    target = str(dummy_tesseract_package)
    result = subprocess.run(
        [
            "tesseract",
            "run",
            target,
            "apply",
            "--help",
            "--python-executable",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_cli_env(),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"Usage: tesseract run {target} apply" in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="sends SIGINT, as Ctrl+C would")
def test_serve_from_source_until_interrupted(dummy_tesseract_package, free_port):
    url = f"http://127.0.0.1:{free_port}"
    with tempfile.TemporaryFile("w+", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            [
                "tesseract",
                "serve",
                str(dummy_tesseract_package),
                "--port",
                str(free_port),
                "--python-executable",
                sys.executable,
            ],
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            encoding="utf-8",
            env=_cli_env(**_without_docker()),
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                stderr.seek(0)
                if "press Ctrl+C to stop" in stderr.read():
                    break
                assert process.poll() is None, "serve exited early"
                time.sleep(0.1)
            else:
                pytest.fail("serve never reported that it was ready")

            response = requests.post(
                f"{url}/apply", json=json.loads(DUMMY_PAYLOAD), timeout=10
            )
            assert response.status_code == 200, response.text
            assert response.json()["result"]["data"]["buffer"] == [6.0, 10.0]

            process.send_signal(signal.SIGINT)
            stdout, _ = process.communicate(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

        stderr.seek(0)
        logs = stderr.read()

    assert process.returncode == 0, logs
    assert stdout == ""
    assert "Application startup complete" in logs
    with pytest.raises(requests.ConnectionError):
        requests.get(f"{url}/health", timeout=2)


def test_serve_from_source_reports_a_tesseract_that_fails_to_start(tmp_path):
    shutil.copytree(EXAMPLES / "helloworld", tmp_path / "broken")
    with open(tmp_path / "broken" / "tesseract_api.py", "a") as api:
        api.write("\nraise RuntimeError('broken on import')\n")

    result = subprocess.run(
        [
            "tesseract",
            "serve",
            str(tmp_path / "broken"),
            "--python-executable",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_cli_env(),
        check=False,
        timeout=60,
    )

    assert result.returncode == 1
    assert "Could not start the Tesseract at" in result.stderr
    assert "broken on import" in result.stderr
