# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for Tesseract workflows."""

import json
from pathlib import Path

import pytest
import requests
import yaml
from common import build_tesseract, image_exists
from typer.testing import CliRunner

from tesseract_core.sdk.cli import AVAILABLE_RECIPES, app


@pytest.fixture(scope="module")
def built_image_name(docker_client, shared_dummy_image_name, dummy_tesseract_location):
    """Build the dummy Tesseract image for the tests."""
    image_name = build_tesseract(dummy_tesseract_location, shared_dummy_image_name)
    assert image_exists(docker_client, image_name)
    yield image_name


tested_images = ("ubuntu:24.04",)

build_matrix = [
    *[(tag, None, None) for tag in (True, False)],
    *[(False, r, None) for r in AVAILABLE_RECIPES],
    *[(False, None, img) for img in tested_images],
]


@pytest.mark.parametrize("tag,recipe,base_image", build_matrix)
def test_build_from_init_endtoend(
    docker_client, dummy_image_name, tmp_path, tag, recipe, base_image
):
    """Test that a trivial (empty) Tesseract image can be built from init."""
    cli_runner = CliRunner(mix_stderr=False)

    init_args = ["init", "--target-dir", str(tmp_path), "--name", dummy_image_name]
    if recipe:
        init_args.extend(["--recipe", recipe])

    result = cli_runner.invoke(app, init_args, catch_exceptions=False)
    assert result.exit_code == 0, result.stderr
    assert (tmp_path / "tesseract_api.py").exists()
    with open(tmp_path / "tesseract_config.yaml") as config_yaml:
        assert yaml.safe_load(config_yaml)["name"] == dummy_image_name

    img_tag = "foo" if tag else None

    config_override = {}
    if base_image is not None:
        config_override["build_config.base_image"] = base_image

    image_name = build_tesseract(
        tmp_path, dummy_image_name, config_override=config_override, tag=img_tag
    )
    assert image_exists(docker_client, image_name)

    # Test that the image can be run and that --help is forwarded correctly
    result = cli_runner.invoke(
        app,
        [
            "run",
            image_name,
            "apply",
            "--help",
        ],
        catch_exceptions=False,
    )
    assert f"Usage: tesseract run {image_name} apply" in result.stderr


def test_build_generate_only(dummy_tesseract_location):
    """Test output of build with --generate_only flag."""
    cli_runner = CliRunner(mix_stderr=False)
    build_res = cli_runner.invoke(
        app,
        [
            "build",
            str(dummy_tesseract_location),
            "--generate-only",
        ],
        # Ensure that the output is not truncated
        env={"COLUMNS": "1000"},
        catch_exceptions=False,
    )
    assert build_res.exit_code == 0, build_res.stderr
    # Check that stdout contains "docker buildx"
    command = "docker buildx build"
    assert command in build_res.stderr

    build_dir = Path(build_res.stdout.strip())
    assert build_dir.exists()
    dockerfile_path = build_dir / "Dockerfile"
    assert dockerfile_path.exists()


def test_tesseract_list(built_image_name):
    # Test List Command
    cli_runner = CliRunner(mix_stderr=False)

    list_res = cli_runner.invoke(
        app,
        [
            "list",
        ],
        # Ensure that the output is not truncated
        env={"COLUMNS": "1000"},
        catch_exceptions=False,
    )
    assert list_res.exit_code == 0, list_res.stderr
    assert built_image_name.split(":")[0] in list_res.stdout


def test_tesseract_run_stdout(built_image_name):
    # Test List Command
    cli_runner = CliRunner(mix_stderr=False)

    test_commands = ("input-schema", "output-schema", "openapi-schema", "health")

    for command in test_commands:
        run_res = cli_runner.invoke(
            app,
            [
                "run",
                built_image_name,
                command,
            ],
            catch_exceptions=False,
        )
        assert run_res.exit_code == 0, run_res.stderr
        assert run_res.stdout

        try:
            json.loads(run_res.stdout)
        except json.JSONDecodeError:
            print(f"failed to decode {command} stdout as JSON")
            print(run_res.stdout)
            raise


def test_tesseract_serve_pipeline(docker_client, built_image_name):
    cli_runner = CliRunner(mix_stderr=False)
    project_id = None
    try:
        run_res = cli_runner.invoke(
            app,
            [
                "serve",
                built_image_name,
            ],
            catch_exceptions=False,
        )

        assert run_res.exit_code == 0, run_res.stderr
        assert run_res.stdout

        project_meta = json.loads(run_res.stdout)

        project_id = project_meta["project_id"]
        project_containers = project_meta["containers"][0]["name"]
        if not project_containers:
            raise ValueError(f"Could not find container for project '{project_id}'")

        project_container = docker_client.containers.get(project_containers)
        assert project_container.name == project_meta["containers"][0]["name"]
        assert project_container.host_port == project_meta["containers"][0]["port"]

        # Ensure served Tesseract is usable
        res = requests.get(f"http://localhost:{project_container.host_port}/health")
        assert res.status_code == 200, res.text

        # Ensure project id is shown in `tesseract ps`
        run_res = cli_runner.invoke(
            app,
            ["ps"],
            env={"COLUMNS": "1000"},
            catch_exceptions=False,
        )
        assert run_res.exit_code == 0, run_res.stderr
        assert project_id in run_res.stdout
        assert project_container.host_port in run_res.stdout
        assert project_container.short_id in run_res.stdout
    finally:
        if project_id:
            run_res = cli_runner.invoke(
                app,
                [
                    "teardown",
                    project_id,
                ],
                catch_exceptions=False,
            )
            assert run_res.exit_code == 0, run_res.stderr


@pytest.mark.parametrize("tear_all", [True, False])
def test_tesseract_teardown_multiple(built_image_name, tear_all):
    """Teardown multiple projects."""
    cli_runner = CliRunner(mix_stderr=False)

    project_ids = []
    try:
        for _ in range(0, 5):
            # Serve
            run_res = cli_runner.invoke(
                app,
                [
                    "serve",
                    built_image_name,
                ],
                catch_exceptions=False,
            )
            assert run_res.exit_code == 0, run_res.stderr
            assert run_res.stdout

            project_meta = json.loads(run_res.stdout)

            project_id = project_meta["project_id"]
            project_ids.append(project_id)

    finally:
        # Teardown multiple/all
        args = ["teardown"]
        if tear_all:
            args.extend(["--all"])
        else:
            args.extend(project_ids)

        run_res = cli_runner.invoke(
            app,
            args,
            catch_exceptions=False,
        )
        assert run_res.exit_code == 0, run_res.stderr
        # Ensure all projects are killed
        run_res = cli_runner.invoke(
            app,
            ["ps"],
            env={"COLUMNS": "1000"},
            catch_exceptions=False,
        )
        assert run_res.exit_code == 0, run_res.stderr
        for project_id in project_ids:
            assert project_id not in run_res.stdout


def test_tesseract_serve_ports_error(built_image_name):
    """Check error handling for serve -p flag."""
    cli_runner = CliRunner(mix_stderr=False)

    # Check multiple Tesseracts being served.
    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            built_image_name,
            built_image_name,
            built_image_name,
            "-p",
            "8000-8001",
        ],
        env={"COLUMNS": "1000"},
        catch_exceptions=False,
    )
    assert run_res.exit_code
    assert (
        "Port specification only works if exactly one Tesseract is being served."
        in run_res.stderr
    )

    # Check invalid ports.
    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            built_image_name,
            "-p",
            "8000-999999",
        ],
        env={"COLUMNS": "1000"},
        catch_exceptions=False,
    )
    assert run_res.exit_code
    assert "Ports '8000-999999' must be between 1025 and 65535." in run_res.stderr

    # Check poorly formatted ports.
    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            built_image_name,
            "-p",
            "8000:8081",
        ],
        env={"COLUMNS": "1000"},
        catch_exceptions=False,
    )
    assert run_res.exit_code
    assert "Port '8000:8081' must be single integer or a range" in run_res.stderr

    # Check invalid port range.
    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            built_image_name,
            "-p",
            "8000-7000",
        ],
        env={"COLUMNS": "1000"},
        catch_exceptions=False,
    )
    assert run_res.exit_code
    assert "Start port '8000' must be less than or equal to end" in run_res.stderr


@pytest.mark.parametrize("port", ["34567", "34567-34569"])
def test_tesseract_serve_ports(built_image_name, port):
    """Try to serve multiple Tesseracts on multiple ports."""
    cli_runner = CliRunner(mix_stderr=False)
    project_id = None

    # Serve tesseract on specified ports.
    run_res = cli_runner.invoke(
        app,
        ["serve", built_image_name, "-p", port],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr
    assert run_res.stdout

    project_meta = json.loads(run_res.stdout)
    project_id = project_meta["project_id"]

    # Wrap test in try-finally to ensure teardown of served Tesseract.
    try:
        # Ensure that actual used ports are in the specified port range.
        test_ports = port.split("-")
        start_port = int(test_ports[0])
        end_port = int(test_ports[1]) if len(test_ports) > 1 else start_port

        port = int(project_meta["containers"][0]["port"])
        assert port in range(start_port, end_port + 1)

        # Ensure specified ports are in `tesseract ps` and served Tesseracts are usable.
        run_res = cli_runner.invoke(
            app,
            ["ps"],
            env={"COLUMNS": "1000"},
            catch_exceptions=False,
        )

        res = requests.get(f"http://localhost:{port}/health")
        assert res.status_code == 200, res.text
        assert str(port) in run_res.stdout
    finally:
        if project_id:
            run_res = cli_runner.invoke(
                app,
                [
                    "teardown",
                    project_id,
                ],
                catch_exceptions=False,
            )
            assert run_res.exit_code == 0, run_res.stderr


def test_tesseract_serve_with_volumes(built_image_name, tmp_path, docker_client):
    """Try to serve multiple Tesseracts with volume mounting."""
    cli_runner = CliRunner(mix_stderr=False)
    project_id = None
    # Pytest creates the tmp_path fixture with drwx------ mode, we need others
    # to be able to read and execute the path so the Docker volume is readable
    # from within the container
    tmp_path.chmod(0o0707)

    dest = Path("/foo/")
    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            "--volume",
            f"{tmp_path}:{dest}",
            built_image_name,
            built_image_name,
        ],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr
    assert run_res.stdout

    project_meta = json.loads(run_res.stdout)
    project_id = project_meta["project_id"]

    try:
        tesseract0_id = project_meta["containers"][0]["name"]
        tesseract0 = docker_client.containers.get(tesseract0_id)
        tesseract1_id = project_meta["containers"][1]["name"]
        tesseract1 = docker_client.containers.get(tesseract1_id)

        # Create file outside the containers and check it from inside the container
        tmpfile = Path(tmp_path) / "hi"
        with open(tmpfile, "w") as hello:
            hello.write("world")
            hello.flush()

        exit_code, output = tesseract0.exec_run(["cat", f"{dest}/hi"])
        assert exit_code == 0
        assert output.decode() == "world"

        # Create file inside a container and check it from the other
        bar_file = dest / "bar"
        exit_code, output = tesseract0.exec_run(["touch", f"{bar_file}"])
        assert exit_code == 0
        exit_code, output = tesseract1.exec_run(["cat", f"{bar_file}"])
        assert exit_code == 0

        # The file should exist outside the container
        assert (tmp_path / "bar").exists()
    finally:
        if project_id:
            run_res = cli_runner.invoke(
                app,
                [
                    "teardown",
                    project_id,
                ],
                catch_exceptions=False,
            )
            assert run_res.exit_code == 0, run_res.stderr


def test_tesseract_cli_options_parsing(built_image_name, tmpdir):
    cli_runner = CliRunner(mix_stderr=False)

    tmpdir.chmod(0o0707)

    examples_dir = Path(__file__).parent.parent.parent / "examples"
    example_inputs = examples_dir / "vectoradd" / "example_inputs.json"

    test_commands = (
        ["apply", "-f", "json+binref", "-o", str(tmpdir), f"@{example_inputs}"],
        ["apply", f"@{example_inputs}", "-f", "json+binref", "-o", str(tmpdir)],
        ["apply", "-o", str(tmpdir), f"@{example_inputs}", "-f", "json+binref"],
    )

    for args in test_commands:
        run_res = cli_runner.invoke(
            app,
            [
                "run",
                built_image_name,
                *args,
            ],
            catch_exceptions=False,
        )
        assert run_res.exit_code == 0, run_res.stderr

        with open(Path(tmpdir) / "results.json") as fi:
            results = fi.read()
            assert ".bin:0" in results


def test_tarball_install(dummy_tesseract_package):
    import subprocess
    from textwrap import dedent

    tesseract_api = dedent(
        """
    import cowsay
    from pydantic import BaseModel

    class InputSchema(BaseModel):
        message: str = "Hello, Tesseractor!"

    class OutputSchema(BaseModel):
        out: str

    def apply(inputs: InputSchema) -> OutputSchema:
        return OutputSchema(out=cowsay.get_output_string("cow", inputs.message))
    """
    )

    tesseract_requirements = "./cowsay-6.1-py3-none-any.whl"

    subprocess.run(
        ["pip", "download", "cowsay==6.1", "-d", str(dummy_tesseract_package)]
    )
    with open(dummy_tesseract_package / "tesseract_api.py", "w") as f:
        f.write(tesseract_api)
    with open(dummy_tesseract_package / "tesseract_requirements.txt", "w") as f:
        f.write(tesseract_requirements)

    cli_runner = CliRunner(mix_stderr=False)
    result = cli_runner.invoke(
        app,
        ["--loglevel", "debug", "build", str(dummy_tesseract_package)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr
