# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for Tesseract workflows."""

import json
import os
import subprocess
from pathlib import Path
from shutil import rmtree

import pytest
import requests
import yaml
from common import build_tesseract, image_exists
from typer.testing import CliRunner

from tesseract_core.sdk.cli import AVAILABLE_RECIPES, app
from tesseract_core.sdk.docker_client import is_podman


@pytest.fixture(scope="module")
def built_image_name(
    docker_client,
    docker_cleanup_module,
    shared_dummy_image_name,
    dummy_tesseract_location,
):
    """Build the dummy Tesseract image for the tests."""
    image_name = build_tesseract(
        docker_client, dummy_tesseract_location, shared_dummy_image_name
    )
    assert image_exists(docker_client, image_name)
    docker_cleanup_module["images"].append(image_name)
    yield image_name


tested_images = ("ubuntu:24.04",)

build_matrix = [
    *[(tag, None, None) for tag in (True, False)],
    *[(False, r, None) for r in AVAILABLE_RECIPES],
    *[(False, None, img) for img in tested_images],
]


@pytest.mark.parametrize("tag,recipe,base_image", build_matrix)
def test_build_from_init_endtoend(
    docker_client, docker_cleanup, dummy_image_name, tmp_path, tag, recipe, base_image
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
        docker_client,
        tmp_path,
        dummy_image_name,
        config_override=config_override,
        tag=img_tag,
    )

    docker_cleanup["images"].append(image_name)
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


@pytest.mark.parametrize("skip_checks", [True, False])
def test_build_generate_only(dummy_tesseract_location, skip_checks):
    """Test output of build with --generate_only flag."""
    cli_runner = CliRunner(mix_stderr=False)
    build_res = cli_runner.invoke(
        app,
        [
            "build",
            str(dummy_tesseract_location),
            "--generate-only",
            *(
                ("--config-override=build_config.skip_checks=True",)
                if skip_checks
                else ()
            ),
        ],
        # Ensure that the output is not truncated
        env={"COLUMNS": "1000"},
        catch_exceptions=False,
    )
    assert build_res.exit_code == 0, build_res.stderr
    # Check that stdout contains build command
    command = "buildx build"
    assert command in build_res.stderr

    build_dir = Path(build_res.stdout.strip())
    assert build_dir.exists()
    dockerfile_path = build_dir / "Dockerfile"
    assert dockerfile_path.exists()

    with open(build_dir / "Dockerfile") as f:
        docker_file_contents = f.read()
        if skip_checks:
            assert 'RUN ["tesseract-runtime", "check"]' not in docker_file_contents
        else:
            assert 'RUN ["tesseract-runtime", "check"]' in docker_file_contents


@pytest.mark.parametrize("no_compose", [True, False])
def test_env_passthrough_serve(
    docker_cleanup, docker_client, built_image_name, no_compose, tmpdir
):
    """Ensure we can pass environment variables to tesseracts when serving."""
    tmpdir.chmod(0o777)
    (tmpdir / "input").mkdir()
    (tmpdir / "output").mkdir()

    run_res = subprocess.run(
        [
            "tesseract",
            "serve",
            built_image_name,
            "--env=TEST_ENV_VAR=foo",
            *(["--no-compose"] if no_compose else []),
            "--input-path",
            str(tmpdir / "input"),
            "--output-path",
            str(tmpdir / "output"),
        ],
        capture_output=True,
        text=True,
    )
    assert run_res.returncode == 0, run_res.stderr
    assert run_res.stdout

    project_meta = json.loads(run_res.stdout)
    project_id = project_meta["project_id"]
    tesseract_id = project_meta["containers"][0]["name"]

    if no_compose:
        docker_cleanup["containers"].append(tesseract_id)
    else:
        docker_cleanup["project_ids"].append(project_id)

    container = docker_client.containers.get(tesseract_id)
    exit_code, output = container.exec_run(["sh", "-c", "echo $TEST_ENV_VAR"])
    assert exit_code == 0, f"Command failed with exit code {exit_code}"
    assert "foo" in output.decode("utf-8"), f"Output was: {output.decode('utf-8')}"

    exit_code, input_path = container.exec_run(
        ["sh", "-c", "echo $TESSERACT_INPUT_PATH"]
    )
    exit_code, output_path = container.exec_run(
        ["sh", "-c", "echo $TESSERACT_OUTPUT_PATH"]
    )
    assert exit_code == 0, f"Command failed with exit code {exit_code}"
    assert "/tesseract/input_data" in input_path.decode("utf-8")
    assert "/tesseract/output_data" in output_path.decode("utf-8")


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


@pytest.mark.parametrize("no_compose", [True, False])
@pytest.mark.parametrize("user", [None, "root", "1000:1000"])
def test_run_as_user(docker_client, built_image_name, user, no_compose, docker_cleanup):
    """Ensure we can run a basic Tesseract image as any user."""
    cli_runner = CliRunner(mix_stderr=False)

    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            built_image_name,
            "--user",
            user,
            *(["--no-compose"] if no_compose else []),
        ],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr

    project_meta = json.loads(run_res.stdout)
    project_id = project_meta["project_id"]
    container = docker_client.containers.get(project_meta["containers"][0]["name"])
    if no_compose:
        docker_cleanup["containers"].append(container)
    else:
        docker_cleanup["project_ids"].append(project_id)

    exit_code, output = container.exec_run(["id", "-u"])
    if user is None:
        expected_user = os.getuid()
    elif user == "root":
        expected_user = 0
    else:
        expected_user = int(user.split(":")[0])

    assert exit_code == 0
    assert output.decode("utf-8").strip() == str(expected_user)


@pytest.mark.parametrize("no_compose", [True, False])
def test_tesseract_serve_pipeline(
    docker_client, built_image_name, no_compose, docker_cleanup, tmpdir
):
    cli_runner = CliRunner(mix_stderr=False)
    project_id = None
    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            built_image_name,
            *(["--no-compose"] if no_compose else []),
        ],
        catch_exceptions=False,
    )

    assert run_res.exit_code == 0, run_res.stderr
    assert run_res.stdout

    project_meta = json.loads(run_res.stdout)

    project_id = project_meta["project_id"]
    if no_compose:
        project_container = docker_client.containers.get(project_id)
        docker_cleanup["containers"].append(project_container)
    else:
        docker_cleanup["project_ids"].append(project_id)
        project_containers = project_meta["containers"][0]["name"]
        if not project_containers:
            raise ValueError(f"Could not find container for project '{project_id}'")

        project_container = docker_client.containers.get(project_containers)

    assert project_container.name == project_meta["containers"][0]["name"]
    assert project_container.host_port == project_meta["containers"][0]["port"]
    assert project_container.host_ip == project_meta["containers"][0]["ip"]

    # Ensure served Tesseract is usable
    res = requests.get(
        f"http://{project_container.host_ip}:{project_container.host_port}/health"
    )
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
    assert project_container.host_ip in run_res.stdout
    assert project_container.short_id in run_res.stdout


@pytest.mark.parametrize("tear_all", [True, False])
@pytest.mark.parametrize("no_compose", [True, False])
def test_tesseract_teardown_multiple(built_image_name, tear_all, no_compose):
    """Teardown multiple projects."""
    cli_runner = CliRunner(mix_stderr=False)

    project_ids = []
    try:
        for _ in range(2):
            # Serve
            run_res = cli_runner.invoke(
                app,
                [
                    "serve",
                    built_image_name,
                    *(["--no-compose"] if no_compose else []),
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
    assert "Ports '8000-999999' must be between" in run_res.stderr

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


@pytest.mark.parametrize("port", ["fixed", "range"])
def test_tesseract_serve_ports(built_image_name, port, docker_cleanup, free_port):
    """Try to serve multiple Tesseracts on multiple ports."""
    cli_runner = CliRunner(mix_stderr=False)
    project_id = None

    if port == "fixed":
        port_arg = str(free_port)
    elif port == "range":
        port_arg = f"{free_port}-{free_port + 1}"
    else:
        raise ValueError(f"Unknown port type: {port}")

    # Serve tesseract on specified ports.
    run_res = cli_runner.invoke(
        app,
        ["serve", built_image_name, "-p", port_arg],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr
    assert run_res.stdout

    project_meta = json.loads(run_res.stdout)
    project_id = project_meta["project_id"]
    docker_cleanup["project_ids"].append(project_id)

    # Ensure that actual used ports are in the specified port range.
    test_ports = port_arg.split("-")
    start_port = int(test_ports[0])
    end_port = int(test_ports[1]) if len(test_ports) > 1 else start_port

    actual_port = int(project_meta["containers"][0]["port"])
    assert actual_port in range(start_port, end_port + 1)

    # Ensure specified ports are in `tesseract ps` and served Tesseracts are usable.
    run_res = cli_runner.invoke(
        app,
        ["ps"],
        env={"COLUMNS": "1000"},
        catch_exceptions=False,
    )

    res = requests.get(f"http://localhost:{actual_port}/health")
    assert res.status_code == 200, res.text
    assert str(actual_port) in run_res.stdout


@pytest.mark.parametrize("no_compose", [True, False])
@pytest.mark.parametrize("volume_type", ["bind", "named"])
@pytest.mark.parametrize("user", [None, "root", "1000:1000"])
def test_tesseract_serve_volume_permissions(
    built_image_name,
    docker_client,
    docker_volume,
    tmp_path,
    docker_cleanup,
    user,
    volume_type,
    no_compose,
):
    """Test serving Tesseract with a Docker volume or bind mount.

    This should cover most permissions issues that can arise with Docker volumes.
    """
    if is_podman() and not no_compose:
        pytest.xfail("Podman does not support --no-compose option.")

    cli_runner = CliRunner(mix_stderr=False)
    project_id = None

    dest = Path("/tesseract/output_data")

    if volume_type == "bind":
        # Use bind mount with a temporary directory
        volume_to_bind = str(tmp_path)
    elif volume_type == "named":
        # Use docker volume instead of bind mounting
        volume_to_bind = docker_volume.name
    else:
        raise ValueError(f"Unknown volume type: {volume_type}")

    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            "--volume",
            f"{volume_to_bind}:{dest}:rw",
            *(("--user", user) if user else []),
            built_image_name,
            *(("--no-compose",) if no_compose else [built_image_name]),
        ],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr
    assert run_res.stdout

    project_meta = json.loads(run_res.stdout)
    project_id = project_meta["project_id"]

    if no_compose:
        docker_cleanup["containers"].append(project_id)
    else:
        docker_cleanup["project_ids"].append(project_id)

    tesseract0_id = project_meta["containers"][0]["name"]
    tesseract0 = docker_client.containers.get(tesseract0_id)

    # Sanity check: Should always be allowed to read/write files in the default workdir
    exit_code, output = tesseract0.exec_run(["touch", "./test.txt"])
    assert exit_code == 0, output.decode()

    if volume_type == "bind":
        # Create file outside the containers and check it from inside the container
        tmpfile = Path(tmp_path) / "hi"
        with open(tmpfile, "w") as hello:
            hello.write("world")
            hello.flush()

        if volume_type == "bind" and user not in (None, "root"):
            # If we are not running as root, ensure the file is readable by the target user
            tmp_path.chmod(0o777)
            tmpfile.chmod(0o644)

        exit_code, output = tesseract0.exec_run(["cat", f"{dest}/hi"])
        assert exit_code == 0
        assert output.decode() == "world"

    # Create file inside a container and access it from the other
    bar_file = dest / "bar"
    exit_code, output = tesseract0.exec_run(["ls", "-la", str(dest)])
    assert exit_code == 0, output.decode()

    exit_code, output = tesseract0.exec_run(["touch", str(bar_file)])
    assert exit_code == 0

    if not no_compose:
        tesseract1_id = project_meta["containers"][1]["name"]
        tesseract1 = docker_client.containers.get(tesseract1_id)

        exit_code, output = tesseract1.exec_run(["cat", str(bar_file)])
        assert exit_code == 0
        exit_code, output = tesseract1.exec_run(
            ["bash", "-c", f'echo "hello" > {bar_file}']
        )
        assert exit_code == 0

    if volume_type == "bind":
        # The file should exist outside the container
        assert (tmp_path / "bar").exists()


@pytest.mark.parametrize("default_output_path", [True, False])
def test_tesseract_serve_multiple_outputs(
    built_image_name,
    docker_client,
    tmp_path,
    docker_cleanup,
    default_output_path,
):
    """Test serving Tesseract with a Docker volume or bind mount.

    This should cover most permissions issues that can arise with Docker volumes.
    """
    cli_runner = CliRunner(mix_stderr=False)
    project_id = None

    if default_output_path:
        output_args = []
        output_dir = Path(os.getcwd()) / "tesseract_output"
    else:
        output_args = ["--output-path", str(tmp_path)]
        output_dir = tmp_path
        tmp_path.chmod(0o777)

    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            built_image_name,
            built_image_name,
            built_image_name,
            *output_args,
            "--service-names",
            "tess-1,tess-2,tess-3",
        ],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr
    assert run_res.stdout

    project_meta = json.loads(run_res.stdout)
    project_id = project_meta["project_id"]

    docker_cleanup["project_ids"].append(project_id)

    tesseract0_id = project_meta["containers"][0]["name"]
    tesseract0 = docker_client.containers.get(tesseract0_id)
    tesseract1_id = project_meta["containers"][1]["name"]
    tesseract1 = docker_client.containers.get(tesseract1_id)
    tesseract2_id = project_meta["containers"][2]["name"]
    tesseract2 = docker_client.containers.get(tesseract2_id)

    exit_code, output = tesseract0.exec_run(
        ["touch", "/tesseract/output_data/test_0.txt"]
    )
    assert exit_code == 0, output.decode()
    exit_code, output = tesseract1.exec_run(
        ["touch", "/tesseract/output_data/test_1.txt"]
    )
    assert exit_code == 0, output.decode()
    exit_code, output = tesseract2.exec_run(
        ["touch", "/tesseract/output_data/test_2.txt"]
    )
    assert exit_code == 0, output.decode()

    assert (output_dir / "tess-1" / "test_0.txt").exists()
    assert (output_dir / "tess-2" / "test_1.txt").exists()
    assert (output_dir / "tess-3" / "test_2.txt").exists()

    if default_output_path:
        # Remove the tesseract_output directory
        rmtree(output_dir)


def test_tesseract_serve_interop(built_image_name, docker_client, docker_cleanup):
    cli_runner = CliRunner(mix_stderr=False)

    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            built_image_name,
            built_image_name,
            "--service-names",
            "tess-1,tess-2",
        ],
        env={"COLUMNS": "1000"},
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0

    project_meta = json.loads(run_res.stdout)
    project_id = project_meta["project_id"]
    docker_cleanup["project_ids"].append(project_id)

    project_containers = [project_meta["containers"][i]["name"] for i in range(2)]

    tess_1 = docker_client.containers.get(project_containers[0])

    returncode, stdout = tess_1.exec_run(
        [
            "python",
            "-c",
            'import requests; requests.get("http://tess-2:8000/health").raise_for_status()',
        ]
    )
    assert returncode == 0, stdout.decode()


@pytest.mark.parametrize("no_compose", [True, False])
def test_serve_nonstandard_host_ip(
    docker_client, built_image_name, docker_cleanup, free_port, no_compose
):
    """Test serving Tesseract with a non-standard host IP."""

    def _get_host_ip():
        """Get a network interface IP address that is not localhost."""
        import socket
        from contextlib import closing

        with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as s:
            # We ping to the Google DNS server to get a valid external IP address
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]

    cli_runner = CliRunner(mix_stderr=False)
    project_id = None

    # Use a non-standard host IP
    host_ip = _get_host_ip()
    assert host_ip not in ("", "127.0.0.1", "localhost")  # sanity check

    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            built_image_name,
            "-p",
            str(free_port),
            "--host-ip",
            host_ip,
            *(["--no-compose"] if no_compose else []),
        ],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr
    assert run_res.stdout
    project_meta = json.loads(run_res.stdout)
    project_id = project_meta["project_id"]

    if no_compose:
        docker_cleanup["containers"].append(project_id)
    else:
        docker_cleanup["project_ids"].append(project_id)

    project_container = docker_client.containers.get(
        project_meta["containers"][0]["name"]
    )
    assert project_container.host_ip == host_ip

    res = requests.get(f"http://{host_ip}:{project_container.host_port}/health")
    assert res.status_code == 200, res.text

    with pytest.raises(requests.ConnectionError):
        # Ensure that the Tesseract is not accessible from localhost
        requests.get(f"http://localhost:{project_container.host_port}/health")


def test_tesseract_cli_options_parsing(built_image_name, tmpdir):
    cli_runner = CliRunner(mix_stderr=False)

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


def test_tarball_install(dummy_tesseract_package, docker_cleanup):
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

    img_tag = json.loads(result.stdout)[0]
    docker_cleanup["images"].append(img_tag)


def test_logging(dummy_tesseract_package, tmpdir, docker_cleanup):
    from textwrap import dedent

    tesseract_api = dedent(
        """
    from pydantic import BaseModel

    print("Hello from tesseract_api.py!")

    class InputSchema(BaseModel):
        message: str = "Hello, Tesseractor!"

    class OutputSchema(BaseModel):
        out: str

    def apply(inputs: InputSchema) -> OutputSchema:
        print("Hello from apply!")
        return OutputSchema(out=f"Received message: {inputs.message}")
    """
    )

    with open(dummy_tesseract_package / "tesseract_api.py", "w") as f:
        f.write(tesseract_api)

    cli_runner = CliRunner(mix_stderr=False)
    result = cli_runner.invoke(
        app,
        ["--loglevel", "debug", "build", str(dummy_tesseract_package)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr

    img_tag = json.loads(result.stdout)[0]
    docker_cleanup["images"].append(img_tag)

    # Run the Tesseract and capture logs
    # Use subprocess because pytest messes with stdout/stderr
    run_res = subprocess.run(
        [
            "tesseract",
            "run",
            img_tag,
            "apply",
            '{"inputs": {"message": "Test message"}}',
            "--output-path",
            tmpdir,
        ],
        capture_output=True,
        text=True,
    )
    assert run_res.returncode == 0, run_res.stderr
    assert "" == run_res.stdout.strip()
    assert "Hello from tesseract_api.py!" == run_res.stderr.strip()

    out_file = Path(tmpdir) / "results.json"
    assert out_file.exists()
    with open(out_file) as f:
        results = json.load(f)
        assert results["out"] == "Received message: Test message"

    logdir = next((Path(tmpdir) / "logs").iterdir())
    log_file = logdir / "tesseract.log"
    assert log_file.exists()

    with open(log_file) as f:
        log_content = f.read()
    assert "Hello from apply!" == log_content.strip()


def _prepare_mpa_test_image(dummy_tesseract_package, docker_cleanup):
    from textwrap import dedent

    tesseract_api = dedent(
        """
    from pydantic import BaseModel
    from tesseract_core.runtime.experimental import log_artifact, log_metric, log_parameter

    class InputSchema(BaseModel):
        pass

    class OutputSchema(BaseModel):
        pass

    def apply(inputs: InputSchema) -> OutputSchema:
        steps = 5
        param_value = "test_param"
        # Log parameters
        log_parameter("test_parameter", param_value)
        log_parameter("steps_config", steps)

        # Log metrics over multiple steps
        for step in range(steps):
            log_metric("squared_step", step ** 2, step=step)

        # Create and log an artifact
        artifact_content = "Test artifact content"

        artifact_path = "/tmp/test_artifact.txt"
        with open(artifact_path, "w") as f:
            f.write(artifact_content)

        log_artifact(artifact_path)

        return OutputSchema()
        """
    )

    # Write the API file
    with open(dummy_tesseract_package / "tesseract_api.py", "w") as f:
        f.write(tesseract_api)
    # Add mlflow dependency
    with open(dummy_tesseract_package / "tesseract_requirements.txt", "w") as f:
        f.write("mlflow\n")

    cli_runner = CliRunner(mix_stderr=False)

    # Build the Tesseract
    result = cli_runner.invoke(
        app,
        ["--loglevel", "debug", "build", str(dummy_tesseract_package)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr

    img_tag = json.loads(result.stdout)[0]
    docker_cleanup["images"].append(img_tag)

    return img_tag


@pytest.mark.parametrize("default_log_dir", ["default", "custom"])
def test_mpa_file_backend(
    dummy_tesseract_package, tmpdir, docker_cleanup, default_log_dir
):
    """Test the MPA (Metrics, Parameters, and Artifacts) submodule with file backend."""
    import csv

    img_tag = _prepare_mpa_test_image(dummy_tesseract_package, docker_cleanup)

    if default_log_dir == "default":
        run_cmd = [
            "tesseract",
            "run",
            img_tag,
            "apply",
            '{"inputs": {}}',
            "--output-path",
            tmpdir,
        ]
        log_dir = Path(tmpdir) / "logs"
    elif default_log_dir == "custom":
        run_cmd = [
            "tesseract",
            "run",
            "--env",
            "LOG_DIR=/tesseract/output_data/mpa_logs",
            img_tag,
            "apply",
            '{"inputs": {}}',
            "--output-path",
            tmpdir,
        ]
        log_dir = Path(tmpdir) / "mpa_logs"

    run_res = subprocess.run(
        run_cmd,
        capture_output=True,
        text=True,
    )
    assert run_res.returncode == 0, run_res.stderr

    assert log_dir.exists()

    # Find the run directory (should be only one)
    run_dirs = list(log_dir.glob("run_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    # Verify parameters file
    params_file = run_dir / "parameters.json"
    assert params_file.exists()
    with open(params_file) as f:
        params = json.load(f)
        assert params["test_parameter"] == "test_param"
        assert params["steps_config"] == 5

    # Verify metrics file
    metrics_file = run_dir / "metrics.csv"
    assert metrics_file.exists()

    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        metrics = list(reader)

        # Should have 5 metrics: 5 squared_step (0, 1, 4, 9, 16)
        assert len(metrics) == 5

        # Check squared_step values
        squared_metrics = [m for m in metrics if m["key"] == "squared_step"]
        assert len(squared_metrics) == 5
        for i, metric in enumerate(squared_metrics):
            assert float(metric["value"]) == i**2
            assert int(metric["step"]) == i

    # Verify artifacts directory and artifact file
    artifacts_dir = run_dir / "artifacts"
    assert artifacts_dir.exists()

    artifact_file = artifacts_dir / "test_artifact.txt"
    assert artifact_file.exists()

    with open(artifact_file) as f:
        artifact_data = f.read()
        assert artifact_data == "Test artifact content"


def test_mpa_mlflow_backend(dummy_tesseract_package, tmpdir, docker_cleanup):
    """Test the MPA (Metrics, Parameters, and Artifacts) submodule with MLflow backend."""
    img_tag = _prepare_mpa_test_image(dummy_tesseract_package, docker_cleanup)

    # Point MLflow to a local directory
    run_cmd = [
        "tesseract",
        "run",
        "--env",
        "MLFLOW_TRACKING_URI=/tesseract/output_data/mlruns",
        img_tag,
        "apply",
        '{"inputs": {}}',
        "--output-path",
        tmpdir,
    ]

    run_res = subprocess.run(
        run_cmd,
        capture_output=True,
        text=True,
    )
    assert run_res.returncode == 0, run_res.stderr

    # Check for mlruns directory structure
    mlruns_dir = Path(tmpdir) / "mlruns"
    assert mlruns_dir.exists()
    assert (mlruns_dir / "0").exists()  # Default experiment ID is 0

    # Find run directories
    run_dirs = [d for d in (mlruns_dir / "0").iterdir() if d.is_dir()]
    assert len(run_dirs) == 1  # Should be only one run
    run_dir = run_dirs[0]
    assert run_dir.is_dir()
    assert (run_dir / "artifacts").exists()
    assert (run_dir / "metrics").exists()
    assert (run_dir / "params").exists()

    # Verify parameters file
    param_file = run_dir / "params" / "test_parameter"
    assert param_file.exists()
    with open(param_file) as f:
        param_value = f.read().strip()
        assert param_value == "test_param"

    # Verify metrics file
    metrics_file = run_dir / "metrics" / "squared_step"
    assert metrics_file.exists()
    with open(metrics_file) as f:
        metrics = f.readlines()
        assert len(metrics) == 5
        for i, metric in enumerate(metrics):
            parts = metric.split()
            assert len(parts) == 3
            assert float(parts[1]) == i**2  # Check squared_step values
            assert int(parts[2]) == i

    # Verify artifacts directory and artifact file
    artifacts_dir = run_dir / "artifacts"
    assert artifacts_dir.exists()
