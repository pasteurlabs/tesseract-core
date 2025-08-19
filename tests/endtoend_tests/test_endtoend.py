# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for Tesseract workflows."""

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
import requests
import yaml
from common import build_tesseract, encode_array, image_exists
from typer.testing import CliRunner

from tesseract_core.sdk.cli import AVAILABLE_RECIPES, app
from tesseract_core.sdk.config import get_config
from tesseract_core.sdk.docker_client import _get_docker_executable


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
            assert "tesseract-runtime check" not in docker_file_contents
        else:
            assert "tesseract-runtime check" in docker_file_contents


def test_env_passthrough_serve(docker_cleanup, docker_client, built_image_name):
    """Ensure we can pass environment variables to tesseracts when serving."""
    run_res = subprocess.run(
        [
            "tesseract",
            "serve",
            built_image_name,
            "--env=TEST_ENV_VAR=foo",
        ],
        capture_output=True,
        text=True,
    )
    assert run_res.returncode == 0, run_res.stderr
    assert run_res.stdout

    serve_meta = json.loads(run_res.stdout)
    container_name = serve_meta["container_name"]
    docker_cleanup["containers"].append(container_name)

    container = docker_client.containers.get(container_name)
    exit_code, output = container.exec_run(["sh", "-c", "echo $TEST_ENV_VAR"])
    assert exit_code == 0, f"Command failed with exit code {exit_code}"
    assert "foo" in output.decode("utf-8"), f"Output was: {output.decode('utf-8')}"


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
    cli_runner = CliRunner(mix_stderr=False)

    test_commands = ("openapi-schema", "health")

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


@pytest.mark.parametrize("user", [None, "root", "1000:1000"])
def test_run_as_user(docker_client, built_image_name, user, docker_cleanup):
    """Ensure we can run a basic Tesseract image as any user."""
    cli_runner = CliRunner(mix_stderr=False)

    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            built_image_name,
            "--user",
            user,
        ],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr

    serve_meta = json.loads(run_res.stdout)
    container = docker_client.containers.get(serve_meta["container_name"])
    docker_cleanup["containers"].append(container)

    exit_code, output = container.exec_run(["id", "-u"])
    if user is None:
        expected_user = os.getuid()
    elif user == "root":
        expected_user = 0
    else:
        expected_user = int(user.split(":")[0])

    assert exit_code == 0
    assert output.decode("utf-8").strip() == str(expected_user)


def test_tesseract_serve_pipeline(docker_client, built_image_name, docker_cleanup):
    cli_runner = CliRunner(mix_stderr=False)
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

    serve_meta = json.loads(run_res.stdout)

    container_name = serve_meta["container_name"]
    container = docker_client.containers.get(container_name)
    docker_cleanup["containers"].append(container)

    assert container.name == container_name
    assert container.host_port == serve_meta["containers"][0]["port"]
    assert container.host_ip == serve_meta["containers"][0]["ip"]

    # Ensure served Tesseract is usable
    res = requests.get(f"http://{container.host_ip}:{container.host_port}/health")
    assert res.status_code == 200, res.text

    # Ensure container name is shown in `tesseract ps`
    run_res = cli_runner.invoke(
        app,
        ["ps"],
        env={"COLUMNS": "1000"},
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr
    assert container_name in run_res.stdout
    assert container.host_port in run_res.stdout
    assert container.host_ip in run_res.stdout
    assert container.short_id in run_res.stdout


@pytest.mark.parametrize("tear_all", [True, False])
def test_tesseract_teardown_multiple(built_image_name, tear_all):
    """Teardown multiple served tesseracts."""
    cli_runner = CliRunner(mix_stderr=False)

    container_names = []
    try:
        for _ in range(2):
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

            serve_meta = json.loads(run_res.stdout)

            container_name = serve_meta["container_name"]
            container_names.append(container_name)

    finally:
        # Teardown multiple/all
        args = ["teardown"]
        if tear_all:
            args.extend(["--all"])
        else:
            args.extend(container_names)

        run_res = cli_runner.invoke(
            app,
            args,
            catch_exceptions=False,
        )
        assert run_res.exit_code == 0, run_res.stderr
        # Ensure all containers are killed
        run_res = cli_runner.invoke(
            app,
            ["ps"],
            env={"COLUMNS": "1000"},
            catch_exceptions=False,
        )
        assert run_res.exit_code == 0, run_res.stderr
        for container_name in container_names:
            assert container_name not in run_res.stdout


def test_tesseract_serve_ports_error(built_image_name):
    """Check error handling for serve -p flag."""
    cli_runner = CliRunner(mix_stderr=False)

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
    container_name = None

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

    serve_meta = json.loads(run_res.stdout)
    container_name = serve_meta["container_name"]
    docker_cleanup["containers"].append(container_name)

    # Ensure that actual used ports are in the specified port range.
    test_ports = port_arg.split("-")
    start_port = int(test_ports[0])
    end_port = int(test_ports[1]) if len(test_ports) > 1 else start_port

    actual_port = int(serve_meta["containers"][0]["port"])
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
):
    """Test serving Tesseract with a Docker volume or bind mount.

    This should cover most permissions issues that can arise with Docker volumes.
    """
    cli_runner = CliRunner(mix_stderr=False)

    dest = Path("/tesseract/output_data")

    if volume_type == "bind":
        # Use bind mount with a temporary directory
        volume_to_bind = str(tmp_path)
    elif volume_type == "named":
        # Use docker volume instead of bind mounting
        volume_to_bind = docker_volume.name
    else:
        raise ValueError(f"Unknown volume type: {volume_type}")

    def serve_tesseract():
        run_res = cli_runner.invoke(
            app,
            [
                "serve",
                "--volume",
                f"{volume_to_bind}:{dest}:rw",
                *(("--user", user) if user else []),
                built_image_name,
            ],
            catch_exceptions=False,
        )
        assert run_res.exit_code == 0, run_res.stderr
        assert run_res.stdout
        serve_meta = json.loads(run_res.stdout)
        container_name = serve_meta["container_name"]
        docker_cleanup["containers"].append(container_name)
        return docker_client.containers.get(container_name)

    tesseract0 = serve_tesseract()
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

    tesseract1 = serve_tesseract()
    exit_code, output = tesseract1.exec_run(["cat", str(bar_file)])
    assert exit_code == 0
    exit_code, output = tesseract1.exec_run(
        ["bash", "-c", f'echo "hello" > {bar_file}']
    )
    assert exit_code == 0

    if volume_type == "bind":
        # The file should exist outside the container
        assert (tmp_path / "bar").exists()


@pytest.mark.parametrize("method", ["run", "serve"])
@pytest.mark.parametrize("array_format", ["json", "json+base64", "json+binref"])
def test_io_path_interactions(
    docker_cleanup, built_image_name, tmp_path, method, array_format
):
    """Ensure that input / output paths work across different methods of interaction and file formats."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    encoding = array_format.split("+")[-1]
    example_inputs = {
        "inputs": {
            "a": encode_array(np.array([1, 2]), encoding=encoding, basedir=input_dir),
            "b": encode_array(np.array([3, 4]), encoding=encoding, basedir=input_dir),
            "s": 1.0,
            "normalize": True,
        },
    }

    if method == "serve":
        run_res = subprocess.run(
            [
                "tesseract",
                "serve",
                built_image_name,
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-f",
                array_format,
            ],
            capture_output=True,
            text=True,
        )
        assert run_res.returncode == 0, run_res.stderr
        assert run_res.stdout

        serve_meta = json.loads(run_res.stdout)
        container_name = serve_meta["container_name"]
        docker_cleanup["containers"].append(container_name)

        req = requests.post(
            f"http://localhost:{serve_meta['containers'][0]['port']}/apply",
            json=example_inputs,
        )
        assert req.status_code == 200, req.text
        result = req.json()

    elif method == "run":
        run_res = subprocess.run(
            [
                "tesseract",
                "run",
                built_image_name,
                "apply",
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-f",
                array_format,
                json.dumps(example_inputs),
            ],
            capture_output=True,
            text=True,
        )
        assert run_res.returncode == 0, run_res.stderr
        assert run_res.stdout
        result = json.loads(run_res.stdout)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure result payload is as expected
    assert "result" in result
    assert result["result"]["data"]["encoding"] == encoding

    if array_format == "json+binref":
        binref_path = result["result"]["data"]["buffer"].rsplit(":", maxsplit=1)[0]
        binref_file = output_dir / binref_path
        assert binref_file.exists(), f"Expected binref file {binref_file} to exist"

    # Ensure logs are written to the output directory
    if method == "serve":
        run_dirs = list(output_dir.glob("run_*/"))
        assert len(run_dirs) == 1, f"Expected one run directory, found: {run_dirs}"
        output_dir = run_dirs[0]

    log_dir = output_dir / "logs"
    assert log_dir.exists(), f"Expected log directory {log_dir} to exist"
    assert (log_dir / "tesseract.log").exists(), "Expected tesseract.log to exist"

    # Also try overriding the output format via Accept header
    if method == "serve":
        req = requests.post(
            f"http://localhost:{serve_meta['containers'][0]['port']}/apply",
            json=example_inputs,
            headers={"Accept": "application/json+base64"},
        )
        assert req.status_code == 200, req.text
        result = req.json()
        assert "result" in result
        assert result["result"]["data"]["encoding"] == "base64"


def test_tesseract_serve_interop(
    built_image_name, dummy_network_name, docker_client, docker_cleanup
):
    cli_runner = CliRunner(mix_stderr=False)

    docker = _get_docker_executable()

    # Network create using subprocess
    subprocess.run(
        [*docker, "network", "create", dummy_network_name],
        check=True,
    )
    docker_cleanup["networks"].append(dummy_network_name)

    def serve_tesseract(alias: str):
        run_res = cli_runner.invoke(
            app,
            [
                "serve",
                "--network",
                dummy_network_name,
                "--network-alias",
                alias,
                built_image_name,
            ],
            env={"COLUMNS": "1000"},
            catch_exceptions=False,
        )
        assert run_res.exit_code == 0

        serve_meta = json.loads(run_res.stdout)
        container_name = serve_meta["container_name"]
        container = docker_client.containers.get(container_name)
        docker_cleanup["containers"].append(container)
        return container

    # Serve two separate tesseracts on the same network
    tess_1 = serve_tesseract("tess_1")
    tess_2 = serve_tesseract("tess_2")

    returncode, stdout = tess_1.exec_run(
        [
            "python",
            "-c",
            'import requests; requests.get("http://tess_2:8000/health").raise_for_status()',
        ]
    )
    assert returncode == 0, stdout.decode()

    returncode, stdout = tess_2.exec_run(
        [
            "python",
            "-c",
            'import requests; requests.get("http://tess_1:8000/health").raise_for_status()',
        ]
    )
    assert returncode == 0, stdout.decode()


def test_serve_nonstandard_host_ip(
    docker_client, built_image_name, docker_cleanup, free_port
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
    container_name = None

    # Use a non-standard host IP
    host_ip = _get_host_ip()
    assert host_ip not in ("", "127.0.0.1", "localhost")  # sanity check

    run_res = cli_runner.invoke(
        app,
        ["serve", built_image_name, "-p", str(free_port), "--host-ip", host_ip],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr
    assert run_res.stdout
    serve_meta = json.loads(run_res.stdout)
    container_name = serve_meta["container_name"]

    docker_cleanup["containers"].append(container_name)

    container = docker_client.containers.get(container_name)
    assert container.host_ip == host_ip

    res = requests.get(f"http://{host_ip}:{container.host_port}/health")
    assert res.status_code == 200, res.text

    with pytest.raises(requests.ConnectionError):
        # Ensure that the Tesseract is not accessible from localhost
        requests.get(f"http://localhost:{container.host_port}/health")


def test_tarball_install(dummy_tesseract_package, docker_cleanup):
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


@pytest.fixture(scope="module")
def logging_test_image(dummy_tesseract_location, tmpdir_factory, docker_cleanup_module):
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

    workdir = tmpdir_factory.mktemp("mpa_test_image")

    # Write the API file
    with open(workdir / "tesseract_api.py", "w") as f:
        f.write(tesseract_api)
    # Add mlflow dependency
    with open(workdir / "tesseract_requirements.txt", "w") as f:
        f.write("mlflow\n")

    shutil.copy(
        dummy_tesseract_location / "tesseract_config.yaml",
        workdir / "tesseract_config.yaml",
    )

    cli_runner = CliRunner(mix_stderr=False)

    # Build the Tesseract
    result = cli_runner.invoke(
        app,
        ["--loglevel", "debug", "build", str(workdir), "--tag", "logging_test_image"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr

    img_tag = json.loads(result.stdout)[0]
    docker_cleanup_module["images"].append(img_tag)
    return img_tag


def test_logging_tesseract_run(logging_test_image, tmpdir):
    # Run the Tesseract and capture logs
    # Use subprocess because pytest messes with stdout/stderr
    run_res = subprocess.run(
        [
            "tesseract",
            "run",
            logging_test_image,
            "apply",
            '{"inputs": {"message": "Test message"}}',
            "--output-path",
            tmpdir,
        ],
        capture_output=True,
        text=True,
    )
    assert run_res.returncode == 0, run_res.stderr
    assert "Hello from tesseract_api.py!\nHello from apply!" == run_res.stderr.strip()

    results = json.loads(run_res.stdout.strip())
    assert results["out"] == "Received message: Test message"

    logdir = Path(tmpdir) / "logs"

    log_file = logdir / "tesseract.log"
    assert log_file.exists()

    with open(log_file) as f:
        log_content = f.read()
    assert "Hello from apply!" == log_content.strip()


def test_logging_tesseract_serve(
    logging_test_image, tmpdir, docker_cleanup, docker_client
):
    serve_res = subprocess.run(
        [
            "tesseract",
            "serve",
            logging_test_image,
            "--output-path",
            tmpdir,
        ],
        capture_output=True,
        text=True,
    )
    assert serve_res.returncode == 0, serve_res.stderr
    assert serve_res.stdout

    serve_meta = json.loads(serve_res.stdout)
    container_name = serve_meta["container_name"]
    docker_cleanup["containers"].append(container_name)
    container = docker_client.containers.get(container_name)

    run_id = str(uuid.uuid4())
    res = requests.post(
        f"http://{container.host_ip}:{container.host_port}/apply",
        params={"run_id": run_id},
        json={"inputs": {}},
    )
    assert res.status_code == 200, res.text

    log_file = Path(tmpdir) / f"run_{run_id}/logs/tesseract.log"
    assert log_file.exists()

    with open(log_file) as f:
        log_content = f.read()
    assert "Hello from apply!" == log_content.strip()


@pytest.fixture(scope="module")
def mpa_test_image(dummy_tesseract_location, tmpdir_factory, docker_cleanup_module):
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
    workdir = tmpdir_factory.mktemp("mpa_test_image")

    # Write the API file
    with open(workdir / "tesseract_api.py", "w") as f:
        f.write(tesseract_api)
    # Add mlflow dependency
    with open(workdir / "tesseract_requirements.txt", "w") as f:
        f.write("mlflow\n")

    shutil.copy(
        dummy_tesseract_location / "tesseract_config.yaml",
        workdir / "tesseract_config.yaml",
    )

    cli_runner = CliRunner(mix_stderr=False)

    # Build the Tesseract
    result = cli_runner.invoke(
        app,
        ["--loglevel", "debug", "build", str(workdir), "--tag", "mpa_test_image"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr

    img_tag = json.loads(result.stdout)[0]
    docker_cleanup_module["images"].append(img_tag)
    return img_tag


def test_mpa_file_backend(tmpdir, mpa_test_image):
    """Test the MPA (Metrics, Parameters, and Artifacts) submodule with file backend."""
    import csv

    outdir = Path(tmpdir)

    run_cmd = [
        "tesseract",
        "run",
        mpa_test_image,
        "apply",
        '{"inputs": {}}',
        "--output-path",
        outdir,
    ]

    run_res = subprocess.run(
        run_cmd,
        capture_output=True,
        text=True,
    )
    assert run_res.returncode == 0, run_res.stderr

    log_dir = outdir / "logs"
    assert log_dir.exists()

    # Verify parameters file
    params_file = log_dir / "parameters.json"
    assert params_file.exists()
    with open(params_file) as f:
        params = json.load(f)
        assert params["test_parameter"] == "test_param"
        assert params["steps_config"] == 5

    # Verify metrics file
    metrics_file = log_dir / "metrics.csv"
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
    artifacts_dir = log_dir / "artifacts"
    assert artifacts_dir.exists()

    artifact_file = artifacts_dir / "test_artifact.txt"
    assert artifact_file.exists()

    with open(artifact_file) as f:
        artifact_data = f.read()
        assert artifact_data == "Test artifact content"


def test_mpa_mlflow_backend(mpa_test_image, tmpdir):
    """Test the MPA (Metrics, Parameters, and Artifacts) submodule with MLflow backend."""
    # Point MLflow to a local directory
    run_cmd = [
        "tesseract",
        "run",
        "--env",
        "TESSERACT_MLFLOW_TRACKING_URI=mlruns",
        mpa_test_image,
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


def test_multi_helloworld_endtoend(
    docker_client,
    unit_tesseracts_parent_dir,
    dummy_image_name,
    dummy_network_name,
    docker_cleanup,
):
    """Test that multi_helloworld example can be built, served, and executed."""
    cli_runner = CliRunner(mix_stderr=False)

    # Build Tesseract images
    img_names = []
    for tess_name in ("_multi-tesseract/multi_helloworld", "helloworld"):
        img_name = build_tesseract(
            docker_client,
            unit_tesseracts_parent_dir / tess_name,
            dummy_image_name + f"_{tess_name}",
            tag="sometag",
        )
        img_names.append(img_name)
        assert image_exists(docker_client, img_name)
        docker_cleanup["images"].append(img_name)

    config = get_config()
    docker = config.docker_executable

    result = subprocess.run(
        [*docker, "network", "create", dummy_network_name],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0, result.stderr
    docker_cleanup["networks"].append(dummy_network_name)

    # Serve target Tesseract
    multi_helloworld_tesseract_img_name, helloworld_tesseract_img_name = img_names
    result = cli_runner.invoke(
        app,
        [
            "serve",
            helloworld_tesseract_img_name,
            "--network",
            dummy_network_name,
            "--network-alias",
            "helloworld",
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0, result.output
    docker_cleanup["containers"].append(json.loads(result.output)["container_name"])

    payload = json.dumps(
        {
            "inputs": {
                "name": "you",
                "helloworld_tesseract_url": "http://helloworld:8000",
            }
        }
    )

    # Run multi_helloworld Tesseract
    result = cli_runner.invoke(
        app,
        [
            "run",
            multi_helloworld_tesseract_img_name,
            "apply",
            payload,
            "--network",
            dummy_network_name,
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0, result.output
    assert "The helloworld Tesseract says: Hello you!" in result.output


def test_tesseractarg_endtoend(
    docker_client,
    unit_tesseracts_parent_dir,
    dummy_image_name,
    dummy_network_name,
    docker_cleanup,
):
    """Test that tesseractarg example can be built and executed, calling helloworld tesseract."""
    cli_runner = CliRunner(mix_stderr=False)

    # Build Tesseract images
    img_names = []
    for tess_name in ("tesseractarg", "helloworld"):
        img_name = build_tesseract(
            docker_client,
            unit_tesseracts_parent_dir / tess_name,
            dummy_image_name + f"_{tess_name}",
            tag="sometag",
        )
        img_names.append(img_name)
        assert image_exists(docker_client, img_name)
        docker_cleanup["images"].append(img_name)

    tesseractarg_img_name, helloworld_img_name = img_names

    # Create Docker network
    config = get_config()
    docker = config.docker_executable

    result = subprocess.run(
        [*docker, "network", "create", dummy_network_name],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0, result.stderr
    docker_cleanup["networks"].append(dummy_network_name)

    # Serve helloworld Tesseract on the shared network
    result = cli_runner.invoke(
        app,
        [
            "serve",
            helloworld_img_name,
            "--network",
            dummy_network_name,
            "--network-alias",
            "helloworld",
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0, result.output
    serve_meta = json.loads(result.output)
    docker_cleanup["containers"].append(serve_meta["container_name"])

    # Create payload for tesseractarg with target URL pointing to helloworld via network alias
    payload = '{"inputs": {"target": {"type": "url", "url": "http://helloworld:8000"}}}'

    # Run tesseractarg Tesseract on the same network
    result = cli_runner.invoke(
        app,
        [
            "run",
            tesseractarg_img_name,
            "apply",
            payload,
            "--network",
            dummy_network_name,
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0, result.output

    output_data = json.loads(result.output)
    assert output_data["result"] == "Hello Alice! Hello Bob!"
