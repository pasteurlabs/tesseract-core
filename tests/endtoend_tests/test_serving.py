# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for serving Tesseracts."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import requests

from tesseract_core.sdk.cli import app
from tesseract_core.sdk.docker_client import _get_docker_executable


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


@pytest.mark.parametrize("user", [None, "root", "1000:1000"])
def test_run_as_user(cli_runner, docker_client, built_image_name, user, docker_cleanup):
    """Ensure we can run a basic Tesseract image as any user."""
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


@pytest.mark.parametrize("memory", ["512m", "1g", "256m"])
def test_serve_with_memory(
    cli_runner, docker_client, built_image_name, memory, docker_cleanup
):
    """Ensure we can serve a Tesseract with memory limits."""
    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            built_image_name,
            "--memory",
            memory,
        ],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr

    serve_meta = json.loads(run_res.stdout)
    container = docker_client.containers.get(serve_meta["container_name"])
    docker_cleanup["containers"].append(container)

    # Verify memory limit was set on container
    container_inspect = docker_client.containers.get(container.id)
    memory_limit = container_inspect.attrs["HostConfig"]["Memory"]

    # Convert memory string to bytes for comparison
    memory_value = int(memory[:-1])
    memory_unit = memory[-1].lower()
    expected_bytes = memory_value * (1024**2 if memory_unit == "m" else 1024**3)

    assert memory_limit == expected_bytes


def test_tesseract_serve_pipeline(
    cli_runner, docker_client, built_image_name, docker_cleanup
):
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
def test_tesseract_teardown_multiple(cli_runner, built_image_name, tear_all):
    """Teardown multiple served tesseracts."""
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


def test_tesseract_serve_ports_error(cli_runner, built_image_name):
    """Check error handling for serve -p flag."""
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
def test_tesseract_serve_ports(
    cli_runner, built_image_name, port, docker_cleanup, free_port
):
    """Try to serve multiple Tesseracts on multiple ports."""
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
    cli_runner,
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


def test_tesseract_serve_interop(
    cli_runner, built_image_name, dummy_network_name, docker_client, docker_cleanup
):
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
            f'import requests; requests.get("http://tess_2:{tess_2.api_port}/health").raise_for_status()',
        ]
    )
    assert returncode == 0, stdout.decode()

    returncode, stdout = tess_2.exec_run(
        [
            "python",
            "-c",
            f'import requests; requests.get("http://tess_1:{tess_1.api_port}/health").raise_for_status()',
        ]
    )
    assert returncode == 0, stdout.decode()


def test_serve_nonstandard_host_ip(
    cli_runner, docker_client, built_image_name, docker_cleanup, free_port
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
