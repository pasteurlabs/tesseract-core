# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import json
import os
import random
import shutil
import string
import subprocess
import sys
import time
from pathlib import Path
from shutil import copytree
from textwrap import indent
from traceback import format_exception
from typing import Any

import pytest
import requests

# NOTE: Do NOT import tesseract_core here, as it will cause typeguard to fail

here = Path(__file__).parent

UNIT_TESSERACT_PATH = here / ".." / "examples"
UNIT_TESSERACTS = [
    tr.stem for tr in UNIT_TESSERACT_PATH.glob("*/") if not tr.stem.startswith("_")
]


def pytest_addoption(parser):
    parser.addoption(
        "--always-run-endtoend",
        action="store_true",
        dest="run_endtoend",
        help="Never skip end-to-end tests",
        default=None,
    )
    parser.addoption(
        "--skip-endtoend",
        action="store_false",
        dest="run_endtoend",
        help="Skip end-to-end tests",
    )


def pytest_collection_modifyitems(config, items):
    """Ensure that endtoend tests are run last (expensive!)."""
    # Map items to containing directory
    dir_mapping = {item: Path(item.module.__file__).parent.stem for item in items}

    # Sort items based on directory
    sorted_items = sorted(items, key=lambda item: dir_mapping[item] == "endtoend_tests")
    items[:] = sorted_items

    # Add skip marker to endtoend tests if not explicitly enabled
    # or if Docker is not available
    def has_docker():
        from tesseract_core.sdk import docker_client as docker_client_module

        try:
            docker = docker_client_module.CLIDockerClient()
            docker.info()
            return True
        except Exception:
            return False

    run_endtoend = config.getvalue("run_endtoend")

    if run_endtoend is None:
        # tests may be skipped if Docker is not available
        run_endtoend = has_docker()
        skip_reason = "Docker is required for this test"
    elif not run_endtoend:
        skip_reason = "Skipping end-to-end tests"

    if not run_endtoend:
        for item in items:
            if dir_mapping[item] == "endtoend_tests":
                item.add_marker(pytest.mark.skip(reason=skip_reason))

    # Auto-skip GPU-marked tests when no CUDA GPU is present, so a full test run
    # on a GPU-less machine (including CI) is a benign skip rather than a failure
    # and does not depend on passing `-m "not gpu"`.
    if not _has_cuda_gpu():
        skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
        for item in items:
            if item.get_closest_marker("gpu") is not None:
                item.add_marker(skip_gpu)


def _has_cuda_gpu() -> bool:
    """Whether a CUDA GPU is visible on this host (best effort, via nvidia-smi).

    Probed through ``nvidia-smi`` rather than importing CuPy/torch so the check
    stays dependency-free and cheap on GPU-less runners.
    """
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return (
            subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                timeout=10,
            ).returncode
            == 0
        )
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def tesseract_output_dir(tmp_path_factory):
    """Set the Tesseract output directory for the session."""
    output_path = tmp_path_factory.mktemp("output_path")
    os.environ["TESSERACT_OUTPUT_PATH"] = str(output_path)
    yield output_path


@pytest.fixture(autouse=True)
def reset_config():
    """Reset the runtime configuration before each test."""
    import tesseract_core.runtime.config
    import tesseract_core.sdk.config

    initial_config_sdk = tesseract_core.sdk.config._current_config
    initial_config_runtime = tesseract_core.runtime.config._current_config

    try:
        yield
    finally:
        # Reset the SDK config
        tesseract_core.sdk.config._current_config = initial_config_sdk
        tesseract_core.sdk.config._config_overrides.clear()

        # Reset the runtime config
        tesseract_core.runtime.config._current_config = initial_config_runtime
        tesseract_core.runtime.config._config_overrides.clear()


@pytest.fixture(scope="session")
def unit_tesseract_names():
    """Return all unit tesseract names."""
    return UNIT_TESSERACTS


@pytest.fixture(scope="session", params=UNIT_TESSERACTS)
def unit_tesseract_path(request) -> Path:
    """Parametrized fixture to return all unit tesseracts."""
    # pass only tesseract names as params to get prettier test names
    return UNIT_TESSERACT_PATH / request.param


@pytest.fixture(scope="session")
def unit_tesseracts_parent_dir(request) -> Path:
    """Fixture that return parent dir of unit tesseracts."""
    return UNIT_TESSERACT_PATH


@pytest.fixture()
def dummy_docker_file(tmp_path):
    """Create a dummy Dockerfile for testing."""
    dockerfile_path = tmp_path / "Dockerfile"
    dockerfile_content = """
        FROM alpine

        ENTRYPOINT ["/bin/sh", "-c"]

        # Set environment variables
        ENV TESSERACT_NAME="dummy-tesseract"
        """
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)
    return dockerfile_path


@pytest.fixture(scope="session", autouse=True)
def use_this_environments_console_scripts():
    """Make shelled-out `tesseract`/`tesseract-runtime` this environment's.

    Several tests invoke the console scripts as real subprocesses, which is the
    point -- it covers the entry points, unlike invoking a module. But a bare
    name is resolved through PATH, and anyone with tesseract-core installed
    globally (as a uv tool or pipx app) has that copy first, with its own
    site-packages. The tests then exercise code that is not the code under test.
    In CI everything runs under `uv run`, which puts the project environment
    first, so this only ever bites locally -- and intermittently, since loading
    a tesseract_api.py sets PYTHONPATH as a side effect, which sometimes lends
    the shadowing copy enough of this environment to work.

    Applied to os.environ for the whole session rather than offered as an opt-in
    fixture, so that tests added later cannot forget it. The failure it prevents
    is silent, so it must not be possible to omit.
    """
    scripts_dir = Path(sys.executable).parent
    if not list(scripts_dir.glob("tesseract-runtime*")):
        # Prepending a directory that lacks the scripts is not a harmless no-op:
        # PATH lookup falls straight through to the next entry, which is the
        # globally installed copy this exists to avoid.
        raise RuntimeError(
            f"No tesseract-runtime console script in {scripts_dir}, so PATH would "
            "fall through to any other copy on it. Install tesseract-core into "
            "the environment running the tests (e.g. `uv sync`)."
        )

    original_path = os.environ["PATH"]
    os.environ["PATH"] = f"{scripts_dir}{os.pathsep}{original_path}"
    try:
        yield
    finally:
        os.environ["PATH"] = original_path


@pytest.fixture(scope="session")
def dummy_tesseract_location():
    """Return the dummy tesseract location."""
    return here / "dummy_tesseract"


@pytest.fixture
def dummy_tesseract_package(tmpdir, dummy_tesseract_location):
    """Create a dummy tesseract package on disk for testing."""
    copytree(dummy_tesseract_location, tmpdir, dirs_exist_ok=True)
    return Path(tmpdir)


@pytest.fixture
def dummy_tesseract_module(dummy_tesseract_package):
    """Create a dummy tesseract module for testing."""
    from tesseract_core.runtime.core import load_module_from_path

    return load_module_from_path(dummy_tesseract_package / "tesseract_api.py")


@pytest.fixture
def dummy_tesseract(dummy_tesseract_package, monkeypatch):
    """Set tesseract_api_path env var for testing purposes."""
    api_path = Path(dummy_tesseract_package / "tesseract_api.py").resolve()
    monkeypatch.setenv("TESSERACT_API_PATH", str(api_path))
    yield


@pytest.fixture
def free_port():
    """Find a free port to use for HTTP."""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def serve_in_subprocess():
    """Serve a Tesseract runtime over HTTP in a subprocess.

    Yields a context manager ``serve(api_file, port, num_workers=1,
    timeout=30.0, env=None)`` that starts the server, waits for ``/health``, and
    yields its base URL. ``env`` supplies extra ``TESSERACT_*`` variables (e.g.
    to set the output format or GPU transport), layered on the current
    environment. The server is torn down and its output printed on exit.
    """
    from contextlib import contextmanager

    @contextmanager
    def serve(api_file, port, num_workers=1, timeout=30.0, env=None):
        proc = None
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from tesseract_core.runtime.serve import serve; "
                    f"serve(host='localhost', port={port}, num_workers={num_workers})",
                ],
                env={**os.environ, "TESSERACT_API_PATH": str(api_file), **(env or {})},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    out, err = proc.communicate()
                    raise RuntimeError(
                        f"server exited early (code {proc.returncode}):\n"
                        f"{out.decode()}{err.decode()}"
                    )
                try:
                    resp = requests.get(f"http://localhost:{port}/health")
                except requests.exceptions.ConnectionError:
                    time.sleep(0.1)
                else:
                    if resp.status_code == 200:
                        break
            else:
                raise TimeoutError("Server did not start in time")

            yield f"http://localhost:{port}"

        finally:
            if proc is not None:
                proc.terminate()
                stdout, stderr = proc.communicate()
                print(stdout.decode())
                print(stderr.decode())
                proc.wait(timeout=5)

    return serve


@pytest.fixture(scope="module")
def cli_runner():
    import importlib.metadata

    from typer.testing import CliRunner

    kwargs = {}
    click_version = importlib.metadata.version("click").split(".")
    click_version = (int(click_version[0]), int(click_version[1]))
    if click_version < (8, 2):
        kwargs.update({"mix_stderr": False})
    return CliRunner(**kwargs)


@pytest.fixture(scope="session")
def docker_client():
    from tesseract_core.sdk import docker_client as docker_client_module

    return docker_client_module.CLIDockerClient()


@pytest.fixture
def docker_volume(docker_client):
    # Create the Docker volume
    volume_name = f"test_volume_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}"
    volume = docker_client.volumes.create(name=volume_name)
    try:
        yield volume
    finally:
        volume.remove(force=True)


@pytest.fixture(scope="session")
def docker_cleanup_session(docker_client, request):
    """Clean up all tesseracts created by the tests after the session exits."""
    return _docker_cleanup(docker_client, request)


@pytest.fixture(scope="module")
def docker_cleanup_module(docker_client, request):
    """Clean up all tesseracts created by the tests after the module exits."""
    return _docker_cleanup(docker_client, request)


@pytest.fixture
def docker_cleanup(docker_client, request):
    """Clean up all tesseracts created by the tests after the test exits."""
    return _docker_cleanup(docker_client, request)


def _docker_cleanup(docker_client, request):
    """Clean up all tesseracts created by the tests."""
    # Shared object to track what objects need to be cleaned up in each test
    context = {
        "images": [],
        "containers": [],
        "volumes": [],
        "networks": [],
    }

    def pprint_exc(e: BaseException) -> str:
        """Pretty print exception."""
        return "".join(
            indent(line, "  ") for line in format_exception(type(e), e, e.__traceback__)
        )

    def cleanup_func():
        failures = []

        # Remove containers
        for container in context["containers"]:
            try:
                if isinstance(container, str):
                    container_obj = docker_client.containers.get(container)
                else:
                    container_obj = container

                container_obj.remove(v=True, force=True)
            except Exception as e:
                failures.append(
                    f"Failed to remove container {container}: {pprint_exc(e)}"
                )

        # Remove images
        for image in context["images"]:
            try:
                if isinstance(image, str):
                    image_obj = docker_client.images.get(image)
                else:
                    image_obj = image

                docker_client.images.remove(image_obj.id)
            except Exception as e:
                failures.append(f"Failed to remove image {image}: {pprint_exc(e)}")

        # Remove volumes
        for volume in context["volumes"]:
            try:
                if isinstance(volume, str):
                    volume_obj = docker_client.volumes.get(volume)
                else:
                    volume_obj = volume

                volume_obj.remove(force=True)
            except Exception as e:
                failures.append(f"Failed to remove volume {volume}: {pprint_exc(e)}")

        from tesseract_core.sdk.config import get_config

        config = get_config()
        docker_cmd = config.docker_executable
        for network in context["networks"]:
            try:
                _ = subprocess.run(
                    [*docker_cmd, "network", "rm", network, "--force"],
                    check=True,
                    capture_output=True,
                )
            except Exception as e:
                failures.append(f"Failed to remove network {network}: {pprint_exc(e)}")

        if failures:
            raise RuntimeError(
                "Failed to clean up some Docker objects during test teardown:\n"
                + "\n".join(failures)
            )

    request.addfinalizer(cleanup_func)
    return context


@pytest.fixture(scope="session")
def built_image_name(docker_client, docker_cleanup_session, dummy_tesseract_location):
    """Build the dummy Tesseract image once for the entire test session."""
    from endtoend_tests.common import build_tesseract, image_exists

    image_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=16))
    image_name = f"tmp_tesseract_image_{image_id}"
    image_tag = build_tesseract(docker_client, dummy_tesseract_location, image_name)
    assert image_exists(docker_client, image_tag)
    docker_cleanup_session["images"].append(image_tag)
    yield image_tag


@pytest.fixture
def dummy_image_name():
    """Create a dummy image name."""
    image_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=16))
    image_name = f"tmp_tesseract_image_{image_id}"
    yield image_name


@pytest.fixture(scope="module")
def shared_dummy_image_name():
    """Create a dummy image name."""
    image_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=16))
    image_name = f"tmp_tesseract_image_{image_id}"
    yield image_name


@pytest.fixture
def dummy_network_name():
    """Create a dummy image name."""
    network_name = "".join(random.choices(string.ascii_lowercase + string.digits, k=16))
    network_name = f"tmp_tesseract_network_{network_name}"
    yield network_name


@pytest.fixture
def mocked_docker(monkeypatch):
    """Mock CLIDockerClient class."""
    import tesseract_core.sdk.docker_client
    from tesseract_core.sdk import engine, serving
    from tesseract_core.sdk.docker_client import Container, Image, NotFound

    class MockedContainer(Container):
        """Mock Container class."""

        def __init__(self, return_args: dict):
            self.return_args = return_args

        def wait(self, **kwargs: Any):
            """Mock wait method for Container."""
            return {"StatusCode": 0, "Error": None}

        @property
        def name(self):
            """Mock name property for Container."""
            return json.dumps({**self.return_args, "name": "vectoradd"})

        @property
        def attrs(self):
            """Mock attrs method for Container."""
            return {"Config": {"Env": ["TESSERACT_NAME=vectoradd"]}}

        def logs(self, stderr=False, stdout=False, **kwargs: Any) -> bytes:
            """Mock logs method for Container."""
            res_stdout = json.dumps(self.return_args).encode("utf-8")
            res_stderr = b"hello tesseract"
            if stdout and stderr:
                return res_stdout + res_stderr
            if stdout:
                return res_stdout
            return res_stderr

        def remove(self, **kwargs: Any):
            """Mock remove method for Container."""
            pass

    class MockedDocker:
        """Mock CLIDockerClient class."""

        @staticmethod
        def info() -> tuple:
            """Mock info method for DockerClient."""
            return "", ""

        class volumes:
            """Mock of CLIDockerClient.volumes."""

            @staticmethod
            def create(name: str) -> Any:
                """Mock of CLIDockerClient.volumes.create."""
                return {"Name": name}

            @staticmethod
            def get(name: str) -> Any:
                """Mock of CLIDockerClient.volumes.get."""
                if "/" in name:
                    raise NotFound(f"Volume {name} not found")
                return {"Name": name}

            @staticmethod
            def list() -> list[Any]:
                """Mock of CLIDockerClient.volumes.list."""
                return [{"Name": "test_volume"}]

        class images:
            """Mock of CLIDockerClient.images."""

            @staticmethod
            def get(name: str) -> Image:
                """Mock of CLIDockerClient.images.get."""
                return MockedDocker.images.list()[0]

            @staticmethod
            def list() -> list[Image]:
                """Mock of CLIDockerClient.images.list."""
                return [
                    Image.from_dict(
                        {
                            "Id": "sha256:123456789abcdef",
                            "RepoTags": ["vectoradd:latest"],
                            "Size": 123456789,
                            "Config": {"Env": ["TESSERACT_NAME=vectoradd"]},
                        },
                    ),
                    Image.from_dict(
                        {
                            "Id": "sha256:48932484029303",
                            "RepoTags": ["hello-world:latest"],
                            "Size": 43829489032,
                            "Config": {"Env": ["PATH=/fake-path"]},
                        },
                    ),
                ]

            @staticmethod
            def buildx(*args, **kwargs) -> Image:
                return MockedDocker.images.list()[0]

        class containers:
            @staticmethod
            def get(name: str) -> MockedContainer:
                """Mock of CLIDockerClient.containers.get."""
                if name == "vectoradd":
                    return MockedContainer({"TESSERACT_NAME": "vectoradd"})
                raise NotFound(f"Container {name} not found")

            @staticmethod
            def list() -> list[MockedContainer]:
                """Mock of CLIDockerClient.containers.list."""
                return [MockedContainer({"TESSERACT_NAME": "vectoradd"})]

            @staticmethod
            def run(**kwargs: Any) -> MockedContainer | tuple[bytes, bytes]:
                """Mock run method for containers."""
                container = MockedContainer(kwargs)
                if kwargs.get("detach", False):
                    return container
                return (
                    container.logs(stdout=True, stderr=False),
                    container.logs(stdout=False, stderr=True),
                )

    mock_instance = MockedDocker()
    monkeypatch.setattr(engine, "docker_client", mock_instance)
    monkeypatch.setattr(engine, "is_podman", lambda: False)
    monkeypatch.setattr(
        tesseract_core.sdk.docker_client, "CLIDockerClient", MockedDocker
    )

    def hacked_get(url, *args, **kwargs):
        if url.endswith("/health"):
            # Simulate a successful health check
            return type("Response", (), {"status_code": 200, "json": lambda: {}})()
        raise NotImplementedError(f"Mocked get request to {url} not implemented")

    monkeypatch.setattr(serving.requests, "get", hacked_get)

    yield mock_instance


@pytest.fixture
def mocked_cuda(monkeypatch):
    """Mock the CUDA runtime so cuda_ipc encode/decode runs without a GPU.

    Mirrors ``mocked_docker``: rather than reaching into ctypes internals, it
    swaps the plain-Python functions in ``tesseract_core.runtime.cuda.api``
    for an in-process fake, so the ``cuda_ipc`` encoding *policy* exercises its
    real module boundary. Device memory is modelled with Python ``bytearray``s
    keyed by pointer, and every primitive call is recorded so tests can assert
    on the orchestration (which pointer the handle was taken on, that the mapping
    was closed, the copy synchronised before close, etc.).

    Yields a ``FakeCuda`` whose ``.calls`` dict holds the recorded call log and
    whose ``.device_bytes`` / ``.reject_ipc_below`` attributes let a test seed
    device-to-host reads or force the VMM staging fallback.
    """
    from tesseract_core.runtime.cuda import api as cuda_api
    from tesseract_core.runtime.cuda import ipc as cuda_ipc

    IPC_HANDLE_SIZE = cuda_api.IPC_HANDLE_SIZE

    class FakeCuda:
        def __init__(self) -> None:
            self.calls: dict[str, list] = {
                "set_device": [],
                "malloc": [],
                "free": [],
                "memcpy_d2d": [],
                "memcpy_d2h": [],
                "sync": [],
                "alloc_base": [],
                "get_handle": [],
                "open": [],
                "close": [],
                "stage": [],
            }
            # Simulated device memory, keyed by device pointer.
            self._buffers: dict[int, bytearray] = {}
            self._next_ptr = 0xD000
            # Bytes returned by the next device->host copy, if a test seeds them.
            self.device_bytes: bytes | None = None
            # Force ipc_get_mem_handle to reject a pointer (simulating a
            # VMM/pool-backed allocation) unless it is a staging buffer.
            self.reject_non_staging_ipc = False
            self._staging_ptrs: set[int] = set()

        # -- device / memory management ---------------------------------

        def set_device(self, device: int) -> None:
            self.calls["set_device"].append(device)

        def malloc(self, nbytes: int) -> int:
            ptr = self._next_ptr
            self._next_ptr += 0x1000
            self._buffers[ptr] = bytearray(nbytes)
            self.calls["malloc"].append(nbytes)
            return ptr

        def free(self, device_ptr: int) -> None:
            if not device_ptr:
                return
            self.calls["free"].append(device_ptr)
            self._buffers.pop(device_ptr, None)
            self._staging_ptrs.discard(device_ptr)

        def memcpy_device_to_device(self, dst: int, src: int, nbytes: int) -> None:
            self.calls["memcpy_d2d"].append((dst, src, nbytes))

        def memcpy_device_to_host(self, host_ptr: int, src: int, nbytes: int) -> None:
            self.calls["memcpy_d2h"].append((host_ptr, src, nbytes))
            if self.device_bytes is not None:
                ctypes.memmove(host_ptr, self.device_bytes, nbytes)

        def device_synchronize(self) -> None:
            self.calls["sync"].append(True)

        # -- IPC --------------------------------------------------------

        def get_allocation_base(self, device_ptr: int) -> tuple[int, int]:
            self.calls["alloc_base"].append(device_ptr)
            # Report a base 256 bytes below the data pointer so storage_offset
            # arithmetic is exercised with a non-zero value.
            return device_ptr - 256, 4096

        def ipc_get_mem_handle(self, device_ptr: int) -> bytes:
            if self.reject_non_staging_ipc and device_ptr not in self._staging_ptrs:
                raise RuntimeError("cudaIpcGetMemHandle failed: simulated VMM reject")
            self.calls["get_handle"].append(device_ptr)
            return b"\x01" * IPC_HANDLE_SIZE

        def ipc_open_mem_handle(self, handle_bytes: bytes, device: int) -> int:
            self.calls["open"].append((handle_bytes, device))
            return 0x2000  # pretend mapped base pointer

        def ipc_close_mem_handle(self, device_ptr: int) -> None:
            self.calls["close"].append(device_ptr)

        def stage_for_legacy_ipc(self, src_ptr: int, nbytes: int) -> int:
            self.calls["stage"].append((src_ptr, nbytes))
            ptr = 0x9000
            self._staging_ptrs.add(ptr)
            self._buffers[ptr] = bytearray(nbytes)
            return ptr

    fake = FakeCuda()

    for name in (
        "set_device",
        "malloc",
        "free",
        "memcpy_device_to_device",
        "memcpy_device_to_host",
        "device_synchronize",
        "get_allocation_base",
        "ipc_get_mem_handle",
        "ipc_open_mem_handle",
        "ipc_close_mem_handle",
        "stage_for_legacy_ipc",
    ):
        monkeypatch.setattr(cuda_api, name, getattr(fake, name))

    # Each test starts with empty export registries.
    cuda_ipc._CUDA_IPC_EXPORT_REGISTRY.clear()
    cuda_ipc._CUDA_IPC_STAGING_BUFFERS.clear()
    yield fake
    cuda_ipc._CUDA_IPC_EXPORT_REGISTRY.clear()
    cuda_ipc._CUDA_IPC_STAGING_BUFFERS.clear()


@pytest.fixture(scope="module")
def mlflow_server():
    """MLflow server to use in tests."""
    # Check if docker-compose is available
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.fail("docker-compose not available")

    # Start MLflow server with unique project name
    project_name = f"test_mlflow_{int(time.time())}"

    compose_file = (
        Path(__file__).parent.parent / "extra" / "mlflow" / "docker-compose-mlflow.yml"
    )

    try:
        # Start the services
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "-p",
                project_name,
                "up",
                "-d",
            ],
            check=True,
            capture_output=True,
        )

        res = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "-p",
                project_name,
                "ps",
                "--format",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        service_data = json.loads(res.stdout)
        service_port = service_data["Publishers"][0]["PublishedPort"]

        # Note: We don't track containers/volumes here because docker-compose down -v
        # will handle cleanup automatically in the finally block

        # Wait for MLflow to be ready (with timeout)
        tracking_uri = f"http://localhost:{service_port}"
        max_wait = 30  # seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = requests.get(tracking_uri, timeout=2)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            pytest.fail(f"MLflow server did not become ready within {max_wait}s")

        yield tracking_uri

    finally:
        # Get logs for debugging
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "-p", project_name, "logs"],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        # Stop and remove containers
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "-p",
                project_name,
                "down",
                "-v",
            ],
            capture_output=True,
        )
