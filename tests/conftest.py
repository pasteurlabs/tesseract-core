# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import string
import subprocess
from pathlib import Path
from shutil import copytree
from textwrap import indent
from traceback import format_exception
from typing import Any

import pytest

# NOTE: Do NOT import tesseract_core here, as it will cause typeguard to fail

here = Path(__file__).parent

UNIT_TESSERACT_PATH = here / ".." / "examples"
UNIT_TESSERACTS = [
    Path(tr).parent.stem for tr in UNIT_TESSERACT_PATH.glob("*/tesseract_api.py")
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
    parser.addoption(
        "--tesseract-dir",
        action="store",
        default=None,
        dest="tesseract_dir",
        help="Directory of your tesseract api",
    )


@pytest.fixture
def tesseract_dir(request):
    """Return the tesseract directory."""
    # This is used to set the tesseract_dir fixture
    # in the pytest_generate_tests function above.
    tesseract_dir = request.config.getoption("tesseract_dir")
    if tesseract_dir:
        return Path(tesseract_dir)
    return None


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


@pytest.fixture(scope="session", autouse=True)
def set_tesseract_output_dir(tmp_path_factory):
    """Set the Tesseract output directory for the session."""
    from tesseract_core.runtime.config import update_config

    output_path = tmp_path_factory.mktemp("output_path")
    os.environ["TESSERACT_OUTPUT_PATH"] = str(output_path)
    update_config(output_path=str(output_path))


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
def dummy_tesseract(dummy_tesseract_package):
    """Set tesseract_api_path env var for testing purposes."""
    from tesseract_core.runtime.config import get_config, update_config

    orig_config_kwargs = {}
    orig_path = get_config().api_path
    # default may have been used and tesseract_api.py is not guaranteed to exist
    # therefore, we only pass the original path in cleanup if not equal to default
    if orig_path != Path("tesseract_api.py"):
        orig_config_kwargs |= {"api_path": orig_path}
    api_path = Path(dummy_tesseract_package / "tesseract_api.py").resolve()

    try:
        # Configure via envvar so we also propagate it to subprocesses
        os.environ["TESSERACT_API_PATH"] = str(api_path)
        update_config(api_path=api_path)
        yield
    finally:
        # As this is used by an auto-use fixture, cleanup may happen
        # after dummy_tesseract_noenv has already unset
        if "TESSERACT_API_PATH" in os.environ:
            del os.environ["TESSERACT_API_PATH"]
        update_config(**orig_config_kwargs)


@pytest.fixture
def dummy_tesseract_noenv(dummy_tesseract_package):
    """Use without tesseract_api_path to test handling of this."""
    from tesseract_core.runtime.config import get_config, update_config

    orig_api_path = get_config().api_path
    orig_cwd = os.getcwd()

    # Ensure TESSERACT_API_PATH is not set with python os
    if "TESSERACT_API_PATH" in os.environ:
        del os.environ["TESSERACT_API_PATH"]

    try:
        os.chdir(dummy_tesseract_package)
        update_config()
        yield
    finally:
        update_config(api_path=orig_api_path)
        os.chdir(orig_cwd)


@pytest.fixture
def free_port():
    """Find a free port to use for HTTP."""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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


@pytest.fixture
def dummy_image_name():
    """Create a dummy image name, and clean up after the test."""
    image_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=16))
    image_name = f"tmp_tesseract_image_{image_id}"
    yield image_name


@pytest.fixture(scope="module")
def shared_dummy_image_name():
    """Create a dummy image name, and clean up after all tests."""
    image_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=16))
    image_name = f"tmp_tesseract_image_{image_id}"
    yield image_name


@pytest.fixture
def dummy_network_name():
    """Create a dummy image name, and clean up after the test."""
    network_name = "".join(random.choices(string.ascii_lowercase + string.digits, k=16))
    network_name = f"tmp_tesseract_network_{network_name}"
    yield network_name


@pytest.fixture
def mocked_docker(monkeypatch):
    """Mock CLIDockerClient class."""
    import tesseract_core.sdk.docker_client
    from tesseract_core.sdk import engine
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

    engine.requests.get = hacked_get

    yield mock_instance
