# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import string
import subprocess
from pathlib import Path
from shutil import copytree
from typing import Any

import pytest

from tesseract_core.sdk import docker_cli_wrapper

here = Path(__file__).parent

UNIT_TESSERACT_PATH = here / ".." / "examples"
UNIT_TESSERACTS = [Path(tr).stem for tr in UNIT_TESSERACT_PATH.glob("*/")]


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
        try:
            docker = docker_cli_wrapper.CLIDockerClient()
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
    orig_path = get_config().tesseract_api_path
    # default may have been used and tesseract_api.py is not guaranteed to exist
    # therefore, we only pass the original path in cleanup if not equal to default
    if orig_path != Path("tesseract_api.py"):
        orig_config_kwargs |= {"tesseract_api_path": orig_path}
    api_path = Path(dummy_tesseract_package / "tesseract_api.py").resolve()

    try:
        # Configure via envvar so we also propagate it to subprocesses
        os.environ["TESSERACT_API_PATH"] = str(api_path)
        update_config()
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

    orig_api_path = get_config().tesseract_api_path
    orig_cwd = os.getcwd()

    # Ensure TESSERACT_API_PATH is not set with python os
    if "TESSERACT_API_PATH" in os.environ:
        del os.environ["TESSERACT_API_PATH"]

    try:
        os.chdir(dummy_tesseract_package)
        update_config()
        yield
    finally:
        update_config(tesseract_api_path=orig_api_path)
        os.chdir(orig_cwd)


@pytest.fixture
def free_port():
    """Find a free port to use for HTTP."""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def docker_client():
    return docker_cli_wrapper.CLIDockerClient()


@pytest.fixture
def dummy_image_name(docker_client):
    """Create a dummy image name, and clean up after the test."""
    image_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=16))
    image_name = f"tmp_tesseract_image_{image_id}"
    try:
        yield image_name
    finally:
        if os.environ.get("TESSERACT_KEEP_BUILD_CACHE", "0").lower() not in (
            "1",
            "true",
        ):
            try:
                docker_client.images.remove(image_name)
            except RuntimeError as ex:
                # If image is not found is in the exception error string, pass
                if "Cannot remove image" in str(ex).lower():
                    pass


@pytest.fixture(scope="module")
def shared_dummy_image_name(docker_client):
    """Create a dummy image name, and clean up after all tests."""
    image_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=16))
    image_name = f"tmp_tesseract_image_{image_id}"
    try:
        yield image_name
    finally:
        if os.environ.get("TESSERACT_KEEP_BUILD_CACHE", "0").lower() not in (
            "1",
            "true",
        ):
            try:
                docker_client.images.remove(image_name)
            except RuntimeError as ex:
                # If image is not found is in the exception error string, pass
                if "Cannot remove image" in str(ex).lower():
                    pass


@pytest.fixture
def mocked_docker(monkeypatch):
    """Mock Docker Wrapper class."""
    from tesseract_core.sdk import engine

    class MockedContainer(docker_cli_wrapper.CLIDockerClient.Containers.Container):
        """Mock Container class."""

        def __init__(self, return_args: dict):
            self.return_args = return_args

        def wait(self, **kwargs: Any):
            """Mock wait method for Container."""
            return {"StatusCode": 0, "Error": None}

        @property
        def attrs(self):
            """Mock attrs method for Container."""
            return {"Config": {"Env": ["TESSERACT_NAME=vectoradd"]}}

        def logs(self, stderr=False, stdout=False, **kwargs: Any):
            """Mock logs method for Container."""
            res_stdout = json.dumps(self.return_args).encode("utf-8")
            res_stderr = "hello tesseract"
            if stdout and stderr:
                return res_stdout, res_stderr
            if stdout:
                return res_stdout
            return res_stderr

        def remove(self, **kwargs: Any):
            """Mock remove method for Container."""
            pass

    created_ids = set()

    class MockedDocker:
        """Mock CLIDockerClient class."""

        @staticmethod
        def info():
            """Mock info method for DockerClient."""
            return docker_cli_wrapper.CLIDockerClient.info()

        class images:
            """Mock of CLIDockerClient.images."""

            @staticmethod
            def get(name: str) -> None:
                """Mock of CLIDockerClient.Images.get."""
                return MockedDocker.images.list()[0]

            @staticmethod
            def list() -> list[docker_cli_wrapper.CLIDockerClient.Images.Image]:
                """Mock of CLIDockerClient.Images.list."""
                return [
                    docker_cli_wrapper.CLIDockerClient.Images.Image(
                        {
                            "Id": "sha256:123456789abcdef",
                            "RepoTags": ["vectoradd:latest"],
                            "Size": 123456789,
                            "Config": {"Env": ["TESSERACT_NAME=vectoradd"]},
                        },
                    ),
                    docker_cli_wrapper.CLIDockerClient.Images.Image(
                        {
                            "Id": "sha256:48932484029303",
                            "RepoTags": ["hello-world:latest"],
                            "Size": 43829489032,
                            "Config": {"Env": ["PATH=/fake-path"]},
                        },
                    ),
                ]

            @staticmethod
            def buildx(*args, **kwargs) -> str:
                return docker_cli_wrapper.CLIDockerClient.Images.buildx(*args, **kwargs)

        class containers:
            @staticmethod
            def get(name: str) -> MockedContainer:
                """Mock of CLIDockerClient.Containers.get."""
                return MockedDocker.containers.list()[0]

            @staticmethod
            def list() -> list[MockedContainer]:
                """Mock of CLIDockerClient.Containers.list."""
                return [MockedContainer({"TESSERACT_NAME": "vectoradd"})]

            @staticmethod
            def run(**kwargs: Any) -> bytes:
                """Mock run method for containers."""
                container = MockedContainer(kwargs)
                if kwargs.get("detach", False):
                    return container
                return container.logs(stdout=True, stderr=True)

        class compose:
            @staticmethod
            def list() -> dict:
                """Return ids of all created tesseracts projects."""
                return created_ids

            @staticmethod
            def up(compose_fpath: str, project_name: str) -> str:
                """Mock of CLIDockerClient.Compose.up."""
                created_ids.add(project_name)
                return project_name

            @staticmethod
            def down(project_id: str) -> bool:
                """Mock of CLIDockerClient.Compose.down."""
                created_ids.remove(project_id)
                return True

            @staticmethod
            def exists(project_id: str) -> bool:
                """Mock of CLIDockerClient.Compose.exists."""
                return project_id in created_ids

    def mocked_subprocess_run(*args, **kwargs):
        """Mock subprocess.run."""
        return subprocess.CompletedProcess(
            args=args, returncode=0, stderr=b"", stdout=b""
        )

    mock_instance = MockedDocker()
    monkeypatch.setattr(docker_cli_wrapper.subprocess, "run", mocked_subprocess_run)
    monkeypatch.setattr(engine, "docker_client", MockedDocker)

    yield mock_instance
