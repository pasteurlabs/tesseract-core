# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End to end tests for docker cli wrapper."""

import subprocess
import textwrap
from contextlib import closing
from pathlib import Path

import docker
import pytest
from common import image_exists

from tesseract_core.sdk.docker_client import (
    ContainerError,
    ImageNotFound,
    build_docker_image,
)


@pytest.fixture(scope="module")
def docker_py_client():
    # Create a Docker client using the docker-py library
    # You may need to set the environment variable $DOCKER_HOST if encountering issues
    with closing(docker.from_env()) as client:
        yield client


@pytest.fixture()
def docker_client_built_image_name(
    docker_client,
    dummy_tesseract_location,
    dummy_docker_file,
):
    """Build the dummy image for the tests."""
    image_name = "docker_client_create_image_test:dummy"

    build_docker_image(dummy_tesseract_location, image_name, dummy_docker_file)
    try:
        yield image_name
    finally:
        try:
            docker_client.images.remove(image_name)
        except ImageNotFound:
            # already removed
            pass


def test_get_image(docker_client, docker_client_built_image_name, docker_py_client):
    """Test image retrieval."""
    # Get the image
    image = docker_client.images.get(docker_client_built_image_name)
    assert image is not None
    assert docker_client_built_image_name in image.tags

    docker_py_image = docker_py_client.images.get(docker_client_built_image_name)
    assert docker_py_image is not None

    non_sha_id = docker_py_image.short_id.split(":")[1]
    docker_image_short = docker_client.images.get(docker_py_image.short_id)
    docker_image_non_sha = docker_client.images.get(non_sha_id)
    assert docker_image_short is not None
    assert docker_image_non_sha is not None

    assert image.id == docker_py_image.id
    assert image.short_id == docker_py_image.short_id
    assert image.tags == docker_py_image.tags
    assert docker_image_short.id == docker_image_non_sha.id
    # Check the repr function
    assert (
        str(image)
        == f"Image(id='{image.id}', short_id='{image.short_id}', tags={image.tags}, attrs={image.attrs})"
    )

    # Check that every image listed by docker cli can be found
    # Dangling images are not listed by the client
    image_ids = subprocess.run(
        [
            "docker",
            "images",
            "--filter",
            "dangling=false",
            "-q",
        ],  # List only image IDs
        capture_output=True,
        text=True,
        check=True,
    )
    image_ids = image_ids.stdout.strip().split("\n")
    # Filter list to exclude empty strings.
    image_ids = [image_id for image_id in image_ids if image_id]
    for image_id in image_ids:
        assert image_exists(docker_client, image_id, tesseract_only=False)

    image_list = docker_client.images.list(tesseract_only=False)
    assert image_list is not None
    assert len(image_list) > 0
    docker_py_image_list = docker_py_client.images.list()
    assert docker_py_image_list is not None
    # Check that every image in image_list is also in docker_py_image_list
    for image in image_list:
        assert image.id in [img.id for img in docker_py_image_list]


def test_create_image(
    docker_client,
    docker_py_client,
    dummy_tesseract_location,
    dummy_docker_file,
    docker_client_built_image_name,
):
    """Test image building, getting, and removing.

    Validate image existence in both docker_py_client and docker_client to ensure
    handling of names/ids are the same.
    """
    image1, image2 = None, None
    # Create an image
    try:
        image = docker_client.images.get(docker_client_built_image_name)
        assert image is not None
        assert docker_client_built_image_name in image.tags

        image_id_obj = docker_client.images.get(image.id)
        image_short_id_obj = docker_client.images.get(image.short_id)
        assert image == image_id_obj
        assert image_short_id_obj == image
        assert image_exists(docker_client, docker_client_built_image_name)
        assert image_exists(docker_py_client, docker_client_built_image_name)

        # Create a second image with no label
        # Check that :latest gets added automatically
        image1_name = "docker_client_create_image_test"
        docker_client.images.buildx(
            dummy_tesseract_location, image1_name, dummy_docker_file
        )
        image1 = docker_client.images.get(image1_name)
        image1_name = image1_name + ":latest"
        assert image1 is not None
        assert image1_name in image1.tags
        assert image_exists(docker_client, image1_name)
        assert image_exists(docker_py_client, image1_name)

        # Create a third image with prefixed with repo url
        # Check that name gets handled properly
        repo_url = "local_host/foo/bar/"
        image2_name = "docker_client_create_image_url_test"
        docker_client.images.buildx(
            dummy_tesseract_location, repo_url + image2_name, dummy_docker_file
        )
        image2_py = docker_py_client.images.get(repo_url + image2_name)
        assert image2_py is not None

        # Our docker client should be able to retrieve images with just the name
        image2 = docker_client.images.get(repo_url + image2_name)
        image2_no_url = docker_client.images.get(image2_name)
        assert image2 is not None
        assert image2_no_url is not None
        assert image2.id == image2_py.id
        assert image2_no_url.id == image2_py.id

        assert image_exists(docker_client, image2_name)
        assert image_exists(docker_client, repo_url + image2_name)
        assert image_exists(docker_py_client, repo_url + image2_name)

    finally:
        # Clean up the images
        if image1:
            docker_client.images.remove(image1_name)
            assert not image_exists(docker_client, image1_name)
            assert not image_exists(docker_py_client, image1_name)

        if image2:
            docker_client.images.remove(repo_url + image2_name)
            assert not image_exists(docker_client, image2_name)
            assert not image_exists(docker_py_client, repo_url + image2_name)


def test_create_container(
    docker_client, docker_py_client, docker_client_built_image_name
):
    """Test container creation, run, logs, and remove."""
    # Create a container
    container, container_py = None, None
    try:
        container = docker_client.containers.run(
            docker_client_built_image_name, ['echo "Hello, Tesseract!"'], detach=True
        )
        assert container is not None
        container_py = docker_py_client.containers.run(
            docker_client_built_image_name, ['echo "Hello, Tesseract!"'], detach=True
        )
        assert container_py is not None
        # Check property fields
        assert container.project_id is None
        assert container.host_port is None
        assert container_py.image.id == container.image.id

        container_get = docker_client.containers.get(container.id)
        container_name_get = docker_client.containers.get(container.name)
        container_py_get = docker_py_client.containers.get(container.id)
        container_py_name_get = docker_py_client.containers.get(container.name)
        assert container_get
        assert container_name_get
        assert container_py_get
        assert container_py_name_get

        assert container_name_get == container_get
        # Only compare the key fields of the original container and the ones from get
        # in case container attrs have been updated since creation
        assert container.id == container_get.id
        assert container.name == container_get.name
        # Compare the docker-py container fields vs docker client fields
        assert container_get.id == container_py_get.id
        assert container_get.short_id == container_py_get.short_id
        assert container_get.name == container_py_get.name
        assert container_get.id == container_py_name_get.id

        status = container.wait()
        status_py = container_py.wait()
        assert status["StatusCode"] == 0
        assert status_py["StatusCode"] == 0

        stdout = container.logs(stdout=True, stderr=False)
        stderr = container.logs(stdout=False, stderr=True)
        assert stdout == b"Hello, Tesseract!\n"
        assert stderr == b""

        stdout_py = container_py.logs(stdout=True, stderr=False)
        stderr_py = container_py.logs(stdout=False, stderr=True)
        assert stdout_py == stdout
        assert stderr_py == stderr

    finally:
        # Clean up the container
        if container:
            container.remove(v=True, force=True)
            # Check that the container is removed
            containers = docker_client.containers.list()
            containers_py = docker_py_client.containers.list()
            assert container.id not in containers
            assert container.id not in containers_py

        if container_py:
            container_py.remove(v=True, force=True)
            # Check that the container is removed
            containers = docker_client.containers.list()
            containers_py = docker_py_client.containers.list()
            assert container_py.id not in containers
            assert container_py.id not in containers_py


def test_container_volume_mounts(
    docker_client, docker_client_built_image_name, tmp_path
):
    """Test container volume mounts."""
    container1 = None
    try:
        # Pytest creates the tmp_path fixture with drwx------ mode, we need others
        # to be able to read and execute the path so the Docker volume is readable
        # from within the container
        tmp_path.chmod(0o0707)

        dest = Path("/foo/")
        bar_file = dest / "hello.txt"
        stdout = docker_client.containers.run(
            docker_client_built_image_name,
            [f"touch {bar_file} && chmod 777 {bar_file} && echo hello"],
            detach=False,
            volumes={tmp_path: {"bind": dest, "mode": "rw"}},
            remove=True,
        )

        assert stdout == "hello\n"
        # Check file exists in tmp path
        assert (tmp_path / "hello.txt").exists()

        # Check container is removed and there are no running containers associated with the test image
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                f"ancestor={docker_client_built_image_name}",
                "-q",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout == ""

        # Open tmp_path/hello.txt in write mode
        with open(tmp_path / "hello.txt", "w") as f:
            f.write("hello tesseract\n")

        # Check we can read it in another container
        container1 = docker_client.containers.run(
            docker_client_built_image_name,
            [f"cat {dest}/hello.txt"],
            detach=True,
            volumes={tmp_path: {"bind": dest, "mode": "rw"}},
        )
        status = container1.wait()
        assert status["StatusCode"] == 0
        stdout = container1.logs(stdout=True, stderr=False)
        assert stdout == b"hello tesseract\n"

    finally:
        # Clean up the container
        if container1:
            container1.remove(v=True, force=True)


def test_compose_up_down(
    docker_client, docker_py_client, tmp_path, docker_client_built_image_name
):
    """Test docker-compose up and down."""
    project_name, project_name1 = None, None
    try:
        compose_file = tmp_path / "docker-compose.yml"
        # Use tail -f /dev/null to keep the container running
        compose_content = textwrap.dedent(f"""
            services:
              test:
                image: {docker_client_built_image_name}
                command: ["echo 'Hello Tesseract' && tail -f /dev/null"]
        """)
        compose_file.write_text(compose_content)
        # Run docker-compose up
        project_name = docker_client.compose.up(
            str(compose_file), "docker_client_compose_test"
        )
        assert project_name == "docker_client_compose_test"

        # Check that project is visible in list
        projects = docker_client.compose.list()
        assert project_name in projects
        assert docker_client.compose.exists(project_name)

        # Get container associated with this project
        containers = projects.get(project_name)
        container = docker_client.containers.get(containers[0])
        assert container is not None
        stdout = container.logs(stdout=True, stderr=False)
        assert stdout == b"Hello Tesseract\n"

        # Get container from docker-py
        container_py = docker_py_client.containers.get(containers[0])
        assert container_py is not None
        stdout_py = container_py.logs(stdout=True, stderr=False)
        assert stdout_py == stdout

        # Create a second project
        project_name1 = docker_client.compose.up(
            str(compose_file), "docker_client_compose_test_1"
        )
        # Check both projects exist
        assert docker_client.compose.exists(project_name)
        assert docker_client.compose.exists(project_name1)

    finally:
        # Remove one project
        if project_name:
            docker_client.compose.down(project_name)
            assert not docker_client.compose.exists(project_name)

        # Remove second project
        if project_name1:
            assert docker_client.compose.exists(project_name1)
            docker_client.compose.down(project_name1)
            assert not docker_client.compose.exists(project_name1)


def test_compose_error(docker_client, tmp_path, docker_client_built_image_name):
    """Test docker-compose error handling."""
    compose_file = tmp_path / "docker-compose.yml"
    # Write a malformed compose file
    compose_content = textwrap.dedent(f"""
        services:
            test:
            image: {docker_client_built_image_name}
    """)
    compose_file.write_text(compose_content)
    with pytest.raises(ContainerError) as e:
        docker_client.compose.up(str(compose_file), "docker_client_compose_test")
    # Check that the container's logs were printed to stderr
    assert "Failed to start Tesseract container" in str(e.value)
