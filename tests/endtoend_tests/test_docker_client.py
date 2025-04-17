# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End to end tests for docker cli wrapper."""

import subprocess
import textwrap
from pathlib import Path

import pytest
from common import image_exists

from tesseract_core.sdk.docker_client import ContainerError, ImageNotFound


@pytest.fixture()
def docker_client_built_image_name(
    docker_client, dummy_tesseract_location, dummy_docker_file
):
    """Build the dummy image for the tests."""
    image_name = "docker_client_create_image_test:dummy"

    docker_client.images.buildx(
        dummy_tesseract_location, image_name, dummy_docker_file, keep_build_cache=True
    )
    image = docker_client.images.get(image_name)

    try:
        yield image.name
    finally:
        try:
            docker_client.images.remove(image.name)
        except ImageNotFound:
            # already removed
            pass


def test_get_image(
    docker_client,
    docker_client_built_image_name,
):
    """Test image retrieval."""
    # Get the image
    image = docker_client.images.get(docker_client_built_image_name)
    assert image is not None
    assert docker_client_built_image_name == image.name

    # Check the custom str function
    assert (
        str(image)
        == f"Image id: {image.id}, name: {image.name}, tags: {image.tags}, attrs: {image.attrs}"
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


def test_create_image(
    docker_client,
    dummy_tesseract_location,
    dummy_docker_file,
    docker_client_built_image_name,
):
    """Test image building, getting, and removing."""
    image, image1, image2 = None, None, None
    # Create an image
    try:
        image = docker_client.images.get(docker_client_built_image_name)
        assert image is not None
        assert docker_client_built_image_name == image.name
        # Check the custom str function
        assert (
            str(image)
            == f"Image id: {image.id}, name: {image.name}, tags: {image.tags}, attrs: {image.attrs}"
        )

        image_id_obj = docker_client.images.get(image.id)
        image_short_id_obj = docker_client.images.get(image.short_id)
        assert str(image_id_obj) == str(image)
        assert str(image_short_id_obj) == str(image)

        # Create a second image with no label
        # Check that :latest gets added automatically
        image1_name = "docker_client_create_image_test"
        docker_client.images.buildx(
            dummy_tesseract_location, image1_name, dummy_docker_file
        )
        image1 = docker_client.images.get(image1_name)
        image1_get_name, image1_get_tag = (
            image1.name.split(":")[0],
            image1.name.split(":")[1],
        )
        assert image1 is not None
        assert image1_name == image1_get_name
        assert image1_get_tag == "latest"

        # Create a third image with prefixed with repo url
        # Check that name gets handled properly
        repo_url = "local_host/foo/bar/"
        image2_name = "docker_client_create_image_url_test"
        docker_client.images.buildx(
            dummy_tesseract_location, repo_url + image2_name, dummy_docker_file
        )
        image2 = docker_client.images.get(image2_name)
        image2_get_name, image2_get_tag = (
            image2.name.split(":")[0],
            image2.name.split(":")[1],
        )
        assert image2 is not None
        assert image2_name == image2_get_name
        assert image2_get_tag == "latest"

        # Check that image and image1 both exist
        assert image_exists(docker_client, image.name)
        assert image_exists(docker_client, image1.name)
        assert image_exists(docker_client, image2.name)

    finally:
        # Clean up the images (using different methods)
        if image1:
            docker_client.images.remove(image1.id)
            assert not image_exists(docker_client, image1_name)

        if image2:
            docker_client.images.remove(image2.id)
            assert not image_exists(docker_client, image2_name)


def test_create_container(docker_client, docker_client_built_image_name):
    """Test container creation, run, logs, and remove."""
    # Create a container
    container = None
    try:
        container = docker_client.containers.run(
            docker_client_built_image_name, ['echo "Hello, Tesseract!"'], detach=True
        )
        assert container is not None
        # Test custom str container function

        assert (
            f"Container id: {container.id}, name: {container.name}, project_id: {container.project_id}"
            in str(container)
        )

        container_get = docker_client.containers.get(container.id)
        container_name_get = docker_client.containers.get(container.name)
        assert container_get is not None
        assert str(container_get) == str(container)
        assert str(container_name_get) == str(container_get)

        status = container.wait()
        assert status["StatusCode"] == 0

        stdout = container.logs(stdout=True, stderr=False)
        stderr = container.logs(stdout=False, stderr=True)
        assert stdout == b"Hello, Tesseract!\n"
        assert stderr == b""

    finally:
        try:
            # Clean up the container
            if container:
                container.remove(v=True, force=True)

                # Check that the container is removed
                containers = docker_client.containers.list()
                assert container.id not in containers.keys()
        except ContainerError:
            pass

        # Image has to be removed after container is removed
        try:
            if docker_client_built_image_name:
                docker_client.images.remove(docker_client_built_image_name)
        except ImageNotFound:
            pass

        try:
            docker_client.containers.get(container.id)
        except ContainerError:
            pass


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
        stdout, _ = docker_client.containers.run(
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
        try:
            if docker_client_built_image_name:
                docker_client.images.remove(docker_client_built_image_name)
        except ImageNotFound:
            pass

        try:
            # Clean up the container
            if container1:
                container1.remove(v=True, force=True)
        except ContainerError:
            pass


def test_compose_up_down(docker_client, tmp_path, docker_client_built_image_name):
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
    # Use tail -f /dev/null to keep the container running
    # Override the image's command otherwise it immediately exits
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
