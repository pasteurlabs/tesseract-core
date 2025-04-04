"""End to end tests for docker cli wrapper."""

from common import image_exists

from tesseract_core.sdk.docker_cli_wrapper import CLIDockerClient


def test_create_image(docker_client, dummy_tesseract_location, dummy_docker_file):
    """Test image building, getting, and removing."""
    image, image1 = None, None
    # Create an image
    try:
        image_name = "tesseract_create_image_test:dummy"

        docker_client.images.buildx(
            dummy_tesseract_location, image_name, dummy_docker_file
        )
        image = docker_client.images.get(image_name)
        assert image is not None
        assert image.name == image_name

        # Create a second image
        image1_name = "tesseract_core_test:dummy1"
        docker_client.images.buildx(
            dummy_tesseract_location, image1_name, dummy_docker_file
        )
        image1 = docker_client.images.get(image1_name)
        assert image1 is not None
        assert image1.name == image1_name

        # Check that image and image1 both exist
        assert image_exists(docker_client, image.name)
        assert image_exists(docker_client, image1.name)

    finally:
        # Clean up the images
        try:
            if image:
                docker_client.images.remove(image.name)
            if image1:
                docker_client.images.remove(image1.name)

        except CLIDockerClient.Errors.ImageNotFound:
            pass

        # Check that images are removed
        assert not image_exists(docker_client, image.name)
        assert not image_exists(docker_client, image1.name)


def test_create_container(docker_client, dummy_tesseract_location, dummy_docker_file):
    """Test container creation, run, logs, and remove."""
    # Create a container
    image, container = None, None
    try:
        image_name = "tesseract_create_container_test:dummy"

        docker_client.images.buildx(
            dummy_tesseract_location, image_name, dummy_docker_file
        )
        image = docker_client.images.get(image_name)
        assert image is not None
        assert image.name == image_name

        container = docker_client.containers.run(image_name, [], detach=True)
        assert container is not None
        container_id = container.id

        container_get = docker_client.containers.get(container_id)
        assert container_get is not None
        assert container_get.id == container.id

        logs = container.logs()
        assert logs == ("Hello, Tesseract!\n", "")

    finally:
        try:
            # Clean up the container
            if container:
                container.remove(v=True, force=True)

            # Check that the container is removed
            containers = docker_client.containers.list()
            assert container.id not in containers.keys()
        except CLIDockerClient.Errors.ContainerError:
            pass

        # Image has to be removed after container is removed
        try:
            if image:
                docker_client.images.remove(image.id)
        except CLIDockerClient.Errors.ImageNotFound:
            pass
