"""Docker for Tesseract usage."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from pathlib import Path

logger = logging.getLogger("tesseract")


class CLIDockerClient:
    """Wrapper around Docker CLI to manage Docker containers and images.

    Initializes a new instance of the current Docker state from the
    perspective of Tesseracts. Loads in a previous state if one exists.
    """

    def __init__(self) -> None:
        self.project_container_map = {}  # Mapping from project ID to list of container ids
        self.containers = self.Containers()
        self.images = self.Images()
        self.compose = self.Compose(self)

    class Images:
        """Class to interface with docker images."""

        def __init__(self) -> None:
            self.images = []  # List of Image objects

        class Image:
            """Image class to wrap Docker image details."""

            def __init__(self, json_dict: dict) -> None:
                self.id = json_dict.get("Id", None)
                self.short_id = self.id[:12] if self.id else None
                self.attrs = json_dict
                self.tags = json_dict.get("RepoTags", None)
                # Docker images may be prefixed with the registry URL
                self.name = self.tags[0].split("/")[-1] if self.tags else None

            def __str__(self) -> str:
                return f"Image id: {self.id}, name: {self.name}, tags: {self.tags}, attrs: {self.attrs}"

        def get(self, image: str) -> Image:
            """Returns the metadata for a specific image."""
            # Iterate through all the images and see if any of them
            # have id or name matching the image str

            # Use getter func to make sure it's updated
            if ":" not in image:
                image = image + ":latest"
            images = self.list()
            for image_obj in images:
                if image_obj.id == image or image_obj.name == image:
                    return image_obj
            raise CLIDockerClient.Errors.ImageNotFound(f"Image {image} not found.")

        def list(self) -> list[Image]:
            """Returns the current list of images."""
            self._update_images()
            return self.images

        def remove_image(self, image_id: str) -> None:
            """Remove an image from the local Docker registry."""
            try:
                _ = subprocess.run(
                    ["docker", "rmi", image_id, "--force"],
                    check=True,
                    capture_output=True,
                )

                self._update_images()

            except subprocess.CalledProcessError as ex:
                raise CLIDockerClient.Errors.ImageNotFound(
                    f"Cannot remove image {image_id}: {ex}"
                ) from ex

        def buildx(
            self,
            path: str | Path,
            tag: str,
            dockerfile: str | Path,
            inject_ssh: bool = False,
            keep_build_cache: bool = False,
            print_and_exit: bool = False,
        ) -> Image | None:
            """Build a Docker image from a Dockerfile using BuildKit."""
            from tesseract_core.sdk.engine import LogPipe

            build_cmd = [
                "docker",
                "buildx",
                "build",
                "--load",
                *(["--no-cache"] if not keep_build_cache else []),
                "--tag",
                tag,
                "--file",
                str(dockerfile),
                str(path),
            ]

            if inject_ssh:
                ssh_sock = os.environ.get("SSH_AUTH_SOCK")
                if ssh_sock is None:
                    raise ValueError(
                        "SSH_AUTH_SOCK environment variable not set (try running `ssh-agent`)"
                    )

                ssh_keys = subprocess.run(["ssh-add", "-L"], capture_output=True)
                if ssh_keys.returncode != 0 or not ssh_keys.stdout:
                    raise ValueError(
                        "No SSH keys found in SSH agent (try running `ssh-add`)"
                    )

                build_cmd += ["--ssh", f"default={ssh_sock}"]

            if print_and_exit:
                logger.info(
                    f"To build the Docker image manually, run:\n    $ {shlex.join(build_cmd)}"
                )
                return None

            out_pipe = LogPipe(logging.DEBUG)
            with out_pipe as out_pipe_fd:
                proc = subprocess.run(build_cmd, stdout=out_pipe_fd, stderr=out_pipe_fd)

            logs = out_pipe.captured_lines
            return_code = proc.returncode

            if return_code != 0:
                raise CLIDockerClient.Errors.BuildError(logs)

            # Update self.images
            self._update_images()
            # Get image object
            image = self.get(tag)
            return image

        def _update_images(self) -> None:
            """Updates the list of images by querying Docker CLI."""
            try:
                image_ids = subprocess.run(
                    ["docker", "images", "-q"],  # List only image IDs
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if not image_ids.stdout:
                    images = []
                else:
                    images = image_ids.stdout.strip().split("\n")

                # Clean up deleted images.
                for image in self.images:
                    if image.id not in images:
                        self.images.remove(image)

                # Filter list to exclude image ids that are already in self.images.
                images = [
                    image_id
                    for image_id in images
                    if image_id not in self.images and image_id
                ]
                json_dicts = get_docker_metadata(images, is_image=True)
                for _, json_dict in json_dicts.items():
                    image = self.Image(json_dict)
                    self.images.append(image)

            except subprocess.CalledProcessError as ex:
                raise CLIDockerClient.Errors.APIError(
                    f"Cannot list Docker images: {ex}"
                ) from ex

    class Containers:
        """Class to interface with docker containers."""

        def __init__(self) -> None:
            self.containers = {}

        class Container:
            """Container class to wrap Docker container details."""

            def __init__(self, json_dict: dict) -> None:
                self.id = json_dict.get("Id", None)
                self.short_id = self.id[:12] if self.id else None
                self.name = json_dict.get("Name", None)
                ports = json_dict.get("NetworkSettings", None)
                if ports and ports.get("Ports", None):
                    ports = ports["Ports"]
                    port_key = next(iter(ports))  # Get the first port key
                    if ports[port_key]:
                        self.host_port = ports[port_key][0].get(
                            "HostPort"
                        )  # Get the host port
                else:
                    self.host_port = None
                self.attrs = json_dict
                self.project_id = json_dict.get("Config", None)
                if self.project_id:
                    self.project_id = self.project_id["Labels"].get(
                        "com.docker.compose.project", None
                    )

            def exec_run(self, command: str) -> tuple:
                """Run a command in this container.

                Return exit code and stdout.
                """
                try:
                    result = subprocess.run(
                        ["docker", "exec", self.id, *command],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    return result.returncode, result.stdout
                except subprocess.CalledProcessError as ex:
                    raise CLIDockerClient.Errors.ContainerError(
                        f"Cannot run command in container {self.id}: {ex}"
                    ) from ex

            def __str__(self) -> str:
                string = (
                    f"Container id: {self.id}, name: {self.name}, project_id: {self.project_id},"
                    "ports: {self.host_port}, attrs: {self.attrs}"
                )
                return string

        def list(self) -> dict:
            """Returns the current list of containers."""
            self._update_containers()
            return self.containers

        def get(self, container: str) -> Container:
            """Returns the metadata for a specific container."""
            # Use the get_all_containers() function to make sure it's updated
            # First check if we can find the container by id
            container_obj = self.list().get(container, None)
            if not container_obj:
                # Check for container names if id does not work
                for _, container_obj in self.list().items():
                    if container_obj.name == container:
                        return container_obj
            return container_obj

        def run(
            self,
            image_id: str,
            command: list[str],
            parsed_volumes: dict,
            gpus: list[int | str] | None = None,
        ) -> bool:
            """Run a command in a container from an image."""
            from tesseract_core.sdk.engine import LogPipe

            # Convert the parsed_volumes into a list of strings in proper argument format,
            # `-v host_path:container_path:mode`.
            if not parsed_volumes:
                volume_args = []
            else:
                volume_args = []
                for host_path, volume_info in parsed_volumes.items():
                    volume_args.append("-v")
                    volume_args.append(
                        f"{host_path}:{volume_info['bind']}:{volume_info['mode']}"
                    )

            if gpus:
                gpus_str = ",".join(gpus)
                gpus_option = f'--gpus "device={gpus_str}"'
            else:
                gpus_option = ""

            # Remove the container after usage so we do not need to update self.containers.
            cmd_list = [
                "docker",
                "run",
                "--rm",
                *volume_args,
                *([gpus_option] if gpus_option else []),
                image_id,
                *command,
            ]

            try:
                out_pipe = LogPipe(logging.DEBUG)
                with out_pipe as out_pipe_fd:
                    proc = subprocess.run(
                        cmd_list,
                        stdout=out_pipe_fd,
                        stderr=out_pipe_fd,
                    )

                logs = out_pipe.captured_lines
                return_code = proc.returncode

                if return_code != 0:
                    raise CLIDockerClient.Errors.APIError(logs)

                return logs

            except subprocess.CalledProcessError as ex:
                raise CLIDockerClient.Errors.APIError(
                    f"Error running command: `{' '.join(cmd_list)}`. \n\n{ex.stderr}"
                ) from ex

        def _update_containers(self) -> None:
            """Update self.containers."""
            try:
                result = subprocess.run(
                    ["docker", "ps", "-q"],  # List only container IDs
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if not result.stdout:
                    container_ids = []
                else:
                    container_ids = result.stdout.strip().split("\n")

                # Check if theres any cleaned up containers.
                for container_id in list(self.containers.keys()):
                    if container_id not in container_ids:
                        del self.containers[container_id]
                # Filter list to exclude container ids that are already in self.containers
                # also exclude empty strings.
                container_ids = [
                    container_id
                    for container_id in container_ids
                    if container_id not in self.containers and container_id
                ]
                json_dicts = get_docker_metadata(container_ids)
                for container_id, json_dict in json_dicts.items():
                    container = self.Container(json_dict)
                    self.containers[container_id] = container

            except subprocess.CalledProcessError as ex:
                raise CLIDockerClient.Errors.APIError(
                    f"Cannot list Docker containers: {ex}"
                ) from ex

    class Compose:
        """Class to interface with docker projects."""

        def __init__(self, docker_cli):
            self.project_container_map = {}  # Mapping from project ID to list of container ids
            self.containers = docker_cli.Containers()

        def list(self) -> dict:
            """Returns the current list of projects."""
            # Check if containers is updated
            self.containers.list()
            self._update_projects()
            return self.project_container_map

        def up(self, compose_fpath: str, project_name: str) -> bool:
            """Start containers using Docker Compose template."""
            logger.info("Waiting for Tesseract containers to start ...")
            try:
                _ = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "-f",
                        compose_fpath,
                        "-p",
                        project_name,
                        "up",
                        "-d",
                        "--wait",
                    ],
                    check=True,
                    capture_output=True,
                )
                return project_name
            except subprocess.CalledProcessError as ex:
                logger.error(str(ex))
                logger.error(ex.stderr.decode())
                raise CLIDockerClient.Errors.ContainerError(
                    "Failed to start Tesseract containers."
                ) from ex

        def down(self, project_id: str) -> bool:
            """Stop and remove containers and networks associated to a project."""
            try:
                __ = subprocess.run(
                    ["docker", "compose", "-p", project_id, "down"],
                    check=True,
                    capture_output=True,
                )
                return True
            except subprocess.CalledProcessError as ex:
                logger.error(str(ex))
                return False

        def exists(self, project_id: str) -> bool:
            """Check if Docker Compose project exists."""
            return project_id in self.list()

        def _project_containers(
            self,
            project_id: str,
        ) -> list[str]:
            """Find containers associated with a Docker Compose Project ID.

            Args:
                project_id: the Docker Compose project ID.

            Returns:
                A list of Docker container ids.
            """
            # Calling self.get_projects will update containers and projects map
            if project_id in self.get_projects():
                return self.project_container_map[project_id]
            try:
                # Run the docker ps command to list containers
                result = subprocess.run(
                    [
                        "docker",
                        "ps",
                        "--format",
                        "{{.ID}} {{.Names}}",
                    ],  # Get container ID and name
                    capture_output=True,
                    text=True,
                    check=True,
                )
                # Filter containers by project_id (matching the project_id in container names)
                containers = [
                    line.split()[
                        0
                    ]  # Container id is the second element in the output line
                    for line in result.stdout.splitlines()
                    if project_id in line.split()[1]
                ]
                # Update the project_container_map with the list of containers associated
                # with the project_id
                self.project_container_map[project_id] = containers
                return containers

            except subprocess.CalledProcessError as e:
                # Handle errors (e.g., if docker ps fails)
                print(f"Error running docker ps: {e.stderr}")
                return []

        def _update_projects(self) -> None:
            """Updates the list of projects by going through containers."""
            self.project_container_map = {}
            for container_id, container in self.containers.items():
                if container.project_id:
                    if container.project_id not in self.project_container_map:
                        self.project_container_map[container.project_id] = []
                    self.project_container_map[container.project_id].append(
                        container_id
                    )

    def info(self) -> tuple:
        """Wrapper around docker info call."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                check=True,
                capture_output=True,
            )
            return result.stdout, result.stderr
        except subprocess.CalledProcessError as ex:
            raise CLIDockerClient.Errors.APIError() from ex

    class Errors:
        """Errors that can be raised by the Docker CLI."""

        class DockerException(Exception):
            """Base class for Docker CLI exceptions."""

            pass

        class BuildError(DockerException):
            """Raised when a build fails."""

            def __init__(self, build_log):
                self.build_log = build_log

        class ContainerError(DockerException):
            """Raised when a container has error."""

            pass

        class APIError(DockerException):
            """Raised when a Docker API error occurs."""

            pass

        class ImageNotFound(DockerException):
            """Raised when an image is not found."""

            pass


def get_docker_metadata(docker_asset_ids: list[str], is_image: bool = False) -> dict:
    """Get metadata for Docker images/containers.

    Returns a dict mapping asset ids to their metadata.
    """
    if not docker_asset_ids:
        return {}

    # Set metadata in case no images exist and metadata does not get initialized.
    metadata = None
    try:
        result = subprocess.run(
            ["docker", "inspect", *docker_asset_ids],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata = json.loads(result.stdout)

    except subprocess.CalledProcessError as e:
        # Handle the error if some images do not exist.
        error_message = e.stderr
        for asset_id in docker_asset_ids:
            if f"No such image: {asset_id}" in error_message:
                print(f"Image {asset_id} is not a valid image.")
        if "No such object" in error_message:
            raise CLIDockerClient.Errors.ContainerError(
                "Unhealthy container found. Please restart docker."
            ) from e

    if not metadata:
        return {}

    asset_meta_dict = {}
    # Parse the output into a dictionary of only Tesseract assets
    # with the id as the key for easy parsing, and the metadata as the value.
    for asset in metadata:
        env_vars = asset["Config"]["Env"]
        if not any("TESSERACT_NAME" in env_var for env_var in env_vars):
            # Do not add items if there is no "TESSERACT_NAME" in env vars.
            continue
        if is_image:
            # If it is an image, use the repotag as the key.
            dict_key = asset["RepoTags"]
            if not dict_key:
                # Old dangling images do not have RepoTags.
                continue
            dict_key = dict_key[0]
        else:
            dict_key = asset["Id"][:12]
        asset_meta_dict[dict_key] = asset
    return asset_meta_dict
