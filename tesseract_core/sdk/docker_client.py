# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker client for Tesseract usage."""

import json
import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

# store a reference to the list type, which is shadowed by some function names below
from typing import List as list_  # noqa: UP035

from tesseract_core.sdk.config import get_config

logger = logging.getLogger("tesseract")


def _get_docker_executable() -> list_[str]:  # noqa: UP006
    """Get the Docker executable command."""
    config = get_config()
    docker_executable = config.docker_executable
    if isinstance(docker_executable, str):
        return shlex.split(docker_executable)
    return list(docker_executable)


def _is_valid_docker_tag(tag: str) -> bool:
    if not (1 <= len(tag) <= 128):
        return False
    if not re.match(r"^[A-Za-z0-9_.-]+$", tag):
        return False
    return True


def is_podman() -> bool:
    """Check if the current environment is using Podman instead of Docker."""
    docker = _get_docker_executable()
    try:
        result = subprocess.run(
            [*docker, "version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return "podman" in result.stdout.lower()
    except subprocess.CalledProcessError:
        return False


@dataclass
class Image:
    """Image class to wrap Docker image details."""

    id: str | None
    short_id: str | None
    tags: list[str] | None
    attrs: dict

    @classmethod
    def from_dict(cls, json_dict: dict) -> "Image":
        """Create an Image object from a json dictionary."""
        image_id = json_dict.get("Id", None)
        short_id = None
        if image_id:
            if image_id.startswith("sha256:"):
                short_id = image_id[:19]
            else:
                # Some container engines (e.g., Podman) do not prefix IDs with sha256
                short_id = image_id[:12]

        return cls(
            id=image_id,
            short_id=short_id,
            tags=json_dict.get("RepoTags", None),
            attrs=json_dict,
        )


class Images:
    """Namespace for functions to interface with Tesseract docker images."""

    @staticmethod
    def get(image_id_or_name: str | bytes, tesseract_only: bool = True) -> Image:
        """Returns the metadata for a specific image.

        In docker-py, there is no substring matching and the image name is the
        last tag in the list of tags, so if an image has multiple tags, only
        one of the tags would be able to find the image.

        However, in podman, this is not the case. Podman has substring matching
        by "/" segments to handle repository urls and returns images even if
        partial name is specified, or if image has multiple tags.

        We chose to support podman's largest string matching functionality here.

        Params:
            image_id_or_name: The image name or id to get.
            tesseract_only: If True, only retrieves Tesseract images.

        Returns:
            Image object.
        """
        if not image_id_or_name:
            raise ValueError("Image name cannot be empty.")

        docker = _get_docker_executable()
        try:
            result = subprocess.run(
                [*docker, "inspect", image_id_or_name, "--type", "image"],
                check=True,
                capture_output=True,
                text=True,
            )
            json_dict = json.loads(result.stdout)
        except subprocess.CalledProcessError as ex:
            raise ImageNotFound(f"Image {image_id_or_name} not found.") from ex
        if not json_dict:
            raise ImageNotFound(f"Image {image_id_or_name} not found.")

        if tesseract_only and not any(
            "TESSERACT_NAME" in env_var for env_var in json_dict[0]["Config"]["Env"]
        ):
            raise ImageNotFound(f"Image {image_id_or_name} is not a Tesseract image.")
        image_obj = Image.from_dict(json_dict[0])
        return image_obj

    @staticmethod
    def list(tesseract_only: bool = True) -> list_[Image]:  # noqa: UP006
        """Returns the current list of images.

        Params:
            tesseract_only: If True, only return Tesseract images.

        Returns:
            List of Image objects.
        """
        return Images._get_images(tesseract_only=tesseract_only)

    @staticmethod
    def remove(image: str) -> None:
        """Remove an image (name or id) from the local Docker registry.

        Params:
            image: The image name or id to remove.
        """
        docker = _get_docker_executable()
        try:
            res = subprocess.run(
                [*docker, "rmi", image, "--force"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as ex:
            raise ImageNotFound(f"Cannot remove image {image}: {ex}") from ex

        if "No such image" in res.stderr:
            raise ImageNotFound(f"Cannot remove image {image}: {res.stderr}")

    @staticmethod
    def _get_buildx_command(
        path: str | Path,
        tags: list_[str],  # noqa: UP006
        dockerfile: str | Path,
        ssh: str | None = None,
    ) -> list_[str]:  # noqa: UP006
        """Get the buildx command for building Docker images.

        Returns:
            The buildx command as a list of strings.
        """
        config = get_config()
        docker = _get_docker_executable()
        extra_args = config.docker_build_args

        if ssh is not None:
            extra_args = ("--ssh", ssh, *extra_args)

        tag_args = [x for t in tags for x in ["--tag", t]]

        build_cmd = [
            *docker,
            "buildx",
            "build",
            "--load",
            *tag_args,
            "--file",
            str(dockerfile),
            *extra_args,
            "--",
            str(path),
        ]

        return build_cmd

    @staticmethod
    def buildx(
        path: str | Path,
        tags: list_[str],  # noqa: UP006
        dockerfile: str | Path,
        ssh: str | None = None,
    ) -> Image:
        """Build a Docker image from a Dockerfile using BuildKit.

        Params:
            path: Path to the directory containing the Dockerfile.
            tag: The name of the image to build.
            dockerfile: path within the build context to the Dockerfile.
            ssh: If not None, pass given argument to buildx --ssh command.

        Returns:
            Built Image object.
        """
        from tesseract_core.sdk.engine import LogPipe

        for tag in tags:
            proper_tag = tag.split(":")[1]
            if not _is_valid_docker_tag(proper_tag):
                raise ValueError(
                    f"Invalid tag {proper_tag}; only alphanumeric characters, "
                    "'.', and '-' are allowed."
                )

        build_cmd = Images._get_buildx_command(
            path=path,
            tags=tags,
            dockerfile=dockerfile,
            ssh=ssh,
        )

        out_pipe = LogPipe(logging.DEBUG)

        with out_pipe as out_pipe_fd:
            proc = subprocess.run(build_cmd, stdout=out_pipe_fd, stderr=out_pipe_fd)

        logs = out_pipe.captured_lines
        return_code = proc.returncode

        if return_code != 0:
            raise BuildError(logs)

        return Images.get(tags[0])

    @staticmethod
    def _get_images(tesseract_only: bool = True) -> list_[Image]:  # noqa: UP006
        """Gets the list of images by querying Docker CLI.

        Params:
            tesseract_only: If True, only return Tesseract images.

        Returns:
            List of (non-dangling) Image objects.
        """
        docker = _get_docker_executable()
        images = []
        try:
            image_ids = subprocess.run(
                [*docker, "images", "-q"],  # List only image IDs
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as ex:
            raise APIError(f"Cannot list Docker images: {ex}") from ex

        if not image_ids.stdout:
            return []

        image_ids = image_ids.stdout.strip().split("\n")
        # Filter list to exclude empty strings.
        image_ids = [image_id for image_id in image_ids if image_id]

        # If image shows up multiple times, that means it is tagged multiple times
        # So we need to make multiple copies of the image with different names
        image_counts = {}
        for image_id in image_ids:
            image_counts[image_id] = image_counts.get(image_id, 0) + 1

        json_dicts = get_docker_metadata(
            image_ids, is_image=True, tesseract_only=tesseract_only
        )
        for _, json_dict in json_dicts.items():
            image = Image.from_dict(json_dict)
            images.append(image)

        return images


@dataclass
class Container:
    """Container class to wrap Docker container details.

    Container class has additional member variables `host_port` and `host_ip` that
    docker-py does not have. This is because Tesseract requires frequent access to the host
    port mappings.
    """

    id: str
    short_id: str
    name: str
    attrs: dict

    @classmethod
    def from_dict(cls, json_dict: dict) -> "Container":
        """Create a Container object from a json dictionary."""
        return cls(
            id=json_dict["Id"],
            short_id=json_dict["Id"][:12],
            name=json_dict["Name"].lstrip("/"),
            attrs=json_dict,
        )

    @property
    def image(self) -> Image | None:
        """Gets the image ID of the container."""
        image_id = self.attrs.get("ImageID", self.attrs["Image"])
        if image_id is None:
            return None
        return Images.get(image_id)

    @property
    def host_port(self) -> str | None:
        """Gets the host port of the container."""
        if self.attrs.get("NetworkSettings", None):
            ports = self.attrs["NetworkSettings"].get("Ports", None)
            if ports:
                api_port_key = "8000/tcp"
                if ports[api_port_key]:
                    return ports[api_port_key][0].get("HostPort")
        return None

    @property
    def host_debugpy_port(self) -> str | None:
        """Gets the host port which maps to debugpy server in the container."""
        if self.attrs.get("NetworkSettings", None):
            ports = self.attrs["NetworkSettings"].get("Ports", None)
            if ports:
                debugpy_port_key = "5678/tcp"
                if debugpy_port_key in ports:
                    return ports[debugpy_port_key][0].get("HostPort")
        return None

    @property
    def host_ip(self) -> str | None:
        """Gets the host IP of the container."""
        if self.attrs.get("NetworkSettings", None):
            ports = self.attrs["NetworkSettings"].get("Ports", None)
            if ports:
                api_port_key = "8000/tcp"
                if ports[api_port_key]:
                    return ports[api_port_key][0].get("HostIp")
        return None

    @property
    def status(self) -> str:
        """Gets the status of the container."""
        return self.attrs.get("State", {}).get("Status", "unknown")

    def exec_run(self, command: list) -> tuple[int, bytes]:
        """Run a command in this container.

        Return exit code and stdout.
        """
        docker = _get_docker_executable()
        result = subprocess.run(
            [*docker, "exec", self.id, *command],
            check=False,
            capture_output=True,
            text=False,
        )
        if result.returncode != 0:
            raise ContainerError(
                self.id,
                result.returncode,
                shlex.join(command),
                self.image.id if self.image else "unknown",
                result.stderr,
            )
        return result.returncode, result.stdout

    def stop(self) -> None:
        """Stop the container."""
        docker = _get_docker_executable()
        try:
            subprocess.run(
                [*docker, "stop", self.id],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as ex:
            raise APIError(f"Cannot stop container {self.id}: {ex}") from ex

    def logs(self, stdout: bool = True, stderr: bool = True) -> bytes:
        """Get the logs for this container.

        Logs needs to be called if container is running in a detached state,
        and we wish to retrieve the logs from the command executing in the container.

        Params:
            stdout: If True, return stdout.
            stderr: If True, return stderr.
        """
        docker = _get_docker_executable()

        if stdout and stderr:
            # use subprocess.STDOUT to combine stdout and stderr into one stream
            # with the correct order of output
            stdout_pipe = subprocess.PIPE
            stderr_pipe = subprocess.STDOUT
            output_attr = "stdout"
        elif not stdout and stderr:
            stdout_pipe = subprocess.DEVNULL
            stderr_pipe = subprocess.PIPE
            output_attr = "stderr"
        elif stdout and not stderr:
            stdout_pipe = subprocess.PIPE
            stderr_pipe = subprocess.DEVNULL
            output_attr = "stdout"
        else:
            raise ValueError("At least one of stdout or stderr must be True.")

        try:
            result = subprocess.run(
                [*docker, "logs", self.id],
                check=True,
                stdout=stdout_pipe,
                stderr=stderr_pipe,
            )
        except subprocess.CalledProcessError as ex:
            raise APIError(f"Cannot get logs for container {self.id}: {ex}") from ex

        return getattr(result, output_attr)

    def wait(self) -> dict:
        """Wait for container to finish running.

        Returns:
            A dict with the exit code of the container.
        """
        docker = _get_docker_executable()

        try:
            result = subprocess.run(
                [*docker, "wait", self.id],
                check=True,
                capture_output=True,
                text=True,
            )
            # Container's exit code is printed by the wait command
            return {"StatusCode": int(result.stdout)}
        except subprocess.CalledProcessError as ex:
            raise APIError(f"Cannot wait for container {self.id}: {ex}") from ex

    def remove(self, v: bool = False, link: bool = False, force: bool = False) -> str:
        """Remove the container.

        Params:
            v: If True, remove volumes associated with the container.
            link: If True, remove links to the container.
            force: If True, force remove the container.

        Returns:
            The output of the remove command.
        """
        docker = _get_docker_executable()
        try:
            result = subprocess.run(
                [
                    *docker,
                    "rm",
                    *(["-f"] if force else []),
                    *(["-v"] if v else []),
                    *(["-l"] if link else []),
                    self.id,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as ex:
            if "docker" in ex.stderr:
                raise APIError(f"Cannot remove container {self.id}: {ex}") from ex
            raise ex


class Containers:
    """Namespace to interface with docker containers."""

    @staticmethod
    def list(all: bool = False, tesseract_only: bool = True) -> list_[Container]:  # noqa: UP006
        """Returns the current list of containers.

        Params:
            all: If True, include stopped containers.
            tesseract_only: If True, only return Tesseract containers.

        Returns:
            List of Container objects.
        """
        return Containers._get_containers(
            include_stopped=all, tesseract_only=tesseract_only
        )

    @staticmethod
    def get(id_or_name: str, tesseract_only: bool = True) -> Container:
        """Returns the metadata for a specific container.

        Params:
            id_or_name: The container name or id to get.
            tesseract_only: If True, only retrieves Tesseract containers.

        Returns:
            Container object.
        """
        docker = _get_docker_executable()

        try:
            result = subprocess.run(
                [*docker, "inspect", id_or_name, "--type", "container"],
                check=True,
                capture_output=True,
                text=True,
            )
            json_dict = json.loads(result.stdout)
        except subprocess.CalledProcessError as ex:
            raise NotFound(f"Container {id_or_name} not found.") from ex

        if not json_dict:
            raise NotFound(f"Container {id_or_name} not found.")
        if tesseract_only and not any(
            "TESSERACT_NAME" in env_var for env_var in json_dict[0]["Config"]["Env"]
        ):
            raise NotFound(f"Container {id_or_name} is not a Tesseract container.")

        container_obj = Container.from_dict(json_dict[0])
        return container_obj

    @staticmethod
    def run(
        image: str,
        command: list_[str],  # noqa: UP006
        volumes: dict | None = None,
        device_requests: list_[int | str] | None = None,  # noqa: UP006
        environment: dict[str, str] | None = None,
        network: str | None = None,
        detach: bool = False,
        remove: bool = False,
        ports: dict | None = None,
        stdout: bool = True,
        stderr: bool = False,
        user: str | None = None,
        extra_args: list_[str] | None = None,  # noqa: UP006
    ) -> Container | tuple[bytes, bytes] | bytes:
        """Run a command in a container from an image.

        Params:
            image: The image name or id to run the command in.
            command: The command to run in the container.
            volumes: A dict of volumes to mount in the container.
            user: String of user information to run command as in the format "uid:(optional)gid".
            device_requests: A list of device requests for the container.
            detach: If True, run the container in detached mode. Detach must be set to
                    True if we wish to retrieve the container id of the running container,
                    and if detach is true, we must wait on the container to finish
                    running and retrieve the logs of the container manually.
            remove: If remove is set to True, the container will automatically remove itself
                    after it finishes executing the command. This means that we cannot set
                    both detach and remove simulataneously to True or else there
                    would be no way of retrieving the logs from the removed container.
            ports: A dict of ports to expose in the container. The keys are the host ports
                   and the values are the container ports.
            stdout: If True, return stdout.
            stderr: If True, return stderr.
            environment: Environment variables to set in the container.
            extra_args: Additional arguments to pass to the `docker run` CLI command.

        Returns:
            Container object if detach is True, otherwise returns list of stdout and stderr.
        """
        config = get_config()
        docker = _get_docker_executable()

        if isinstance(command, str):
            command = [command]

        optional_args = []

        # Convert the parsed_volumes into a list of strings in proper argument format,
        # `-v host_path:container_path:mode`.
        if volumes:
            volume_args = []
            for host_path, volume_info in volumes.items():
                volume_args.append("-v")
                volume_args.append(
                    f"{host_path}:{volume_info['bind']}:{volume_info['mode']}"
                )
            optional_args.extend(volume_args)

        if network:
            optional_args.extend(["--network", network])

        if user:
            optional_args.extend(["-u", user])

        if device_requests:
            gpus_str = ",".join(device_requests)
            optional_args.extend(["--gpus", f'"device={gpus_str}"'])

        if environment:
            env_args = []
            for env_var, value in environment.items():
                env_args.extend(["-e", f"{env_var}={value}"])
            optional_args.extend(env_args)

        # Remove and detached cannot both be set to true
        if remove and detach:
            raise ValueError(
                "Cannot set both remove and detach to True when running a container."
            )
        if detach:
            optional_args.append("--detach")
        if remove:
            optional_args.append("--rm")

        if ports:
            for host_port, container_port in ports.items():
                optional_args.extend(["-p", f"{host_port}:{container_port}"])

        if extra_args is None:
            extra_args = []

        full_cmd = [
            *docker,
            "run",
            *optional_args,
            *config.docker_run_args,
            *extra_args,
            image,
            *command,
        ]

        logger.debug(f"Running command: {full_cmd}")

        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=False,
            check=False,
        )

        if result.returncode != 0:
            stderr_str = result.stderr.decode("utf-8", errors="ignore")
            if "repository" in stderr_str:
                raise ImageNotFound(stderr_str)
            raise ContainerError(
                None,
                result.returncode,
                shlex.join(full_cmd),
                image,
                result.stderr,
            )

        if detach:
            # If detach is True, stdout prints out the container ID of the running container
            container_id = result.stdout.decode("utf-8", errors="ignore").strip()
            container_obj = Containers.get(container_id)
            return container_obj

        if stdout and stderr:
            return result.stdout, result.stderr
        if stderr:
            return result.stderr
        return result.stdout

    @staticmethod
    def _get_containers(
        include_stopped: bool = False, tesseract_only: bool = True
    ) -> list_[Container]:  # noqa: UP006
        """Updates and retrieves the list of containers by querying Docker CLI.

        Params:
            include_stopped: If True, include stopped containers.
            tesseract_only: If True, only return Tesseract containers.

        Returns:
            List of Container objects.
        """
        docker = _get_docker_executable()
        containers = []

        cmd = [*docker, "ps", "-q"]
        if include_stopped:
            cmd.append("--all")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as ex:
            raise APIError(f"Cannot list Docker containers: {ex}") from ex

        if not result.stdout:
            return []

        container_ids = result.stdout.strip().split("\n")

        # Filter list to  exclude empty strings.
        container_ids = [container_id for container_id in container_ids if container_id]
        json_dicts = get_docker_metadata(container_ids, tesseract_only=tesseract_only)
        for _, json_dict in json_dicts.items():
            container = Container.from_dict(json_dict)
            containers.append(container)

        return containers


@dataclass
class Volume:
    """Volume class to wrap Docker volumes."""

    name: str
    attrs: dict

    @classmethod
    def from_dict(cls, json_dict: dict) -> "Volume":
        """Create an Image object from a json dictionary.

        Params:
            json_dict: The json dictionary to create the object from.

        Returns:
            The created volume object.
        """
        return cls(
            name=json_dict.get("Name", None),
            attrs=json_dict,
        )

    def remove(self, force: bool = False) -> None:
        """Remove a Docker volume.

        Params:
            force: If True, force the removal of the volume.
        """
        docker = _get_docker_executable()
        try:
            _ = subprocess.run(
                [*docker, "volume", "rm", "--force" if force else "", self.name],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as ex:
            raise NotFound(f"Error removing volume {self.name}: {ex}") from ex


class Volumes:
    """Volume class to wrap Docker volumes."""

    @staticmethod
    def create(name: str) -> Volume:
        """Create a Docker volume.

        Params:
            name: The name of the volume to create.

        Returns:
            The created volume object.
        """
        docker = _get_docker_executable()
        try:
            _ = subprocess.run(
                [*docker, "volume", "create", name],
                check=True,
                capture_output=True,
                text=True,
            )
            return Volumes.get(name)
        except subprocess.CalledProcessError as ex:
            raise NotFound(f"Error creating volume {name}: {ex}") from ex

    @staticmethod
    def get(name: str) -> Volume:
        """Get a Docker volume.

        Params:
            name: The name of the volume to get.

        Returns:
            The volume object.
        """
        docker = _get_docker_executable()
        try:
            result = subprocess.run(
                [*docker, "volume", "inspect", name],
                check=True,
                capture_output=True,
                text=True,
            )
            json_dict = json.loads(result.stdout)
        except subprocess.CalledProcessError as ex:
            raise NotFound(f"Volume {name} not found: {ex}") from ex
        if not json_dict:
            raise NotFound(f"Volume {name} not found.")
        return Volume.from_dict(json_dict[0])

    @staticmethod
    def list() -> list[str]:
        """List all Docker volumes.

        Returns:
            List of volume names.
        """
        docker = _get_docker_executable()
        try:
            result = subprocess.run(
                [*docker, "volume", "ls", "-q"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as ex:
            raise APIError(f"Error listing volumes: {ex}") from ex
        return result.stdout.strip().split("\n")


class DockerException(Exception):
    """Base class for Docker CLI exceptions."""

    pass


class BuildError(DockerException):
    """Raised when a build fails."""

    def __init__(self, build_log: list_[str]) -> None:  # noqa: UP006
        self.build_log = build_log

    def __str__(self) -> str:
        return (
            "Docker build failed. Please check the build log for details:\n"
            + "\n".join(self.build_log)
        )


class ContainerError(DockerException):
    """Raised when a container encounters an error."""

    def __init__(
        self,
        container: str | None,
        exit_status: int,
        command: str,
        image: str,
        stderr: bytes,
    ) -> None:
        self.container = container
        self.exit_status = exit_status
        self.command = command
        self.image = image
        self.stderr = stderr


class APIError(DockerException):
    """Raised when a Docker API error occurs."""

    pass


class NotFound(DockerException):
    """Raised when a Docker resource is not found."""

    pass


class ImageNotFound(NotFound):
    """Raised when an image is not found."""

    pass


class CLIDockerClient:
    """Wrapper around Docker CLI to manage Docker containers, images, and projects.

    Initializes a new instance of the current Docker state from the
    perspective of Tesseracts, while mimicking the interface of Docker-Py, with additional
    features for the convenience of Tesseract usage.

    Most calls to CLIDockerClient could be replaced by official Docker-Py Client. However,
    CLIDockerClient by default only sees Tesseract relevant images, containers, and projects;
    the flag `tesseract_only` must be set to False to see non-Tesseract images, containers, and projects.
    """

    def __init__(self) -> None:
        self.containers = Containers()
        self.images = Images()
        self.volumes = Volumes()

    @staticmethod
    def info() -> tuple:
        """Wrapper around docker info call."""
        docker = _get_docker_executable()
        try:
            result = subprocess.run(
                [*docker, "info"],
                check=True,
                capture_output=True,
            )
            return result.stdout, result.stderr
        except subprocess.CalledProcessError as ex:
            raise APIError() from ex


def get_docker_metadata(
    docker_asset_ids: list[str], is_image: bool = False, tesseract_only: bool = True
) -> dict:
    """Get metadata for Docker images/containers.

    Params:
        docker_asset_ids: List of image/container ids to get metadata for.
        is_image: If True, get metadata for images. If False, get metadata for containers.
        tesseract_only: If True, only get metadata for Tesseract images/containers.

    Returns:
        A dict mapping asset ids to their metadata.
    """
    docker = _get_docker_executable()
    if not docker_asset_ids:
        return {}

    # Set metadata in case no images exist and metadata does not get initialized.
    metadata = None
    try:
        result = subprocess.run(
            [*docker, "inspect", *docker_asset_ids],
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
                logger.error(f"Image {asset_id} is not a valid image.")
        if "No such object" in error_message:
            raise APIError("Unhealthy container found. Please restart docker.") from e

    if not metadata:
        return {}

    asset_meta_dict = {}
    # Parse the output into a dictionary of only Tesseract assets
    # with the id as the key for easy parsing, and the metadata as the value.
    for asset in metadata:
        env_vars = asset["Config"]["Env"]
        if tesseract_only and (
            not any("TESSERACT_NAME" in env_var for env_var in env_vars)
        ):
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


def build_docker_image(
    path: str | Path,
    tags: list[str],
    dockerfile: str | Path,
    inject_ssh: bool = False,
    print_and_exit: bool = False,
) -> Image | None:
    """Build a Docker image from a Dockerfile using BuildKit.

    Params:
        path: Path to the directory containing the Dockerfile.
        tag: The name of the image to build.
        dockerfile: path within the build context to the Dockerfile.
        inject_ssh: If True, inject SSH keys into the build.
        print_and_exit: If True, log the build command and exit without building.

    Returns:
        Built Image object if print_and_exit is False, otherwise None.
    """
    # use an instantiated client here, which may be mocked in tests
    client = CLIDockerClient()
    build_args = dict(path=path, tags=tags, dockerfile=dockerfile)

    if inject_ssh:
        ssh_sock = os.environ.get("SSH_AUTH_SOCK")
        if ssh_sock is None:
            raise ValueError(
                "SSH_AUTH_SOCK environment variable not set (try running `ssh-agent`)"
            )

        ssh_keys = subprocess.run(["ssh-add", "-L"], capture_output=True)
        if ssh_keys.returncode != 0 or not ssh_keys.stdout:
            raise ValueError("No SSH keys found in SSH agent (try running `ssh-add`)")
        build_args["ssh"] = f"default={ssh_sock}"

    build_cmd = Images._get_buildx_command(**build_args)

    if print_and_exit:
        logger.info(
            f"To build the Docker image manually, run:\n    $ {shlex.join(build_cmd)}"
        )
        return None

    return client.images.buildx(**build_args)
