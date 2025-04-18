# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker client for Tesseract usage."""

import json
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path

logger = logging.getLogger("tesseract")


# store a reference to list, which is shadowed by some function names below
list_ = list


class Image:
    """Image class to wrap Docker image details."""

    def __init__(self, json_dict: dict) -> None:
        self.id = json_dict.get("Id", None)
        self.short_id = self.id[:19] if self.id else None
        self.attrs = json_dict
        self.tags = json_dict.get("RepoTags", None)

    def __str__(self) -> str:
        return f"Image id: {self.id}, tags: {self.tags}, attrs: {self.attrs}"


class Images:
    """Namespace for functions to interface with Tesseract docker images."""

    @staticmethod
    def get(image_id_or_name: str | bytes, tesseract_only: bool = True) -> Image:
        """Returns the metadata for a specific image."""
        if not image_id_or_name:
            raise ValueError("Image name cannot be empty.")

        def is_image_id(s: str) -> bool:
            # Check if string is image name or id so we can append tag
            return bool(re.fullmatch(r"(sha256:)?[a-fA-F0-9]{12,64}", s))

        if ":" not in image_id_or_name:
            if not is_image_id(image_id_or_name):
                image_id_or_name = image_id_or_name + ":latest"
            else:
                # If image_id_or_name is an image id, we need to get the full id
                # by prepending sha256
                image_id_or_name = "sha256:" + image_id_or_name
        images = Images.list(tesseract_only=tesseract_only)

        # Check for both name and id to find the image
        # Tags may be prefixed by repository url
        for image_obj in images:
            if (
                image_obj.id == image_id_or_name
                or image_obj.short_id == image_id_or_name
                or image_id_or_name in image_obj.tags
                or (
                    any(
                        tag.split("/")[-1] == image_id_or_name for tag in image_obj.tags
                    )
                )
            ):
                return image_obj

        raise ImageNotFound(f"Image {image_id_or_name} not found.")

    @staticmethod
    def list(tesseract_only: bool = True) -> list_[Image]:
        """Returns the current list of images."""
        return Images._get_images(tesseract_only=tesseract_only)

    @staticmethod
    def remove(image: str) -> None:
        """Remove an image (name or id) from the local Docker registry."""
        # Use the getter to make sure urls are handled properly
        try:
            res = subprocess.run(
                ["docker", "rmi", image, "--force"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as ex:
            raise ImageNotFound(f"Cannot remove image {image}: {ex}") from ex

        if "No such image" in res.stderr:
            raise ImageNotFound(f"Cannot remove image {image}: {res.stderr}")

    @staticmethod
    def buildx(
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

        # NOTE: Do this before error checking to ensure we always prune the cache
        # Prune all until docker builder prune filter is fixed for timestamp or label specification
        # (might prune too much, but that's fine)
        if not keep_build_cache:
            try:
                prune_cmd = [
                    "docker",
                    "builder",
                    "prune",
                    "-a",
                    "-f",
                ]
                # Use docker builder prune to remove build cache
                prune_res = subprocess.run(
                    prune_cmd, check=True, text=True, capture_output=True
                )
                logger.debug("Pruning build cache: %s", prune_cmd)
                logger.debug(prune_res.stdout)
            except subprocess.CalledProcessError as ex:
                logger.warning(
                    "Docker build cache could not be cleared; consider doing so manually."
                )
                logger.debug(ex.stderr)

        if return_code != 0:
            raise BuildError(logs)

        return Images.get(tag)

    @staticmethod
    def _get_images(tesseract_only: bool = True) -> list_[Image]:
        """Gets the list of images by querying Docker CLI."""
        images = []
        try:
            image_ids = subprocess.run(
                ["docker", "images", "-q"],  # List only image IDs
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
            image = Image(json_dict)
            images.append(image)

        return images


class Container:
    """Container class to wrap Docker container details.

    Container class has additional member variable `host_port` that docker-py
    does not have. This is because Tesseract requires frequent access to the host port.
    """

    def __init__(self, json_dict: dict) -> None:
        self.id = json_dict.get("Id", None)
        self.short_id = self.id[:12] if self.id else None
        self.name = json_dict.get("Name", None).lstrip("/")
        ports = json_dict.get("NetworkSettings", None)
        if ports and ports.get("Ports", None):
            ports = ports["Ports"]
            port_key = next(iter(ports))  # Get the first port key
            if ports[port_key]:
                self.host_port = ports[port_key][0].get("HostPort")
        else:
            self.host_port = None
        self.attrs = json_dict
        self.project_id = json_dict.get("Config", None)
        if self.project_id:
            self.project_id = self.project_id["Labels"].get(
                "com.docker.compose.project", None
            )

    @property
    def image(self) -> Image:
        """Gets the image ID of the container."""
        image_id = self.attrs.get("ImageID", self.attrs["Image"])
        if image_id is None:
            return None
        return Images.get(image_id.split(":")[1])

    def exec_run(self, command: list) -> tuple:
        """Run a command in this container.

        Return exit code and stdout.
        """
        try:
            result = subprocess.run(
                ["docker", "exec", self.id, *command],
                check=True,
                capture_output=True,
            )
            return result.returncode, result.stdout
        except subprocess.CalledProcessError as ex:
            raise ContainerError(
                f"Cannot run command in container {self.id}: {ex}"
            ) from ex

    def logs(self, stdout: bool = True, stderr: bool = True) -> bytes:
        """Get the logs for this container.

        Logs needs to be called if container is running in a detached state,
        and we wish to retrieve the logs from the command executing in the container.
        """
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
                ["docker", "logs", self.id],
                check=True,
                stdout=stdout_pipe,
                stderr=stderr_pipe,
            )
        except subprocess.CalledProcessError as ex:
            raise ContainerError(
                f"Cannot get logs for container {self.id}: {ex}"
            ) from ex

        return getattr(result, output_attr)

    def wait(self) -> dict:
        """Wait for container to finish running."""
        try:
            result = subprocess.run(
                ["docker", "wait", self.id],
                check=True,
                capture_output=True,
                text=True,
            )
            # Container's exit code is printed by the wait command
            return {"StatusCode": int(result.stdout)}
        except subprocess.CalledProcessError as ex:
            raise ContainerError(f"Cannot wait for container {self.id}: {ex}") from ex

    def remove(self, v: bool = False, link: bool = False, force: bool = False) -> str:
        """Remove the container."""
        try:
            result = subprocess.run(
                [
                    "docker",
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
                raise ContainerError(f"Cannot remove container {self.id}: {ex}") from ex
            raise ex

    def __str__(self) -> str:
        string = (
            f"Container id: {self.id}, name: {self.name}, project_id: {self.project_id},"
            "ports: {self.host_port}, attrs: {self.attrs}"
        )
        return string


class Containers:
    """Namespace to interface with docker containers."""

    @staticmethod
    def list(all: bool = False, tesseract_only: bool = True) -> list:
        """Returns the current list of containers."""
        return Containers._get_containers(
            include_stopped=all, tesseract_only=tesseract_only
        )

    @staticmethod
    def get(id_or_name: str, tesseract_only: bool = True) -> Container:
        """Returns the metadata for a specific container."""
        container_list = Containers.list(all=True, tesseract_only=tesseract_only)

        for container_obj in container_list:
            got_container = (
                container_obj.id == id_or_name
                or container_obj.short_id == id_or_name
                or container_obj.name == id_or_name
            )
            if got_container:
                break
        else:
            raise ContainerError(f"Container {id_or_name} not found.")

        return container_obj

    @staticmethod
    def run(
        image: str,
        command: list_[str],
        volumes: dict | None = None,
        device_requests: list_[int | str] | None = None,
        detach: bool = False,
        remove: bool = False,
        stdout: bool = True,
        stderr: bool = False,
    ) -> tuple | Container | str:
        """Run a command in a container from an image.

        Returns Container object if detach is True, otherwise returns list of stdout and stderr.

        Detach must be set to True if we wish to retrieve the container id of the running container,
        and if detach is true, we must wait on the container to finish running and retrieve the logs
        of the container manually.

        If remove is set to True, the container will automatically remove itself after it finishes executing
        the command. This means that we cannot set both detach and remove simulataneously to True or else there
        would be no way of retrieving the logs from the removed container.
        """
        # Convert the parsed_volumes into a list of strings in proper argument format,
        # `-v host_path:container_path:mode`.
        # If command is a type string and not list, make list
        if isinstance(command, str):
            command = [command]
        logger.debug(f"Running command: {command}")

        if not volumes:
            volume_args = []
        else:
            volume_args = []
            for host_path, volume_info in volumes.items():
                volume_args.append("-v")
                volume_args.append(
                    f"{host_path}:{volume_info['bind']}:{volume_info['mode']}"
                )

        if device_requests:
            gpus_str = ",".join(device_requests)
            gpus_option = f'--gpus "device={gpus_str}"'
        else:
            gpus_option = ""

        # Remove and detached cannot both be set to true
        if remove and detach:
            raise ContainerError(
                "Cannot set both remove and detach to True when running a container."
            )

        # Run with detached to get the container id of the running container.
        cmd_list = [
            "docker",
            "run",
            *(["-d"] if detach else []),
            *(["--rm"] if remove else []),
            *volume_args,
            *([gpus_option] if gpus_option else []),
            image,
            *command,
        ]

        try:
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                check=True,
            )

            if detach:
                # If detach is True, stdout prints out the container ID of the running container
                container_id = result.stdout.strip()
                container_obj = Containers.get(container_id)
                return container_obj

            if result.returncode != 0:
                raise ContainerError("Error running container command.")

            if stdout and stderr:
                return result.stdout, result.stderr
            if stderr:
                return result.stderr
            return result.stdout

        except subprocess.CalledProcessError as ex:
            if "repository" in ex.stderr:
                raise ImageNotFound() from ex
            if "docker" in ex.stderr:
                raise ContainerError(
                    f"Error running container command: `{' '.join(cmd_list)}`. \n\n{ex.stderr}"
                ) from ex
            raise ex

    @staticmethod
    def _get_containers(
        include_stopped: bool = False, tesseract_only: bool = True
    ) -> list:
        containers = []

        cmd = ["docker", "ps", "-q"]
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
            return {}

        container_ids = result.stdout.strip().split("\n")

        # Filter list to  exclude empty strings.
        container_ids = [container_id for container_id in container_ids if container_id]
        json_dicts = get_docker_metadata(container_ids, tesseract_only=tesseract_only)
        for _, json_dict in json_dicts.items():
            container = Container(json_dict)
            containers.append(container)

        return containers


class Compose:
    """Namespace to interface with docker compose projects."""

    @staticmethod
    def list(include_stopped: bool = False) -> dict:
        """Returns the current list of projects."""
        return Compose._update_projects(include_stopped)

    @staticmethod
    def up(compose_fpath: str, project_name: str) -> str:
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
            # If the project successfully started, try to get the logs from the containers
            project_containers = Compose.list(include_stopped=True).get(
                project_name, None
            )
            if project_containers:
                container = Containers.get(project_containers[0])
                stderr = container.logs(stderr=True)
                raise ContainerError(
                    f"Failed to start Tesseract container: {container.name}, logs: ",
                    stderr,
                ) from ex
            logger.error(str(ex))
            logger.error(ex.stderr.decode())
            raise ContainerError(
                "Failed to start Tesseract containers.", ex.stderr
            ) from ex

    @staticmethod
    def down(project_id: str) -> bool:
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

    @staticmethod
    def exists(project_id: str) -> bool:
        """Check if Docker Compose project exists."""
        return project_id in Compose.list()

    @staticmethod
    def _update_projects(include_stopped: bool = False) -> dict[str, list_[str]]:
        """Updates the list of projects by going through containers."""
        project_container_map = {}
        for container_id, container in Containers.list(include_stopped).items():
            if container.project_id:
                if container.project_id not in project_container_map:
                    project_container_map[container.project_id] = []
                project_container_map[container.project_id].append(container_id)
        return project_container_map


class DockerException(Exception):
    """Base class for Docker CLI exceptions."""

    pass


class BuildError(DockerException):
    """Raised when a build fails."""

    def __init__(self, build_log: str) -> None:
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


class CLIDockerClient:
    """Wrapper around Docker CLI to manage Docker containers, images, and projects.

    Initializes a new instance of the current Docker state from the
    perspective of Tesseracts, while mimicking the interface of Docker-Py, with additional
    features for the convenience of Tesseract usage.

    Most calls to CLIDockerClient could be replaced by official Docker-Py Client. However,
    CLIDockerClient by default only sees Tesseract relevant images, containers, and projects;
    the flag `tesseract_only` must be set to False to see non-Tesseract images, containers, and projects.
    CLIDockerClient also has an additional `compose` class for project management that
    Docker-Py does not have due to the Tesseract use case.
    """

    def __init__(self) -> None:
        self.containers = Containers()
        self.images = Images()
        self.compose = Compose()

    @staticmethod
    def info() -> tuple:
        """Wrapper around docker info call."""
        try:
            result = subprocess.run(
                ["docker", "info"],
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
                logger.error(f"Image {asset_id} is not a valid image.")
        if "No such object" in error_message:
            raise ContainerError(
                "Unhealthy container found. Please restart docker."
            ) from e

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
