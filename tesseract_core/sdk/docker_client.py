# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker client for Tesseract usage."""

from __future__ import annotations

import datetime
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path

logger = logging.getLogger("tesseract")


class CLIDockerClient:
    """Wrapper around Docker CLI to manage Docker containers, images, and projects.

    Initializes a new instance of the current Docker state from the
    perspective of Tesseracts.
    """

    def __init__(self) -> None:
        self.project_container_map = {}  # Mapping from project ID to list of container ids
        self.containers = self.Containers()
        self.images = self.Images()
        self.compose = self.Compose(self)

    class Images:
        """Class to interface with Tesseract docker images."""

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
            if not image:
                raise CLIDockerClient.Errors.DockerException(
                    "Image name cannot be empty."
                )
            if ":" not in image:
                image = image + ":latest"
            # Use getter func to make sure self.images is updated
            images = self.list()
            # Check for both name and id to find the image
            for image_obj in images:
                if image_obj.id == image or image_obj.name == image:
                    return image_obj
            raise CLIDockerClient.Errors.ImageNotFound(f"Image {image} not found.")

        def list(self) -> list[Image]:
            """Returns the current list of images."""
            return self._get_images()

        def remove(self, image_id: str) -> None:
            """Remove an image from the local Docker registry."""
            try:
                _ = subprocess.run(
                    ["docker", "rmi", image_id, "--force"],
                    check=True,
                    capture_output=True,
                )

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

            # Use start time to create a unique label for pruning build cache
            start = datetime.datetime.now()
            label = f"tesseract.{tag}.buildx.{start.strftime('%Y-%m-%dT%H:%M:%S')}"
            build_cmd = [
                "docker",
                "buildx",
                "build",
                "--load",
                "--tag",
                tag,
                "--label",
                label,
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
            if not keep_build_cache:
                try:
                    prune_cmd = [
                        "docker",
                        "builder",
                        "prune",
                        "-a",
                        "-f",
                        "--filter",
                        f"label={label}",
                    ]
                    # Use docker builder prune to remove build cache for everything with label
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
                raise CLIDockerClient.Errors.BuildError(logs)

            return self.get(tag)

        @staticmethod
        def _get_images() -> list[Image]:
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
                raise CLIDockerClient.Errors.APIError(
                    f"Cannot list Docker images: {ex}"
                ) from ex

            if not image_ids.stdout:
                return []

            image_ids = image_ids.stdout.strip().split("\n")
            # Filter list to exclude empty strings.
            image_ids = [image_id for image_id in image_ids if image_id]
            json_dicts = get_docker_metadata(image_ids, is_image=True)
            for _, json_dict in json_dicts.items():
                image = CLIDockerClient.Images.Image(json_dict)
                images.append(image)

            return images

    class Containers:
        """Class to interface with docker containers."""

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
                self.image = json_dict.get("Image", None)
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
                    )
                    return result.returncode, result.stdout
                except subprocess.CalledProcessError as ex:
                    raise CLIDockerClient.Errors.ContainerError(
                        f"Cannot run command in container {self.id}: {ex}"
                    ) from ex

            def logs(self, stdout: bool = True, stderr: bool = True) -> bytes:
                """Get the logs for this container."""
                if not stdout and not stderr:
                    raise ValueError("At least one of stdout or stderr must be True.")

                try:
                    result = subprocess.run(
                        ["docker", "logs", self.id],
                        check=True,
                        stdout=subprocess.PIPE if stdout else subprocess.DEVNULL,
                        stderr=subprocess.STDOUT if stderr else subprocess.DEVNULL,
                    )
                except subprocess.CalledProcessError as ex:
                    raise CLIDockerClient.Errors.ContainerError(
                        f"Cannot get logs for container {self.id}: {ex}"
                    ) from ex

                return result.stdout

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
                    raise CLIDockerClient.Errors.ContainerError(
                        f"Cannot wait for container {self.id}: {ex}"
                    ) from ex

            def remove(self, v: bool = False, link: bool = False, force: bool = False):
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
                        raise CLIDockerClient.Errors.ContainerError(
                            f"Cannot remove container {self.id}: {ex}"
                        ) from ex
                    raise ex

            def __str__(self) -> str:
                string = (
                    f"Container id: {self.id}, name: {self.name}, project_id: {self.project_id},"
                    "ports: {self.host_port}, attrs: {self.attrs}"
                )
                return string

        def list(self, all: bool = False) -> dict:
            """Returns the current list of containers."""
            return self._get_containers(include_stopped=all)

        def get(self, id_or_name: str) -> Container:
            """Returns the metadata for a specific container."""
            container_list = self.list(all=True)

            for container_obj in container_list.values():
                got_container = (
                    container_obj.id == id_or_name
                    or container_obj.short_id == id_or_name
                    or container_obj.name == id_or_name
                )
                if got_container:
                    break
            else:
                raise CLIDockerClient.Errors.ContainerError(
                    f"Container {id_or_name} not found."
                )

            return container_obj

        def run(
            self,
            image: str,
            command: list[str],
            volumes: dict | None = None,
            device_requests: list[int | str] | None = None,
            detach: bool = False,
            remove: bool = False,
        ) -> tuple | Container:
            """Run a command in a container from an image.

            Returns Container object if detach is True, otherwise returns list of stdout and stderr.
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
                raise CLIDockerClient.Errors.ContainerError(
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
                    container_id = result.stdout.strip()
                    container_obj = self.get(container_id)
                    return container_obj

                if result.returncode != 0:
                    raise CLIDockerClient.Errors.ContainerError(
                        "Error running container command."
                    )

                return result.stdout, result.stderr

            except subprocess.CalledProcessError as ex:
                if "repository" in ex.stderr:
                    raise CLIDockerClient.Errors.ImageNotFound() from ex
                if "docker" in ex.stderr:
                    raise CLIDockerClient.Errors.ContainerError(
                        f"Error running container command: `{' '.join(cmd_list)}`. \n\n{ex.stderr}"
                    ) from ex
                raise ex

        @staticmethod
        def _get_containers(include_stopped: bool = False) -> dict[str, Container]:
            containers = {}

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
                raise CLIDockerClient.Errors.APIError(
                    f"Cannot list Docker containers: {ex}"
                ) from ex

            if not result.stdout:
                return {}

            container_ids = result.stdout.strip().split("\n")

            # Filter list to  exclude empty strings.
            container_ids = [
                container_id for container_id in container_ids if container_id
            ]
            json_dicts = get_docker_metadata(container_ids)
            for container_id, json_dict in json_dicts.items():
                container = CLIDockerClient.Containers.Container(json_dict)
                containers[container_id] = container

            return containers

    class Compose:
        """Class to interface with docker projects."""

        def __init__(self, docker_cli):
            self.project_container_map = {}  # Mapping from project ID to list of container ids
            self.containers = docker_cli.Containers()

        def list(self) -> dict:
            """Returns the current list of projects."""
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

        def _update_projects(self) -> None:
            """Updates the list of projects by going through containers."""
            self.project_container_map = {}
            for container_id, container in self.containers.list().items():
                if container.project_id:
                    if container.project_id not in self.project_container_map:
                        self.project_container_map[container.project_id] = []
                    self.project_container_map[container.project_id].append(
                        container_id
                    )

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
