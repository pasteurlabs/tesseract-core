# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker client for Tesseract usage."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path

logger = logging.getLogger("tesseract")


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
        self.project_container_map = {}  # Mapping from project ID to list of container ids
        self.containers = self.Containers()
        self.images = self.Images()
        self.compose = self.Compose(self)

    class Images:
        """Class to interface with Tesseract docker images."""

        def __init__(self) -> None:
            self.images = []  # List of Image objects

        class Image:
            """Image class to wrap Docker image details.

            Image class has additional field `name` to store the image name that docker-py does not have.
            Every unique `image_name` will have its own image object.

            Dangling images are not included in the list of images.
            """

            def __init__(self, json_dict: dict, repo_tag_idx: int = 0) -> None:
                self.id = json_dict.get("Id", None)
                self.short_id = self.id[7:19] if self.id else None  # Trim off sha:256
                self.attrs = json_dict
                self.tags = json_dict.get("RepoTags", None)
                # Docker images may be prefixed with the registry URL and may have multiple repo tags
                self.name = (
                    self.tags[repo_tag_idx].split("/")[-1] if self.tags else None
                )

            def __str__(self) -> str:
                return f"Image id: {self.id}, name: {self.name}, tags: {self.tags}, attrs: {self.attrs}"

        def get(self, image_id_or_name: str, tesseract_only: bool = True) -> Image:
            """Returns the metadata for a specific image.

            Params:
                image_id_or_name: The image name or id to get.
                tesseract_only: If True, only retrieves Tesseract images.

            Returns:
                Image object.
            """
            if not image_id_or_name:
                raise CLIDockerClient.Errors.DockerException(
                    "Image name cannot be empty."
                )

            def is_image_id(s: str) -> bool:
                """Check if string is image name or id by checking if it's sha256 format."""
                return bool(re.fullmatch(r"(sha256:)?[a-fA-F0-9]{12,64}", s))

            if ":" not in image_id_or_name:
                # Check if image param is a name or id so we can append latest tag if needed
                if not is_image_id(image_id_or_name):
                    image_id_or_name = image_id_or_name + ":latest"

            # Use getter func to make sure self.images is updated
            images = self.list(tesseract_only=tesseract_only)
            # Check for both name and id to find the image
            for image_obj in images:
                if (
                    image_obj.id == image_id_or_name
                    or image_obj.name == image_id_or_name
                    or image_obj.short_id == image_id_or_name
                ):
                    return image_obj
            raise CLIDockerClient.Errors.ImageNotFound(
                f"Image {image_id_or_name} not found."
            )

        def list(self, tesseract_only: bool = True) -> list[Image]:
            """Returns the current list of images.

            Params:
                tesseract_only: If True, only return Tesseract images.

            Returns:
                List of Image objects.
            """
            return self._get_images(tesseract_only=tesseract_only)

        def remove(self, image: str) -> None:
            """Remove an image (name or id) from the local Docker registry.

            Params:
                image: The image name or id to remove.
            """
            try:
                _ = subprocess.run(
                    ["docker", "rmi", image, "--force"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as ex:
                raise CLIDockerClient.Errors.ImageNotFound(
                    f"Cannot remove image {image}: {ex}"
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
            """Build a Docker image from a Dockerfile using BuildKit.

            Params:
                path:  Path to the directory containing the Dockerfile.
                tag:   The name of the image to build.
                dockerfile: path within the build context to the Dockerfile.
                inject_ssh: If True, inject SSH keys into the build.
                keep_build_cache: If True, keep the build cache.
                print_and_exit: If True, print the build command and exit without building.

            Returns:
                Image object if the build was successful, None otherwise.
            """
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
                raise CLIDockerClient.Errors.BuildError(logs)

            # Tag may be prefixed by repository url
            tag = tag.split("/")[-1]
            return self.get(tag)

        @staticmethod
        def _get_images(tesseract_only: bool = True) -> list[Image]:
            """Gets the list of images by querying Docker CLI.

            Params:
                tesseract_only: If True, only return Tesseract images.

            Returns:
                List of (non-dangling) Image objects.
            """
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

            # If image shows up multiple times, that means it is tagged multiple times
            # So we need to make multiple copies of the image with different names
            image_counts = {}
            for image_id in image_ids:
                image_counts[image_id] = image_counts.get(image_id, 0) + 1

            json_dicts = get_docker_metadata(
                image_ids, is_image=True, tesseract_only=tesseract_only
            )
            for _, json_dict in json_dicts.items():
                # Get short id since that's what is stored in image_counts
                image_id = json_dict.get("Id", None)[7:19]
                if not image_id:
                    continue
                # Set the repotag index to the # of image id - 1 to make sure we create
                # copies of image for each tag (aka unique image name for each)
                idx = image_counts.get(image_id)
                while idx >= 1:
                    image = CLIDockerClient.Images.Image(
                        json_dict, repo_tag_idx=image_counts.get(image_id) - 1
                    )
                    image_counts[image_id] = image_counts.get(image_id, 0) - 1
                    images.append(image)
                    idx = image_counts.get(image_id, 0)

            return images

    class Containers:
        """Class to interface with docker containers."""

        class Container:
            """Container class to wrap Docker container details.

            Container class has additional member variable `host_port` that docker-py
            does not have. This is because Tesseract requires frequent access to the host port.
            """

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
                """Get the logs for this container.

                Logs needs to be called if container is running in a detached state,
                and we wish to retrieve the logs from the command executing in the container.

                Params:
                    stdout: If True, return stdout.
                    stderr: If True, return stderr.
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
                    raise CLIDockerClient.Errors.ContainerError(
                        f"Cannot get logs for container {self.id}: {ex}"
                    ) from ex

                return getattr(result, output_attr)

            def wait(self) -> dict:
                """Wait for container to finish running.

                Returns:
                    A dict with the exit code of the container.
                """
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

            def remove(
                self, v: bool = False, link: bool = False, force: bool = False
            ) -> str:
                """Remove the container.

                Params:
                    v: If True, remove volumes associated with the container.
                    link: If True, remove links to the container.
                    force: If True, force remove the container.

                Returns:
                    The output of the remove command.
                """
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

        def list(self, all: bool = False, tesseract_only: bool = True) -> dict:
            """Returns the current list of containers.

            Params:
                all: If True, include stopped containers.
                tesseract_only: If True, only return Tesseract containers.

            Returns:
                Dict of Container objects, with the container id as the key.
            """
            return self._get_containers(
                include_stopped=all, tesseract_only=tesseract_only
            )

        def get(self, id_or_name: str, tesseract_only: bool = True) -> Container:
            """Returns the metadata for a specific container.

            Params:
                id_or_name: The container name or id to get.
                tesseract_only: If True, only retrieves Tesseract containers.

            Returns:
                Container object.
            """
            container_list = self.list(all=True, tesseract_only=tesseract_only)

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

            Params:
                image: The image name or id to run the command in.
                command: The command to run in the container.
                volumes: A dict of volumes to mount in the container.
                device_requests: A list of device requests for the container.
                detach: If True, run the container in detached mode. Detach must be set to
                        True if we wish to retrieve the container id of the running container,
                        and if detach is true, we must wait on the container to finish
                        running and retrieve the logs of the container manually.
                remove: If remove is set to True, the container will automatically remove itself
                        after it finishes executing the command. This means that we cannot set
                        both detach and remove simulataneously to True or else there
                        would be no way of retrieving the logs from the removed container.

            Returns:
              Container object if detach is True, otherwise returns list of stdout and stderr.

            """
            # If command is a type string and not list, make list
            if isinstance(command, str):
                command = [command]
            logger.debug(f"Running command: {command}")

            # Convert the parsed_volumes into a list of strings in proper argument format,
            # `-v host_path:container_path:mode`.
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
                    # If detach is True, stdout prints out the container ID of the running container
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
        def _get_containers(
            include_stopped: bool = False, tesseract_only: bool = True
        ) -> dict[str, Container]:
            """Updates and retrieves the list of containers by querying Docker CLI.

            Params:
                include_stopped: If True, include stopped containers.
                tesseract_only: If True, only return Tesseract containers.

            Returns:
                Dict of Container objects, with the container id as the key.
            """
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
            json_dicts = get_docker_metadata(
                container_ids, tesseract_only=tesseract_only
            )
            for container_id, json_dict in json_dicts.items():
                container = CLIDockerClient.Containers.Container(json_dict)
                containers[container_id] = container

            return containers

    class Compose:
        """Class to interface with docker projects."""

        def __init__(self, docker_cli):
            self.project_container_map = {}  # Mapping from project ID to list of container ids
            self.containers = docker_cli.Containers()

        def list(self, include_stopped: bool = False) -> dict:
            """Returns the current list of projects.

            Params:
                include_stopped: If True, include stopped projects.

            Returns:
                Dict of projects, with the project name as the key and a list of container ids as the value.
            """
            self._update_projects(include_stopped)
            return self.project_container_map

        def up(self, compose_fpath: str, project_name: str) -> str:
            """Start containers using Docker Compose template.

            Params:
                compose_fpath: Path to the Docker Compose template.
                project_name: Name of the project.

            Returns:
                The project name.
            """
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
                project_containers = self.list(include_stopped=True).get(
                    project_name, None
                )
                if project_containers:
                    container = self.containers.get(project_containers[0])
                    stderr = container.logs(stderr=True)
                    raise CLIDockerClient.Errors.ContainerError(
                        f"Failed to start Tesseract container: {container.name}, logs: ",
                        stderr,
                    ) from ex
                logger.error(str(ex))
                logger.error(ex.stderr.decode())
                raise CLIDockerClient.Errors.ContainerError(
                    "Failed to start Tesseract containers.", ex.stderr
                ) from ex

        def down(self, project_id: str) -> bool:
            """Stop and remove containers and networks associated to a project.

            Params:
                project_id: The project name to stop.

            Returns:
                True if the project was stopped successfully, False otherwise.
            """
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
            """Check if Docker Compose project exists.

            Params:
                project_id: The project name to check.

            Returns:
                True if the project exists, False otherwise.
            """
            return project_id in self.list()

        def _update_projects(self, include_stopped: bool = False) -> None:
            """Updates the list of projects by going through containers."""
            self.project_container_map = {}
            for container_id, container in self.containers.list(
                include_stopped
            ).items():
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


def get_docker_metadata(
    docker_asset_ids: list[str], is_image: bool = False, tesseract_only=True
) -> dict:
    """Get metadata for Docker images/containers.

    Params:
        docker_asset_ids: List of image/container ids to get metadata for.
        is_image: If True, get metadata for images. If False, get metadata for containers.
        tesseract_only: If True, only get metadata for Tesseract images/containers.

    Returns:
        A dict mapping asset ids to their metadata.
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
