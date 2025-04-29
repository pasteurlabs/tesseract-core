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

logger = logging.getLogger("tesseract")


# store a reference to list, which is shadowed by some function names below
list_ = list


@dataclass
class Image:
    """Image class to wrap Docker image details."""

    id: str
    short_id: str
    tags: list[str] | None
    attrs: dict

    @classmethod
    def from_dict(cls, json_dict: dict) -> "Image":
        """Create an Image object from a json dictionary."""
        return cls(
            id=json_dict.get("Id", None),
            short_id=json_dict.get("Id", [])[:19],
            tags=json_dict.get("RepoTags", None),
            attrs=json_dict,
        )


class Images:
    """Namespace for functions to interface with Tesseract docker images."""

    @staticmethod
    def get(image_id_or_name: str | bytes, tesseract_only: bool = True) -> Image:
        """Returns the metadata for a specific image.

        Params:
            image_id_or_name: The image name or id to get.
            tesseract_only: If True, only retrieves Tesseract images.

        Returns:
            Image object.
        """
        if not image_id_or_name:
            raise ValueError("Image name cannot be empty.")

        def is_image_id(s: str) -> bool:
            """Check if string is image name or id by checking if it's sha256 format."""
            return bool(re.fullmatch(r"(sha256:)?[a-fA-F0-9]{12,64}", s))

        if ":" not in image_id_or_name:
            # Check if image param is a name or id so we can append latest tag if needed
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
    def _get_buildx_command(
        path: str | Path,
        tag: str,
        dockerfile: str | Path,
        ssh: str | None = None,
    ) -> list_[str]:
        """Get the buildx command for building Docker images.

        Returns:
            The buildx command as a list of strings.
        """
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

        if ssh is not None:
            build_cmd.extend(["--ssh", ssh])

        return build_cmd

    @staticmethod
    def buildx(
        path: str | Path,
        tag: str,
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

        build_cmd = Images._get_buildx_command(
            path=path,
            tag=tag,
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

        return Images.get(tag)

    @staticmethod
    def _get_images(tesseract_only: bool = True) -> list_[Image]:
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

    Container class has additional member variable `host_port` that docker-py
    does not have. This is because Tesseract requires frequent access to the host port.
    """

    id: str
    short_id: str
    name: str
    attrs: dict

    @classmethod
    def from_dict(cls, json_dict: dict) -> "Container":
        """Create a Container object from a json dictionary."""
        return cls(
            id=json_dict.get("Id", None),
            short_id=json_dict.get("Id", [])[:12],
            name=json_dict.get("Name", None).lstrip("/"),
            attrs=json_dict,
        )

    @property
    def image(self) -> Image | None:
        """Gets the image ID of the container."""
        image_id = self.attrs.get("ImageID", self.attrs["Image"])
        if image_id is None:
            return None
        return Images.get(image_id.split(":")[1])

    @property
    def host_port(self) -> str | None:
        """Gets the host port of the container."""
        if self.attrs.get("NetworkSettings", None):
            ports = self.attrs["NetworkSettings"].get("Ports", None)
            if ports:
                port_key = next(iter(ports))  # Get the first port key
                if ports[port_key]:
                    return ports[port_key][0].get("HostPort")
        return None

    @property
    def project_id(self) -> str | None:
        """Gets the project ID of the container."""
        project_id = self.attrs.get("Config", None)
        if project_id:
            project_id = project_id["Labels"].get("com.docker.compose.project", None)
        return project_id

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
            raise ContainerError(
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
            raise ContainerError(f"Cannot wait for container {self.id}: {ex}") from ex

    def remove(self, v: bool = False, link: bool = False, force: bool = False) -> str:
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
                raise ContainerError(f"Cannot remove container {self.id}: {ex}") from ex
            raise ex


class Containers:
    """Namespace to interface with docker containers."""

    @staticmethod
    def list(all: bool = False, tesseract_only: bool = True) -> list:
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
            stdout: If True, return stdout.
            stderr: If True, return stderr.

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
        """Updates and retrieves the list of containers by querying Docker CLI.

        Params:
            include_stopped: If True, include stopped containers.
            tesseract_only: If True, only return Tesseract containers.

        Returns:
            List of Container objects.
        """
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
            return []

        container_ids = result.stdout.strip().split("\n")

        # Filter list to  exclude empty strings.
        container_ids = [container_id for container_id in container_ids if container_id]
        json_dicts = get_docker_metadata(container_ids, tesseract_only=tesseract_only)
        for _, json_dict in json_dicts.items():
            container = Container.from_dict(json_dict)
            containers.append(container)

        return containers


class Compose:
    """Custom namespace to interface with docker compose projects.

    There is no equivalent for this class in docker-py; however, we frequently
    interact with docker compose projects in Tesseract and this namespace makes
    such interactions easier.
    """

    @staticmethod
    def list(include_stopped: bool = False) -> dict:
        """Returns the current list of projects.

        Params:
            include_stopped: If True, include stopped projects.

        Returns:
            Dict of projects, with the project name as the key and a list of container ids as the value.
        """
        return Compose._update_projects(include_stopped)

    @staticmethod
    def up(compose_fpath: str, project_name: str) -> str:
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

    @staticmethod
    def exists(project_id: str) -> bool:
        """Check if Docker Compose project exists.

        Params:
            project_id: The project name to check.

        Returns:
            True if the project exists, False otherwise.
        """
        return project_id in Compose.list()

    @staticmethod
    def _update_projects(include_stopped: bool = False) -> dict[str, list_[str]]:
        """Updates the list of projects by going through containers.

        Params:
            include_stopped: If True, include stopped projects.

        Returns:
            Dict of projects, with the project name as the key and a list of container ids as the value.
        """
        project_container_map = {}
        for container in Containers.list(include_stopped):
            if container.project_id:
                if container.project_id not in project_container_map:
                    project_container_map[container.project_id] = []
                project_container_map[container.project_id].append(container.id)
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


def build_docker_image(
    path: str | Path,
    tag: str,
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
    build_args = dict(path=path, tag=tag, dockerfile=dockerfile)

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
