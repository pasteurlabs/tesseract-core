"""Docker for Tesseract usage."""

import json
import logging
import os
import shlex
import subprocess
from pathlib import Path

logger = logging.getLogger("tesseract")


class DockerWrapper:
    """Wrapper around Docker CLI to manage Docker containers and images.

    Initializes a new instance of the current Docker state from the
    perspective of Tesseracts. Loads in a previous state if one exists.
    """

    def __init__(self) -> None:
        self.containers = {}  # Dict of container id to Container object
        self.images = []  # List of Image objects
        self.project_container_map = {}  # Mapping from project ID to list of container ids

    class Container:
        """Container class to wrap Docker container details."""

        def __init__(self, json_dict: dict) -> None:
            self.id = json_dict.get("Id", None)
            self.name = json_dict.get("Name", None)
            ports = json_dict.get("NetworkSettings", None)
            if ports:
                ports = ports["Ports"]
                port_key = next(iter(ports))  # Get the first port key
                self.host_port = ports[port_key][0]["HostPort"]  # Get the host port
            self.attrs = json_dict
            self.project_id = json_dict.get("Config", None)
            if self.project_id:
                self.project_id = self.project_id["Labels"].get(
                    "com.docker.compose.project", None
                )

    class Image:
        """Image class to wrap Docker image details."""

        def __init__(self, json_dict: dict) -> None:
            self.id = json_dict.get("Id", None)
            self.short_id = self.id[:12] if self.id else None
            self.attrs = json_dict
            self.tags = json_dict.get("RepoTags", None)
            self.name = self.tags[0] if self.tags else None

    def get_all_containers(self) -> dict:
        """Returns the current list of containers."""
        if not self.containers:
            self._update_containers()
        return self.containers

    def get_all_images(self) -> dict:
        """Returns the current list of images."""
        if not self.images:
            self._update_images()
        return self.images

    def get_projects(self) -> dict:
        """Returns the current list of projects."""
        if not self.project_container_map:
            # Check if containers is updated
            self.get_all_containers()
            self._update_projects()
        return self.project_container_map

    def get_container(self, container: str) -> Container:
        """Returns the metadata for a specific container."""
        # Use the get_all_containers() function to make sure it's updated
        # First check if we can find the container by id
        container_obj = self.get_all_containers().get(container, None)
        if not container_obj:
            # Check for container names if id does not work
            for _, container_obj in self.get_all_containers().items():
                if container_obj.name == container:
                    return container_obj
        return container_obj

    def get_image(self, image: str) -> Image:
        """Returns the metadata for a specific image."""
        # Iterate through all the images and see if any of them
        # have id or name matching the image str

        # Use getter func to make sure it's updated
        if ":" not in image:
            image = image + ":latest"
        images = self.get_all_images()
        for image_obj in images:
            if image_obj.id == image or image_obj.name == image:
                return image_obj
        raise ValueError(f"Image {image} not found.")

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
            raise RuntimeError(f"Cannot remove image {image_id}: {ex}") from ex

    def docker_compose_up(self, compose_fpath: str, project_name: str) -> bool:
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
            # Update project containers map with the new project
            containers = self._project_containers(project_name)
            json_dicts = self._get_docker_metadata(containers)
            for container_id, json_dict in json_dicts.items():
                container = self.Container(json_dict)
                self.containers[container_id] = container
            return project_name
        except subprocess.CalledProcessError as ex:
            logger.error(str(ex))
            logger.error(ex.stderr.decode())
            raise RuntimeError("Failed to start Tesseract containers.") from ex

    def docker_compose_down(self, project_id: str) -> bool:
        """Stop and remove containers and networks associated to a project."""
        try:
            __ = subprocess.run(
                ["docker", "compose", "-p", project_id, "down"],
                check=True,
                capture_output=True,
            )
            # Remove the project from the project_container_map
            # Remove the containers from the containers list
            containers = self.get_projects().get(project_id, None)
            for container in containers:
                del self.containers[container]
            del self.project_container_map[project_id]
            return True
        except subprocess.CalledProcessError as ex:
            logger.error(str(ex))
            return False

    def docker_compose_project_exists(self, project_id: str) -> bool:
        """Check if Docker Compose project exists."""
        return project_id in self.get_projects()

    def docker_info(self) -> tuple:
        """Wrapper around docker info call."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                check=True,
                capture_output=True,
            )
            return result.stdout, result.stderr
        except subprocess.CalledProcessError as ex:
            raise RuntimeError() from ex

    def exec_run(self, container_id: str, command: str) -> tuple:
        """Run a command in a running container.

        Return exit code and stdout.
        """
        try:
            print("AKOAKO === ENTERING EXEC_RUN")
            result = subprocess.run(
                ["docker", "exec", container_id, *command],
                check=True,
                capture_output=True,
                text=True,
            )
            print("AKOAKO === EXITING EXEC_RUN")
            return result.returncode, result.stdout
        except subprocess.CalledProcessError as ex:
            raise RuntimeError(
                f"Cannot run command in container {container_id}: {ex}"
            ) from ex

    def run_container(
        self,
        image_id: str,
        command: str,
        parsed_volumes: dict,
        gpus: list[int | str] | None = None,
    ) -> bool:
        """Run a command in a container from an image."""
        # Convert the parsed_volumes into a list of strings in proper argument format,
        # `-v host_path:container_path:mode`.
        if not parsed_volumes:
            parsed_volumes = []
        else:
            parsed_volumes = [
                f"-v {host_path}:{volume_info['bind']}:{volume_info['mode']}"
                for host_path, volume_info in parsed_volumes.items()
            ]

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
            *parsed_volumes,
            *([gpus_option] if gpus_option else []),
            image_id,
            *command,
        ]

        try:
            result = subprocess.run(
                cmd_list,
                check=True,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"{result.stderr}")

            return result.stdout, result.stderr

        except subprocess.CalledProcessError as ex:
            raise RuntimeError(f"{ex.stderr}") from ex

    def docker_buildx(
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
            raise RuntimeError("Error while building Docker image", logs)

        # Get image object
        image = self.get_image(tag)
        return image

    def _update_containers(self) -> None:
        """Update self.containers."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-q"],  # List only container IDs
                capture_output=True,
                text=True,
                check=True,
            )
            container_ids = result.stdout.strip().split("\n")

            # Check if theres any cleaned up containers.
            for container_id in self.containers:
                if container_id not in container_ids:
                    del self.containers[container_id]

            # Filter list to exclude container ids that are already in self.containers
            # also exclude empty strings.
            container_ids = [
                container_id
                for container_id in container_ids
                if container_id not in self.containers and container_id
            ]
            json_dicts = self._get_docker_metadata(container_ids)
            for container_id, json_dict in json_dicts.items():
                container = self.Container(json_dict)
                self.containers[container_id] = container

        except subprocess.CalledProcessError as ex:
            raise RuntimeError(f"Cannot list Docker containers: {ex}") from ex

    def _update_images(self) -> None:
        """Updates the list of images by querying Docker CLI."""
        try:
            image_ids = subprocess.run(
                ["docker", "images", "-q"],  # List only image IDs
                capture_output=True,
                text=True,
                check=True,
            )
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
            json_dicts = self._get_docker_metadata(images, is_image=True)
            for _, json_dict in json_dicts.items():
                image = self.Image(json_dict)
                self.images.append(image)

        except subprocess.CalledProcessError as ex:
            raise RuntimeError(f"Cannot list Docker images: {ex}") from ex

    def _update_projects(self) -> None:
        """Updates the list of projects by going through containers."""
        for container_id, container in self.containers.items():
            if container.project_id:
                if container.project_id not in self.project_container_map:
                    self.project_container_map[container.project_id] = []
                self.project_container_map[container.project_id].append(container_id)

    def _get_docker_metadata(
        self, docker_asset_ids: list[str], is_image: bool = False
    ) -> dict:
        """Get metadata for Docker images/containers.

        Returns a dict mapping asset ids to their metadata.
        """
        if not docker_asset_ids:
            return {}
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
                line.split()[0]  # Container id is the second element in the output line
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
