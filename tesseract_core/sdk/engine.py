# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine to power Tesseract commands."""

import contextlib
import datetime
import json
import linecache
import logging
import optparse
import os
import random
import shlex
import string
import subprocess
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from shutil import copy, copytree, rmtree
from typing import Any

import docker
import docker.errors
import docker.models
import docker.models.containers
import docker.models.images
from docker.types import DeviceRequest
from jinja2 import Environment, PackageLoader, StrictUndefined
from pip._internal.index.package_finder import PackageFinder
from pip._internal.network.session import PipSession
from pip._internal.req.req_file import (
    RequirementsFileParser,
    get_line_parser,
    handle_line,
)

from .api_parse import TesseractConfig, get_config, validate_tesseract_api

logger = logging.getLogger("tesseract")

# Jinja2 Environment
ENV = Environment(
    loader=PackageLoader("tesseract_core.sdk", "templates"),
    undefined=StrictUndefined,
)


class LogPipe(threading.Thread):
    """Custom wrapper to support live logging from a subprocess via a pipe.

    Runs a thread that logs everything read from the pipe to the standard logger.
    Can be used as a context manager for automatic cleanup.
    """

    daemon = True

    def __init__(self, level: int) -> None:
        """Initialize the LogPipe with the given logging level."""
        super().__init__()
        self._level = level
        self._fd_read, self._fd_write = os.pipe()
        self._pipe_reader = os.fdopen(self._fd_read)
        self._captured_lines = []

    def __enter__(self) -> int:
        """Start the thread and return the write file descriptor of the pipe."""
        self.start()
        return self.fileno()

    def __exit__(self, *args: Any) -> None:
        """Close the pipe and join the thread."""
        os.close(self._fd_write)
        # Use a timeout so something weird happening in the logging thread doesn't
        # cause this to hang indefinitely
        self.join(timeout=10)
        # Do not close reader before thread is joined since there may be pending data
        # This also closes the fd_read pipe
        self._pipe_reader.close()

    def fileno(self) -> int:
        """Return the write file descriptor of the pipe."""
        return self._fd_write

    def run(self) -> None:
        """Run the thread, logging everything."""
        for line in iter(self._pipe_reader.readline, ""):
            if line.endswith("\n"):
                line = line[:-1]
            self._captured_lines.append(line)
            logger.log(self._level, line)

    @property
    def captured_lines(self) -> list[str]:
        """Return all lines captured so far."""
        return self._captured_lines


def needs_docker(func: Callable) -> Callable:
    """A decorator for functions that rely on docker daemon."""
    import functools

    @functools.wraps(func)
    def wrapper_needs_docker(*args: Any, **kwargs: Any) -> None:
        try:
            docker.from_env().info()
        except (
            FileNotFoundError,
            docker.errors.APIError,
            docker.errors.DockerException,
        ) as ex:
            message = "Could not reach Docker daemon, check if it is running."
            logger.error(f"{message} Details: {ex}")
            raise RuntimeError(f"{message} See logs for details.") from None

        return func(*args, **kwargs)

    return wrapper_needs_docker


def parse_requirements(
    filename: str | Path,
    session: PipSession | None = None,
    finder: PackageFinder | None = None,
    options: optparse.Values | None = None,
    constraint: bool = False,
) -> tuple[list[str], list[str]]:
    """Split local dependencies from remote ones in a pip-style requirements file.

    All CLI options that may be part of the given requiremets file are included in
    the remote dependencies.
    """
    if session is None:
        session = PipSession()

    local_dependencies = []
    remote_dependencies = []

    line_parser = get_line_parser(finder)
    parser = RequirementsFileParser(session, line_parser)

    for parsed_line in parser.parse(str(filename), constraint):
        line = linecache.getline(parsed_line.filename, parsed_line.lineno)
        line = line.strip()
        parsed_req = handle_line(
            parsed_line, options=options, finder=finder, session=session
        )
        if not hasattr(parsed_req, "requirement"):
            # this is probably a cli option like --extra-index-url, so we make
            # sure to keep it.
            remote_dependencies.append(line)
        elif parsed_line.requirement.startswith((".", "/", "file://")):
            local_dependencies.append(line)
        else:
            remote_dependencies.append(line)
    return local_dependencies, remote_dependencies


def docker_buildx(
    path: str | Path,
    tag: str,
    dockerfile: str | Path,
    inject_ssh: bool = False,
    keep_build_cache: bool = False,
    print_and_exit: bool = False,
) -> docker.models.images.Image | None:
    """Build a Docker image from a Dockerfile using BuildKit."""
    # Build the Docker image
    # docker-py does not support BuildKit, so we shell out to the Docker CLI
    # see https://github.com/docker/docker-py/issues/2230
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
            raise ValueError("No SSH keys found in SSH agent (try running `ssh-add`)")

        build_cmd += ["--ssh", f"default={ssh_sock}"]

    if print_and_exit:
        logger.info(
            f"To build the Docker image manually, run:\n    $ {shlex.join(build_cmd)}"
        )
        return None

    # Record start time for cache pruning -- this isn't perfect, but should be good enough
    # (might prune too much if multiple builds are running at the same time, but that's fine)
    start = datetime.datetime.now()

    out_pipe = LogPipe(logging.DEBUG)
    with out_pipe as out_pipe_fd:
        proc = subprocess.run(build_cmd, stdout=out_pipe_fd, stderr=out_pipe_fd)

    logs = out_pipe.captured_lines
    return_code = proc.returncode

    # NOTE: Do this before error checking to ensure we always prune the cache
    if not keep_build_cache:
        try:
            with contextlib.closing(docker.APIClient()) as api_client:
                api_client.prune_builds(all=True, filters={"until": start.isoformat()})
        except docker.errors.DockerException:
            logger.warning(
                "Docker build cache could not be cleared; consider doing so manually."
            )

    if return_code != 0:
        raise docker.errors.BuildError("Error while building Docker image", logs)

    # Get image object
    with contextlib.closing(docker.from_env()) as client:
        image = client.images.get(tag)

    return image


def get_runtime_dir() -> Path:
    """Get the source directory for the Tesseract runtime as a context manager."""
    import tesseract_core

    return Path(tesseract_core.__file__).parent / "runtime"


def create_dockerfile(user_config: TesseractConfig, use_ssh_mount: bool = False) -> str:
    """Create the Dockerfile for the package.

    Args:
        user_config: The Tesseract configuration object.
        use_ssh_mount: Whether to use SSH mount to install dependencies (prevents caching).

    Returns:
        A string with the Dockerfile content.
    """
    template_name = "Dockerfile.base"
    template = ENV.get_template(template_name)

    template_values = {
        "tesseract_source_directory": "__tesseract_source__",
        "tesseract_runtime_location": "__tesseract_runtime__",
        "config": user_config,
        "use_ssh_mount": use_ssh_mount,
    }

    logger.debug(f"Generating Dockerfile from template: {template_name}")

    return template.render(template_values)


def build_image(
    src_dir: str | Path,
    image_name: str,
    dockerfile: str | Path,
    build_dir: str | Path,
    inject_ssh: bool = False,
    keep_build_cache: bool = False,
    generate_only: bool = False,
) -> docker.models.images.Image | None:
    """Build the image from a Dockerfile.

    Returns:
        An Image object representing an image on the local machine.
    """
    dockerfile_path = build_dir / "Dockerfile"
    with open(dockerfile_path, "w") as f:
        logger.debug(f"Writing Dockerfile to {dockerfile_path}")
        f.write(dockerfile)

    copytree(src_dir, build_dir / "__tesseract_source__")

    if (Path(src_dir) / "tesseract_requirements.txt").exists():
        local_dependencies, remote_dependencies = parse_requirements(
            Path(src_dir) / "tesseract_requirements.txt"
        )
    else:
        remote_dependencies = []
        local_dependencies = []

    if local_dependencies:
        local_requirements_path = build_dir / "local_requirements"
        Path.mkdir(local_requirements_path)
        for dependency in local_dependencies:
            src = Path(src_dir) / dependency
            dest = build_dir / "local_requirements" / src.name
            if src.is_file():
                copy(src, dest)
            else:
                copytree(src, dest)

    # We need to write a new requirements file in the build dir, where we explicitly
    # removed the local dependencies
    requirements_file_path = (
        Path(build_dir) / "__tesseract_source__" / "tesseract_requirements.txt"
    )
    with requirements_file_path.open("w", encoding="utf-8") as f:
        for dependency in remote_dependencies:
            f.write(f"{dependency}\n")

    def _ignore_pycache(_, names: list[str]) -> list[str]:
        ignore = []
        if "__pycache__" in names:
            ignore.append("__pycache__")
        return ignore

    runtime_source_dir = get_runtime_dir()
    copytree(
        runtime_source_dir,
        build_dir / "__tesseract_runtime__" / "tesseract_core" / "runtime",
        ignore=_ignore_pycache,
    )
    for metafile in (runtime_source_dir / "meta").glob("*"):
        copy(metafile, build_dir / "__tesseract_runtime__")

    if generate_only:
        logger.info(f"Build directory generated at {build_dir}, skipping build")
    else:
        logger.info("Building image ...")

    try:
        image = docker_buildx(
            path=build_dir.as_posix(),
            tag=image_name,
            dockerfile=dockerfile_path,
            inject_ssh=inject_ssh,
            keep_build_cache=keep_build_cache,
            print_and_exit=generate_only,
        )

    except docker.errors.BuildError as e:
        logger.warning("Build failed with logs:")
        for line in e.build_log:
            logger.warning(line)
        raise e
    else:
        if image is not None:
            logger.debug("Build successful")

    return image


def _write_template_file(
    template_name: str,
    target_dir: Path,
    template_vars: dict,
    recipe: Path = Path("."),
    exist_ok: bool = False,
):
    """Write a template to a target directory."""
    template = ENV.get_template(str(recipe / template_name))

    target_file = target_dir / template_name

    if target_file.exists() and not exist_ok:
        raise FileExistsError(f"File {target_file} already exists")

    logger.info(f"Writing template {template_name} to {target_file}")

    with open(target_file, "w") as target_fp:
        target_fp.write(template.render(template_vars))

    return target_file


def init_api(
    target_dir: Path,
    tesseract_name: str,
    recipe: str = "base",
) -> Path:
    """Create a new empty Tesseract API module at the target location."""
    from tesseract_core import __version__ as tesseract_version

    template_vars = {
        "version": tesseract_version,
        "timestamp": datetime.datetime.now().isoformat(),
        "name": tesseract_name,
    }

    # If target dir does not exist, create it
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    _write_template_file(
        "tesseract_api.py", target_dir, template_vars, recipe=Path(recipe)
    )
    _write_template_file(
        "tesseract_config.yaml", target_dir, template_vars, recipe=Path(recipe)
    )
    _write_template_file(
        "tesseract_requirements.txt", target_dir, template_vars, recipe=Path(recipe)
    )

    return target_dir / "tesseract_api.py"


def build_tesseract(
    src_dir: Path,
    image_tag: str | None,
    build_dir: Path | None = None,
    inject_ssh: bool = False,
    config_override: tuple[tuple[list[str], str], ...] = (),
    keep_build_cache: bool = False,
    generate_only: bool = False,
) -> docker.models.images.Image | Path:
    """Build a new Tesseract from a context directory.

    Args:
        src_dir: path to the Tesseract project directory, where the
          `tesseract_api.py` and `tesseract_config.yaml` files
          are located.
        image_tag: name to be used as a tag for the Tesseract image.
        build_dir: directory to be used to store the build context.
          If not provided, a temporary directory will be created.
        inject_ssh: whether or not to forward SSH agent when building the image.
        config_override: overrides for configuration options in the Tesseract.
        keep_build_cache: whether or not to keep the Docker build cache.
        generate_only: only generate the build context but do not build the image.

    Returns:
        docker.models.images.Image representing the built Tesseract image,
        or path to build directory if `generate_only` is True.
    """
    validate_tesseract_api(src_dir)
    config = get_config(src_dir)

    # Apply config overrides
    for path, value in config_override:
        c = config
        for k in path[:-1]:
            c = getattr(c, k)
        setattr(c, path[-1], value)

    image_name = config.name
    if image_tag:
        image_name += f":{image_tag}"

    source_basename = Path(src_dir).name

    if build_dir is None:
        build_dir = Path(tempfile.mkdtemp(prefix=f"tesseract_build_{source_basename}"))
        keep_build_dir = True if generate_only else False
    else:
        build_dir = Path(build_dir)
        build_dir.mkdir(exist_ok=True)
        keep_build_dir = True

    dockerfile = create_dockerfile(config, use_ssh_mount=inject_ssh)

    try:
        out = build_image(
            src_dir,
            image_name,
            dockerfile,
            build_dir=build_dir,
            inject_ssh=inject_ssh,
            keep_build_cache=keep_build_cache,
            generate_only=generate_only,
        )
    finally:
        if not keep_build_dir:
            try:
                rmtree(build_dir)
            except OSError as exc:
                # Permission denied or already removed
                logger.info(
                    f"Could not remove temporary build directory {build_dir}: {exc}"
                )
                pass

    if generate_only:
        return build_dir

    return out


def teardown(project_id: str) -> None:
    """Teardown Tesseract image(s) running in a Docker Compose project.

    Args:
        project_id: Docker Compose project ID to teardown.
    """
    if not project_id:
        raise ValueError("Docker Compose project ID is empty or None, cannot teardown")
    if not _docker_compose_project_exists(project_id):
        raise ValueError(
            f"A Docker Compose project with ID {project_id} cannot be found, use `docker compose ls` to find project ID"
        )

    if not _docker_compose_down(project_id):
        raise RuntimeError(
            f"Cannot teardown Docker Compose project with ID: {project_id}"
        )


def get_tesseract_containers() -> list[docker.models.containers.Container]:
    """Get Tesseract containers."""
    return list(
        filter(
            lambda container: _is_tesseract(container),
            docker.from_env().containers.list(),
        )
    )


def get_tesseract_images() -> list[docker.models.images.Image]:
    """Get Tesseract images."""
    return list(filter(lambda img: _is_tesseract(img), docker.from_env().images.list()))


def _get_docker_image(image_name: str) -> docker.models.images.Image:
    """Get Docker image object."""
    try:
        return docker.from_env().images.get(image_name)
    except docker.errors.ImageNotFound:
        logger.error(f"No Docker image found with name: {image_name}")
        return None


def _is_tesseract(
    docker_asset: docker.models.images.Image | docker.models.containers.Container,
) -> bool:
    """Check if an image is Tesseract."""
    if not any(
        "TESSERACT_NAME" in env_val for env_val in docker_asset.attrs["Config"]["Env"]
    ):
        return False
    return True


def serve(
    images: list[str | docker.models.images.Image],
    port: str = "",
    volumes: list[str] | None = None,
    gpus: list[str] | None = None,
) -> str:
    """Serve one or more Tesseract images.

    Start the Tesseracts listening on an available ports on the host.

    Args:
        images: a list of Tesseract image IDs as strings or `docker`'s
                Image object.
        port: port or port range to serve the tesseract on.
        volumes: list of paths to mount in the Tesseract container.
        gpus: IDs of host Nvidia GPUs to make available to the Tesseracts.

    Returns:
        A string representing the Tesseract Project ID.
    """
    if not images or not all(
        (isinstance(item, str) or isinstance(item, docker.models.images.Image))
        for item in images
    ):
        raise ValueError("One or more Tesseract image IDs must be provided")

    image_ids = []
    for image_ in images:
        if isinstance(image_, docker.models.images.Image):
            image = image_
        else:  # str
            image = _get_docker_image(image_)

        if not image:
            raise ValueError(f"Image ID {image_} is not a valid Docker image")
        if not _is_tesseract(image):
            raise ValueError(f"Input ID {image.id} is not a valid Tesseract")
        image_ids.append(image.id)

    template = _create_docker_compose_template(image_ids, port, volumes, gpus)
    compose_fname = _create_compose_fname()

    with tempfile.NamedTemporaryFile(
        mode="w+",
        prefix=compose_fname,
    ) as compose_file:
        compose_file.write(template)
        compose_file.flush()

        project_name = _create_compose_proj_id()
        if not _docker_compose_up(compose_file.name, project_name):
            raise RuntimeError("Cannot serve Tesseracts")
        return project_name


def _docker_compose_project_exists(project_id: str) -> bool:
    """Check if Docker Compose project exists."""
    try:
        result = subprocess.run(
            ["docker", "compose", "ls", "-a", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
        if not any(
            project["Name"] == project_id for project in json.loads(result.stdout)
        ):
            logger.error(f"Docker Compose project with ID {project_id} does not exist")
            return False
        return True
    except (subprocess.CalledProcessError, json.JSONDecodeError) as ex:
        logger.error(str(ex))
        return False


def _docker_compose_down(project_id: str) -> bool:
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


def _docker_compose_up(compose_fpath: str, project_name: str) -> bool:
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
        return True
    except subprocess.CalledProcessError as ex:
        logger.error(str(ex))
        logger.error(ex.stderr.decode())
        return False


def _create_docker_compose_template(
    image_ids: list[str],
    port: str = "",
    volumes: list[str] | None = None,
    gpus: list[str] | None = None,
) -> str:
    """Create Docker Compose template."""
    services = []
    if not port:
        port = "8000"
    else:
        port = f"{port}:8000"

    gpu_settings = None
    if gpus:
        if (len(gpus) == 1) and (gpus[0] == "all"):
            gpu_settings = "count: all"
        else:
            gpu_settings = f"device_ids: {gpus}"

    for image_id in image_ids:
        service = {
            "name": _create_compose_service_id(image_id),
            "image": image_id,
            "port": port,
            "volumes": volumes,
            "gpus": gpu_settings,
        }

        services.append(service)
    template = ENV.get_template("docker-compose.yml")
    return template.render(services=services)


def _create_compose_service_id(image_id: str) -> str:
    """Create Docker Compose service ID."""
    image_id = image_id.split(":")[0]
    return f"{image_id}-{_id_generator()}"


def _create_compose_proj_id() -> str:
    """Create Docker Compose project ID."""
    return f"tesseract-{_id_generator()}"


def _create_compose_fname() -> str:
    """Create Docker Compose project file name."""
    return f"docker-compose-{_id_generator()}.yml"


def _id_generator(
    size: int = 12, chars: Sequence[str] = string.ascii_lowercase + string.digits
) -> str:
    """Generate ID."""
    return "".join(random.choice(chars) for _ in range(size))


def _parse_volumes(options: list[str]) -> dict[str, dict[str, str]]:
    """Parses volume mount strings to dict accepted by docker SDK.

    Strings of the form 'source:target:(ro|rw)' are parsed to
    `{source: {'bind': target, 'mode': '(ro|rw)'}}`.
    """

    def _parse_option(option: str):
        args = option.split(":")
        if len(args) == 2:
            source, target = args
            mode = "ro"
        elif len(args) == 3:
            source, target, mode = args
        else:
            raise ValueError(
                f"Invalid mount volume specification {option} "
                "(must be `/path/to/source:/path/totarget:(ro|rw)`)",
            )
        # Docker doesn't like paths like ".", so we convert to absolute path here
        source = str(Path(source).resolve())
        return source, {"bind": target, "mode": mode}

    return dict(_parse_option(opt) for opt in options)


def run_tesseract(
    image: str | docker.models.images.Image,
    command: str,
    args: list[str],
    volumes: list[str] | None = None,
    gpus: list[int | str] | None = None,
) -> tuple[str, str]:
    """Start a Tesseract and execute a given command.

    Args:
        image: string or docker.models.images.Image object of the Tesseract to run.
        command: Tesseract command to run, e.g. apply.
        args: arguments for the command.
        volumes: list of paths to mount in the Tesseract container.
        gpus: list of GPUs, as indices or names, to passthrough the container.

    Returns:
        Tuple with the stdout and stderr of the Tesseract.
    """
    client = docker.from_env()

    # Args that require rw access to the mounted volume
    output_args = {"-o", "--output-path"}

    cmd = [command]
    current_cmd = None

    if volumes is None:
        parsed_volumes = {}
    else:
        parsed_volumes = _parse_volumes(volumes)

    if gpus is None:
        device_requests = None
    else:
        device_requests = [DeviceRequest(device_ids=gpus, capabilities=[["gpu"]])]

    for arg in args:
        if arg.startswith("-"):
            current_cmd = arg
            cmd.append(arg)
            continue

        # Mount local output directories into Docker container as a volume
        if current_cmd in output_args and "://" not in arg:
            if arg.startswith("@"):
                raise ValueError(
                    f"Output path {arg} cannot start with '@' (used only for input files)"
                )

            local_path = Path(arg).resolve()
            local_path.mkdir(parents=True, exist_ok=True)

            if not local_path.is_dir():
                raise RuntimeError(
                    f"Path {local_path} provided as output is not a directory"
                )

            path_in_container = "/mnt/output"
            arg = path_in_container

            # Bind-mount directory
            parsed_volumes[str(local_path)] = {"bind": path_in_container, "mode": "rw"}

        # Mount local input files marked by @ into Docker container as a volume
        elif arg.startswith("@") and "://" not in arg:
            local_path = Path(arg.lstrip("@")).resolve()

            if not local_path.is_file():
                raise RuntimeError(f"Path {local_path} provided as input is not a file")

            path_in_container = os.path.join("/mnt", f"payload{local_path.suffix}")
            arg = f"@{path_in_container}"

            # Bind-mount file
            parsed_volumes[str(local_path)] = {"bind": path_in_container, "mode": "ro"}

        current_cmd = None
        cmd.append(arg)

    # Run the container
    if isinstance(image, docker.models.images.Image):
        image_id = image.short_id
    else:
        image_id = image
    container = None
    try:
        container = client.containers.run(
            image=image_id,
            command=cmd,
            volumes=parsed_volumes,
            detach=True,
            device_requests=device_requests,
        )
        result = container.wait()
        stdout = container.logs(stdout=True, stderr=False)
        stderr = container.logs(stdout=False, stderr=True)

        exit_code = result["StatusCode"]
        if exit_code != 0:
            stderr = f"\n{stderr.decode('utf-8')}"
            raise docker.errors.ContainerError(
                container, exit_code, shlex.join(cmd), image_id, stderr
            )
    finally:
        if container is not None:
            container.remove(v=True, force=True)

    return stdout.decode("utf-8"), stderr.decode("utf-8")


def exec_tesseract(
    container_id: str,
    command: str,
    args: list[str],
) -> tuple[str, str]:
    """Execute a given command on an existing Tesseract container.

    See `run_tesseract` for the equivalent command that operates on Tesseract images.

    Args:
        container_id: id of the target Tesseract container.
        command: Tesseract command to run, e.g. apply.
        args: arguments for the command.

    Returns:
        Tuple with the stdout and stderr of the Tesseract.
    """
    try:
        result = subprocess.run(
            ["docker", "exec", container_id, "tesseract-runtime", command, *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr


def project_containers(
    project_id: str,
) -> list[docker.models.containers.Container]:
    """Find containers associated with a Docker Compose Project ID.

    Args:
        project_id: the Docker Compose project ID.

    Returns:
        A list of Docker Images.
    """
    client = docker.from_env()
    return list(filter(lambda x: project_id in x.name, client.containers.list()))
