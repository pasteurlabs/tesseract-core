# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine to power Tesseract commands."""

import datetime
import linecache
import logging
import optparse
import os
import random
import re
import socket
import string
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path
from shutil import copy, copytree, rmtree
from typing import Any

import requests
from jinja2 import Environment, PackageLoader, StrictUndefined
from pip._internal.index.package_finder import PackageFinder
from pip._internal.network.session import PipSession
from pip._internal.req.req_file import (
    RequirementsFileParser,
    get_line_parser,
    handle_line,
)

from .api_parse import TesseractConfig, get_config, validate_tesseract_api
from .docker_client import (
    APIError,
    CLIDockerClient,
    Container,
    ContainerError,
    Image,
    build_docker_image,
    is_podman,
)
from .exceptions import UserError

logger = logging.getLogger("tesseract")
docker_client = CLIDockerClient()

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
            docker_client.info()
        except (APIError, RuntimeError) as ex:
            raise UserError(
                "Could not reach Docker daemon, check if it is running."
            ) from ex
        except FileNotFoundError as ex:
            raise UserError("Docker not found, check if it is installed.") from ex
        return func(*args, **kwargs)

    return wrapper_needs_docker


def get_free_port(
    within_range: tuple[int, int] = (49152, 65535),
    exclude: Sequence[int] = (),
) -> int:
    """Find a random free port to use for HTTP."""
    start, end = within_range
    if start < 0 or end > 65535 or start > end:
        raise ValueError("Invalid port range, must be between 0 and 65535")

    # Try random ports in the given range
    portlist = list(range(start, end))
    random.shuffle(portlist)
    for port in portlist:
        if port in exclude:
            continue
        # Check if the port is free
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                # Port is already in use
                continue
            else:
                return port
    raise RuntimeError(f"No free ports found in range {start}-{end}")


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


def get_runtime_dir() -> Path:
    """Get the source directory for the Tesseract runtime."""
    import tesseract_core

    return Path(tesseract_core.__file__).parent / "runtime"


def get_template_dir() -> Path:
    """Get the template directory for the Tesseract runtime."""
    import tesseract_core

    return Path(tesseract_core.__file__).parent / "sdk" / "templates"


def prepare_build_context(
    src_dir: str | Path,
    context_dir: str | Path,
    user_config: TesseractConfig,
    use_ssh_mount: bool = False,
) -> Path:
    """Populate the build context for a Tesseract.

    Generated folder structure:
    ├── Dockerfile
    ├── __tesseract_source__
    │   ├── tesseract_api.py
    │   ├── tesseract_config.yaml
    │   ├── tesseract_requirements.txt
    │   └── ... any other files in the source directory ...
    └── __tesseract_runtime__
        ├── pyproject.toml
        ├── ... any other files in the tesseract_core/runtime/meta directory ...
        └── tesseract_core
            └── runtime
                ├── __init__.py
                └── ... runtime module files ...

    Args:
        src_dir: The source directory where the Tesseract project is located.
        context_dir: The directory where the build context will be created.
        user_config: The Tesseract configuration object.
        use_ssh_mount: Whether to use SSH mount to install dependencies (prevents caching).

    Returns:
        The path to the build context directory.
    """
    context_dir = Path(context_dir)
    context_dir.mkdir(parents=True, exist_ok=True)

    copytree(src_dir, context_dir / "__tesseract_source__")

    template_name = "Dockerfile.base"
    template = ENV.get_template(template_name)

    template_values = {
        "tesseract_source_directory": "__tesseract_source__",
        "tesseract_runtime_location": "__tesseract_runtime__",
        "config": user_config,
        "use_ssh_mount": use_ssh_mount,
    }

    logger.debug(f"Generating Dockerfile from template: {template_name}")
    dockerfile_content = template.render(template_values)
    dockerfile_path = context_dir / "Dockerfile"

    logger.debug(f"Writing Dockerfile to {dockerfile_path}")

    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)

    template_dir = get_template_dir()

    requirement_config = user_config.build_config.requirements
    copy(
        template_dir / requirement_config._build_script,
        context_dir / "__tesseract_source__" / requirement_config._build_script,
    )

    # When building from a requirements.txt we support local dependencies.
    # We separate local dep. lines from the requirements.txt and copy the
    # corresponding files into the build directory.
    local_requirements_path = context_dir / "local_requirements"
    Path.mkdir(local_requirements_path, parents=True, exist_ok=True)

    if requirement_config.provider == "python-pip":
        reqstxt = src_dir / requirement_config._filename
        if reqstxt.exists():
            local_dependencies, remote_dependencies = parse_requirements(reqstxt)
        else:
            local_dependencies, remote_dependencies = [], []

        if local_dependencies:
            for dependency in local_dependencies:
                src = src_dir / dependency
                dest = context_dir / "local_requirements" / src.name
                if src.is_file():
                    copy(src, dest)
                else:
                    copytree(src, dest)

        # We need to write a new requirements file in the build dir, where we explicitly
        # removed the local dependencies
        requirements_file_path = (
            context_dir / "__tesseract_source__" / "tesseract_requirements.txt"
        )
        with requirements_file_path.open("w", encoding="utf-8") as f:
            for dependency in remote_dependencies:
                f.write(f"{dependency}\n")

    def _ignore_pycache(_: Any, names: list[str]) -> list[str]:
        ignore = []
        if "__pycache__" in names:
            ignore.append("__pycache__")
        return ignore

    runtime_source_dir = get_runtime_dir()
    copytree(
        runtime_source_dir,
        context_dir / "__tesseract_runtime__" / "tesseract_core" / "runtime",
        ignore=_ignore_pycache,
    )
    for metafile in (runtime_source_dir / "meta").glob("*"):
        copy(metafile, context_dir / "__tesseract_runtime__")

    return context_dir


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
    src_dir: str | Path,
    image_tag: str | None,
    build_dir: Path | None = None,
    inject_ssh: bool = False,
    config_override: dict[tuple[str, ...], Any] | None = None,
    generate_only: bool = False,
) -> Image | Path:
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
        generate_only: only generate the build context but do not build the image.

    Returns:
        Image object representing the built Tesseract image,
        or path to build directory if `generate_only` is True.
    """
    src_dir = Path(src_dir)

    validate_tesseract_api(src_dir)
    config = get_config(src_dir)

    # Apply config overrides
    if config_override is not None:
        for path, value in config_override.items():
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

    context_dir = prepare_build_context(
        src_dir, build_dir, config, use_ssh_mount=inject_ssh
    )

    if generate_only:
        logger.info(f"Build directory generated at {build_dir}, skipping build")
    else:
        logger.info("Building image ...")

    try:
        image = build_docker_image(
            path=context_dir.as_posix(),
            tag=image_name,
            dockerfile=context_dir / "Dockerfile",
            inject_ssh=inject_ssh,
            print_and_exit=generate_only,
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

    if generate_only:
        return build_dir

    logger.debug("Build successful")
    assert image is not None
    return image


def teardown(project_ids: Sequence[str] | None = None, tear_all: bool = False) -> None:
    """Teardown Tesseract image(s) running in a Docker Compose project or standalone containers.

    Args:
        project_ids: List of Docker Compose project IDs to teardown.
        tear_all: boolean flag to teardown all Tesseract projects.
    """
    if tear_all:
        # Identify all Tesseract projects to tear down, whether they're running in
        # Docker Compose or as standalone containers
        compose_projects = docker_client.compose.list()
        compose_containers = set()
        for project_containers in compose_projects.values():
            compose_containers.update(project_containers)
        other_containers = set(
            container.id
            for container in docker_client.containers.list()
            if container.id not in compose_containers
        )
        project_ids = [
            *compose_projects.keys(),
            *other_containers,
        ]
        if not project_ids:
            logger.info("No Tesseract projects to teardown")
            return

    if not project_ids:
        raise ValueError("project_ids must be provided if tear_all is False")

    if isinstance(project_ids, str):
        project_ids = [project_ids]

    def _is_container_id(project_id: str) -> bool:
        try:
            docker_client.containers.get(project_id)
            return True
        except ContainerError:
            return False

    for project_id in project_ids:
        if docker_client.compose.exists(project_id):
            if not docker_client.compose.down(project_id):
                raise RuntimeError(
                    f"Cannot teardown Docker Compose project with ID: {project_id}"
                )
            logger.info(
                f"Tesseracts are shutdown for Docker Compose project ID: {project_id}"
            )
        elif _is_container_id(project_id):
            container = docker_client.containers.get(project_id)
            container.remove(force=True)
            logger.info(f"Tesseract is shutdown for Docker container ID: {project_id}")
        else:
            raise ValueError(
                f"A Docker Compose project with ID {project_id} cannot be found, "
                "use `tesseract ps` to find project ID"
            )


def get_tesseract_containers() -> list[Container]:
    """Get Tesseract containers."""
    return docker_client.containers.list()


def get_tesseract_images() -> list[Image]:
    """Get Tesseract images."""
    return docker_client.images.list()


def get_project_containers(project_id: str) -> list[Container]:
    """Get containers for a given Tesseract project ID."""
    if docker_client.compose.exists(project_id):
        containers = docker_client.compose.list()[project_id]
        return [docker_client.containers.get(c) for c in containers]

    try:
        container = docker_client.containers.get(project_id)
        return [container]
    except ContainerError as exc:
        raise ValueError(
            f"A Tesseract project with ID {project_id} cannot be found, "
            "use `tesseract ps` to find project ID"
        ) from exc


def serve(
    images: list[str],
    host_ip: str = "127.0.0.1",
    ports: list[str] | None = None,
    volumes: list[str] | None = None,
    environment: dict[str, str] | None = None,
    gpus: list[str] | None = None,
    debug: bool = False,
    num_workers: int = 1,
    no_compose: bool = False,
    service_names: list[str] | None = None,
    user: str | None = None,
    input_path: str | Path | None = None,
) -> str:
    """Serve one or more Tesseract images.

    Start the Tesseracts listening on an available ports on the host.

    Args:
        images: a list of Tesseract image IDs as strings.
        host_ip: IP address to bind the Tesseracts to.
        ports: port or port range to serve each Tesseract on.
        volumes: list of paths to mount in the Tesseract container.
        environment: dictionary of environment variables to pass to the Tesseract.
        gpus: IDs of host Nvidia GPUs to make available to the Tesseracts.
        debug: Enable debug mode. This will propagate full tracebacks to the client
            and start a debugpy server in the Tesseract.
            WARNING: This may expose sensitive information, use with caution (and never in production).
        num_workers: number of workers to use for serving the Tesseracts.
        no_compose: if True, do not use Docker Compose to serve the Tesseracts.
        service_names: list of service names under which to expose each Tesseract container on the shared network.
        user: user to run the Tesseracts as, e.g. '1000' or '1000:1000' (uid:gid).
              Defaults to the current user.
        input_path: Input path to read input files from, such as local directory or S3 URI.

    Returns:
        A string representing the Tesseract project ID.
    """
    if not images or not all(isinstance(item, str) for item in images):
        raise ValueError("One or more Tesseract image IDs must be provided")

    image_ids = []
    for image_ in images:
        image = docker_client.images.get(image_)

        if not image:
            raise ValueError(f"Image ID {image_} is not a valid Docker image")
        image_ids.append(image.id)

    if ports is not None and len(ports) != len(image_ids):
        raise ValueError(
            f"Number of ports ({len(ports)}) must match number of images ({len(image_ids)})"
        )

    if service_names is not None:
        if len(service_names) != len(image_ids):
            raise ValueError(
                f"Number of service names ({len(service_names)}) must match number of images ({len(image_ids)})"
            )
        _validate_service_names(service_names)

    if user is None:
        # Use the current user if not specified
        user = f"{os.getuid()}:{os.getgid()}" if os.name != "nt" else None

    if input_path:
        if environment is None:
            environment = {}
        environment["TESSERACT_INPUT_PATH"] = "/tesseract/input_data"
        if volumes is None:
            volumes = []
        if "://" not in input_path:
            input_path = Path(input_path).resolve()
            volumes.append(f"{input_path}:/tesseract/input_data:ro")

    if no_compose:
        if len(images) > 1:
            raise ValueError(
                "Docker Compose is required to serve multiple Tesseracts. "
                f"Currently attempting to serve `{len(images)}` Tesseracts."
            )
        if service_names is not None:
            raise ValueError(
                "Tesseract service names can only be set when using Docker Compose."
            )
        args = []
        container_api_port = "8000"
        container_debugpy_port = "5678"

        args.extend(["--port", container_api_port])

        if ports:
            port = ports[0]
        else:
            port = str(get_free_port())

        if num_workers > 1:
            args.extend(["--num-workers", str(num_workers)])

        # Always bind to all interfaces inside the container
        args.extend(["--host", "0.0.0.0"])

        if host_ip == "0.0.0.0":
            ping_ip = "127.0.0.1"
        else:
            ping_ip = host_ip

        port_mappings = {f"{host_ip}:{port}": container_api_port}
        if debug:
            debugpy_port = str(get_free_port())
            port_mappings[f"{host_ip}:{debugpy_port}"] = container_debugpy_port

        logger.info(f"Serving Tesseract at http://{ping_ip}:{port}")
        logger.info(f"View Tesseract: http://{ping_ip}:{port}/docs")
        if debug:
            logger.info(f"Debugpy server listening at http://{ping_ip}:{debugpy_port}")

        parsed_volumes = _parse_volumes(volumes) if volumes else {}

        extra_args = []
        if is_podman():
            extra_args.extend(["--userns", "keep-id"])

        container = docker_client.containers.run(
            image=image_ids[0],
            command=["serve", *args],
            device_requests=gpus,
            ports=port_mappings,
            detach=True,
            volumes=parsed_volumes,
            user=user,
            environment=environment,
            extra_args=extra_args,
        )
        # wait for server to start
        timeout = 30
        while True:
            try:
                response = requests.get(f"http://{ping_ip}:{port}/health")
            except requests.exceptions.ConnectionError:
                pass
            else:
                if response.status_code == 200:
                    break

            time.sleep(0.1)
            timeout -= 0.1

            if timeout < 0:
                raise TimeoutError("Tesseract did not start in time")

        return container.name

    if is_podman() and volumes:
        raise UserError(
            "Podman does not support volume mounts in Docker Compose. "
            "Please use --no-compose / no_compose=True instead."
        )

    template = _create_docker_compose_template(
        image_ids,
        host_ip,
        service_names,
        ports,
        volumes,
        environment,
        gpus,
        num_workers,
        debug=debug,
        user=user,
    )
    compose_fname = f"docker-compose-{_id_generator()}.yml"

    with tempfile.NamedTemporaryFile(
        mode="w+",
        prefix=compose_fname,
    ) as compose_file:
        compose_file.write(template)
        compose_file.flush()

        project_name = f"tesseract-{_id_generator()}"
        if not docker_client.compose.up(compose_file.name, project_name):
            raise RuntimeError("Cannot serve Tesseracts")
        return project_name


def _create_docker_compose_template(
    image_ids: list[str],
    host_ip: str = "127.0.0.1",
    service_names: list[str] | None = None,
    ports: list[str] | None = None,
    volumes: list[str] | None = None,
    environment: dict[str, str] | None = None,
    gpus: list[str] | None = None,
    num_workers: int = 1,
    debug: bool = False,
    user: str | None = None,
) -> str:
    """Create Docker Compose template."""
    services = []

    # Generate random service names for each image if not provided
    if service_names is None:
        service_names = []
        for image_id in image_ids:
            service_names.append(f"{image_id.split(':')[0]}-{_id_generator()}")

    # Get random unique ports for each image if not provided
    if ports is None:
        ports = []
        for _ in image_ids:
            taken_ports = [int(p) for p in ports if "-" not in p]
            ports.append(str(get_free_port(exclude=taken_ports)))

    # Get random unique ports for debugpy if debug mode is active
    debugpy_ports = []
    if debug:
        for _ in image_ids:
            taken_ports = [int(p) for p in ports if "-" not in p]
            debugpy_ports.append(str(get_free_port(exclude=taken_ports)))

    # Convert port ranges to fixed ports
    for i, port in enumerate(ports):
        if "-" in port:
            port_start, port_end = port.split("-")
            taken_ports = [int(p) for p in ports if "-" not in p]
            ports[i] = str(
                get_free_port(
                    within_range=(int(port_start), int(port_end)), exclude=taken_ports
                )
            )

    # Prepend host IP to ports
    ports = [f"{host_ip}:{port}" for port in ports]

    gpu_settings = None
    if gpus:
        if (len(gpus) == 1) and (gpus[0] == "all"):
            gpu_settings = "count: all"
        else:
            gpu_settings = f"device_ids: {gpus}"

    parsed_volumes = _parse_volumes(volumes) if volumes else {}

    for i, image_id in enumerate(image_ids):
        service = {
            "name": service_names[i],
            "user": user,
            "image": image_id,
            "port": f"{ports[i]}:8000",
            "volumes": parsed_volumes,
            "gpus": gpu_settings,
            "environment": {
                "TESSERACT_DEBUG": "1" if debug else "0",
                **(environment or {}),
            },
            "num_workers": num_workers,
            "debugpy_port": debugpy_ports[i] if debug else None,
        }

        services.append(service)

    docker_volumes = {}  # Dictionary of volume names mapped to whether or not they already exist
    if volumes:
        for volume in volumes:
            source = volume.split(":")[0]
            # Check if source exists to determine if specified volume is a docker volume
            if not Path(source).exists():
                # Check if volume exists
                if not docker_client.volumes.get(source):
                    if "/" not in source:
                        docker_volumes[source] = False
                    else:
                        raise ValueError(
                            f"Volume/Path {source} does not already exist, "
                            "and new volume cannot be created due to '/' in name."
                        )
                else:
                    # Docker volume is external
                    docker_volumes[source] = True

    template = ENV.get_template("docker-compose.yml")
    return template.render(services=services, docker_volumes=docker_volumes)


def _id_generator(
    size: int = 12, chars: Sequence[str] = string.ascii_lowercase + string.digits
) -> str:
    """Generate a random ID."""
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

        is_local_mount = "/" in source or Path(source).exists()
        if is_local_mount:
            # Docker doesn't like paths like ".", so we convert to absolute path here
            source = str(Path(source).resolve())
        return source, {"bind": target, "mode": mode}

    return dict(_parse_option(opt) for opt in options)


def _validate_service_names(service_names: list[str]) -> None:
    if len(set(service_names)) != len(service_names):
        raise ValueError("Service names must be unique")

    invalid_names = []
    for name in service_names:
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9-]*$", name):
            invalid_names.append(name)
        if name[-1] == "-":
            invalid_names.append(name)
    if invalid_names:
        raise ValueError(
            "Service names must contain only alphanumeric characters and hyphens, and must "
            f"not begin or end with a hyphen. Found invalid names: {invalid_names}."
        )


def run_tesseract(
    image: str,
    command: str,
    args: list[str],
    volumes: list[str] | None = None,
    gpus: list[int | str] | None = None,
    ports: dict[str, str] | None = None,
    environment: dict[str, str] | None = None,
    user: str | None = None,
    input_path: str | Path | None = None,
) -> tuple[str, str]:
    """Start a Tesseract and execute a given command.

    Args:
        image: string of the Tesseract to run.
        command: Tesseract command to run, e.g. apply.
        args: arguments for the command.
        volumes: list of paths to mount in the Tesseract container.
        gpus: list of GPUs, as indices or names, to passthrough the container.
        ports: dictionary of ports to bind to the host. Key is the host port,
            value is the container port.
        environment: list of environment variables to set in the container,
            in Docker format: key=value.
        user: user to run the Tesseract as, e.g. '1000' or '1000:1000' (uid:gid).
              Defaults to the current user.
        input_path: Input path to read input files from, such as local directory or S3 URI.

    Returns:
        Tuple with the stdout and stderr of the Tesseract.
    """
    # Args that require rw access to the mounted volume
    output_args = {"-o", "--output-path"}

    cmd = [command]
    current_cmd = None

    if environment is None:
        environment = {}

    if input_path:
        if volumes is None:
            volumes = []
        if "://" not in input_path:
            volumes.append(f"{input_path}:/tesseract/input_data")
        # environment["TESSERACT_LOCAL_INPUT_PATH"] = str(input_path)
        environment["TESSERACT_INPUT_PATH"] = "/tesseract/input_data"

    if not volumes:
        parsed_volumes = {}
    else:
        parsed_volumes = _parse_volumes(volumes)

    if user is None:
        # Use the current user if not specified
        user = f"{os.getuid()}:{os.getgid()}" if os.name != "nt" else None

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

            path_in_container = "/tesseract/output_data"
            arg = path_in_container

            # Bind-mount directory
            parsed_volumes[str(local_path)] = {"bind": path_in_container, "mode": "rw"}
            # environment["TESSERACT_LOCAL_OUTPUT_PATH"] = str(local_path)
            environment["TESSERACT_OUTPUT_PATH"] = path_in_container

        # Mount local input files marked by @ into Docker container as a volume
        elif arg.startswith("@") and "://" not in arg:
            local_path = Path(arg.lstrip("@")).resolve()

            if not local_path.is_file():
                raise RuntimeError(f"Path {local_path} provided as input is not a file")

            path_in_container = os.path.join(
                "/tesseract/input_data", f"payload{local_path.suffix}"
            )
            arg = f"@{path_in_container}"

            # Bind-mount file
            parsed_volumes[str(local_path)] = {
                "bind": path_in_container,
                "mode": "ro",
            }

        current_cmd = None
        cmd.append(arg)

    extra_args = []
    if is_podman():
        extra_args.extend(["--userns", "keep-id"])

    # Run the container
    stdout, stderr = docker_client.containers.run(
        image=image,
        command=cmd,
        volumes=parsed_volumes,
        device_requests=gpus,
        environment=environment,
        ports=ports,
        detach=False,
        remove=True,
        stderr=True,
        user=user,
        extra_args=extra_args,
    )
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    return stdout, stderr


def logs(container_id: str) -> str:
    """Get logs from a container.

    Args:
        container_id: the ID of the container.

    Returns:
        The logs of the container.
    """
    container = docker_client.containers.get(container_id)
    return container.logs().decode("utf-8")
