# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine to power Tesseract commands."""

import datetime
import linecache
import logging
import optparse
import os
import random
import shlex
import string
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from shutil import copy, copytree, rmtree
from typing import Any

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
    BuildError,
    CLIDockerClient,
    Container,
    ContainerError,
    Image,
    build_docker_image,
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
        return func(*args, **kwargs)

    return wrapper_needs_docker


def get_free_port() -> int:
    """Find a free port to use for HTTP."""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


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
    config_override: tuple[tuple[list[str], str], ...] = (),
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
    except BuildError as e:
        logger.warning("Build failed with logs:")
        for line in e.build_log:
            logger.warning(line)
        raise UserError("Image build failure. See above logs for details.") from e
    except APIError as e:
        raise UserError(f"Docker server error: {e}") from e
    except TypeError as e:
        raise UserError(f"Input error building Tesseract: {e}") from e
    else:
        if image is not None:
            logger.debug("Build successful")
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

    return image


def teardown(project_ids: Sequence[str] | None = None, tear_all: bool = False) -> None:
    """Teardown Tesseract image(s) running in a Docker Compose project.

    Args:
        project_ids: List of Docker Compose project IDs to teardown.
        tear_all: boolean flag to teardown all Tesseract projects.
    """
    if tear_all:
        # Get copy of keys to iterate over since the dictionary will change as we are
        # tearing down projects.
        project_ids = list(docker_client.compose.list())
        if not project_ids:
            logger.info("No Tesseract projects to teardown")
            return

    if not project_ids:
        raise ValueError("Docker Compose project ID is empty or None, cannot teardown")

    if isinstance(project_ids, str):
        # Single string project_id passes the typecheck but gets parsed one letter at a time
        # in the following for loop, so we need to wrap it in a list.
        project_ids = [project_ids]

    for project_id in project_ids:
        if not docker_client.compose.exists(project_id):
            raise ValueError(
                f"A Docker Compose project with ID {project_id} cannot be found, use `tesseract ps` to find project ID"
            )

        if not docker_client.compose.down(project_id):
            raise RuntimeError(
                f"Cannot teardown Docker Compose project with ID: {project_id}"
            )

        logger.info(
            f"Tesseracts are shutdown for Docker Compose project ID: {project_id}"
        )


def get_tesseract_containers() -> list[Container]:
    """Get Tesseract containers."""
    return docker_client.containers.list()


def get_tesseract_images() -> list[Image]:
    """Get Tesseract images."""
    return docker_client.images.list()


def serve(
    images: list[str],
    ports: list[str] | None = None,
    volumes: list[str] | None = None,
    gpus: list[str] | None = None,
    debug: bool = False,
    num_workers: int = 1,
) -> str:
    """Serve one or more Tesseract images.

    Start the Tesseracts listening on an available ports on the host.

    Args:
        images: a list of Tesseract image IDs as strings.
        ports: port or port range to serve each Tesseract on.
        volumes: list of paths to mount in the Tesseract container.
        gpus: IDs of host Nvidia GPUs to make available to the Tesseracts.
        debug: whether to enable debug mode.
        num_workers: number of workers to use for serving the Tesseracts.

    Returns:
        A string representing the Tesseract Project ID.
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

    template = _create_docker_compose_template(
        image_ids, ports, volumes, gpus, num_workers, debug
    )
    compose_fname = _create_compose_fname()

    with tempfile.NamedTemporaryFile(
        mode="w+",
        prefix=compose_fname,
    ) as compose_file:
        compose_file.write(template)
        compose_file.flush()

        project_name = _create_compose_proj_id()
        if not docker_client.compose.up(compose_file.name, project_name):
            raise RuntimeError("Cannot serve Tesseracts")
        return project_name


def _create_docker_compose_template(
    image_ids: list[str],
    ports: list[str] | None = None,
    volumes: list[str] | None = None,
    gpus: list[str] | None = None,
    num_workers: int = 1,
    debug: bool = False,
) -> str:
    """Create Docker Compose template."""
    services = []
    if ports is None:
        ports = [str(get_free_port()) for _ in range(len(image_ids))]

    gpu_settings = None
    if gpus:
        if (len(gpus) == 1) and (gpus[0] == "all"):
            gpu_settings = "count: all"
        else:
            gpu_settings = f"device_ids: {gpus}"

    for image_id, port in zip(image_ids, ports, strict=True):
        service = {
            "name": _create_compose_service_id(image_id),
            "image": image_id,
            "port": f"{port}:8000",
            "volumes": volumes,
            "gpus": gpu_settings,
            "environment": {
                "DEBUG": "1" if debug else "0",
            },
        }

        services.append(service)
    template = ENV.get_template("docker-compose.yml")
    return template.render(services=services, num_workers=num_workers)


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
    image: str,
    command: str,
    args: list[str],
    volumes: list[str] | None = None,
    gpus: list[int | str] | None = None,
) -> tuple[str, str]:
    """Start a Tesseract and execute a given command.

    Args:
        image: string of the Tesseract to run.
        command: Tesseract command to run, e.g. apply.
        args: arguments for the command.
        volumes: list of paths to mount in the Tesseract container.
        gpus: list of GPUs, as indices or names, to passthrough the container.

    Returns:
        Tuple with the stdout and stderr of the Tesseract.
    """
    # Args that require rw access to the mounted volume
    output_args = {"-o", "--output-path"}

    cmd = [command]
    current_cmd = None

    if volumes is None:
        parsed_volumes = {}
    else:
        parsed_volumes = _parse_volumes(volumes)

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
    image_id = image
    container = None
    try:
        container = docker_client.containers.run(
            image=image_id,
            command=cmd,
            volumes=parsed_volumes,
            detach=True,
            device_requests=gpus,
        )
        result = container.wait()
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8")
        exit_code = result["StatusCode"]
        if exit_code != 0:
            raise ContainerError(
                container, exit_code, shlex.join(cmd), image_id, stderr
            )
    finally:
        if container is not None:
            container.remove(v=True, force=True)

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
