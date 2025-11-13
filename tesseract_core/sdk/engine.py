# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine to power Tesseract commands."""

import datetime
import linecache
import logging
import optparse
import os
import random
import socket
import tempfile
import time
from collections.abc import Callable, Collection, Sequence
from contextlib import closing
from pathlib import Path
from shutil import copy, copytree, rmtree
from typing import TYPE_CHECKING, Any, Literal

import requests
from jinja2 import Environment, PackageLoader, StrictUndefined

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

if TYPE_CHECKING:
    from pip._internal.index.package_finder import PackageFinder
    from pip._internal.network.session import PipSession

logger = logging.getLogger("tesseract")
docker_client = CLIDockerClient()

# Jinja2 Environment
ENV = Environment(
    loader=PackageLoader("tesseract_core.sdk", "templates"),
    undefined=StrictUndefined,
)


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
    session: "PipSession | None" = None,
    finder: "PackageFinder | None" = None,
    options: optparse.Values | None = None,
    constraint: bool = False,
) -> tuple[list[str], list[str]]:
    """Split local dependencies from remote ones in a pip-style requirements file.

    All CLI options that may be part of the given requiremets file are included in
    the remote dependencies.
    """
    # pip internals monkeypatch some typing behavior at import time, so we delay
    # these imports as much as possible to avoid conflicts.
    from pip._internal.network.session import PipSession
    from pip._internal.req.req_file import (
        RequirementsFileParser,
        get_line_parser,
        handle_line,
    )

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
    ├── .dockerignore
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
    src_dir = Path(src_dir)
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

    # Docker requires a .dockerignore file to be at the root of the build context
    dockerignore_path = runtime_source_dir / "meta" / ".dockerignore"
    if dockerignore_path.exists():
        copy(dockerignore_path, context_dir / ".dockerignore")

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
        tags = [f"{image_name}:{image_tag}"]
    else:
        tags = [
            f"{image_name}:{config.version}",
            f"{image_name}:latest",
        ]

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
            tags=tags,
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


def teardown(
    container_ids: Collection[str] | None = None, tear_all: bool = False
) -> None:
    """Teardown Tesseract container(s).

    Args:
        container_ids: List of container IDs to teardown.
        tear_all: boolean flag to teardown all Tesseract containers.
    """
    if tear_all:
        # Identify all Tesseract containers to tear down
        container_ids = set(
            container.id for container in docker_client.containers.list()
        )
        if not container_ids:
            logger.info("No Tesseract containers to teardown")
            return

    if not container_ids:
        raise ValueError("container_id must be provided if tear_all is False")

    if isinstance(container_ids, str):
        container_ids = [container_ids]

    def _is_container_id(container_id: str) -> bool:
        try:
            docker_client.containers.get(container_id)
            return True
        except ContainerError:
            return False

    for container_id in container_ids:
        if _is_container_id(container_id):
            container = docker_client.containers.get(container_id)
            container.remove(force=True)
            logger.info(
                f"Tesseract is shutdown for Docker container ID: {container_id}"
            )
        else:
            raise ValueError(
                f"A Docker container with ID {container_id} cannot be found, "
                "use `tesseract ps` to find container ID"
            )


def get_tesseract_containers() -> list[Container]:
    """Get Tesseract containers."""
    return docker_client.containers.list()


def get_tesseract_images() -> list[Image]:
    """Get Tesseract images."""
    return docker_client.images.list()


def serve(
    image_name: str,
    *,
    host_ip: str = "127.0.0.1",
    port: str | None = None,
    network: str | None = None,
    network_alias: str | None = None,
    volumes: list[str] | None = None,
    environment: dict[str, str] | None = None,
    gpus: list[str] | None = None,
    debug: bool = False,
    num_workers: int = 1,
    user: str | None = None,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    output_format: Literal["json", "json+base64", "json+binref"] | None = None,
) -> tuple:
    """Serve one or more Tesseract images.

    Start the Tesseracts listening on an available ports on the host.

    Args:
        image_name: Tesseract image name to serve.
        host_ip: IP address to bind the Tesseracts to.
        port: port or port range to serve each Tesseract on.
        network: name of the network the Tesseract will be attached to.
        network_alias: alias to use for the Tesseract within the network.
        volumes: list of paths to mount in the Tesseract container.
        environment: dictionary of environment variables to pass to the Tesseract.
        gpus: IDs of host Nvidia GPUs to make available to the Tesseracts.
        debug: Enable debug mode. This will propagate full tracebacks to the client
            and start a debugpy server in the Tesseract.
            WARNING: This may expose sensitive information, use with caution (and never in production).
        num_workers: number of workers to use for serving the Tesseracts.
        user: user to run the Tesseracts as, e.g. '1000' or '1000:1000' (uid:gid).
              Defaults to the current user.
        input_path: Input path to read input files from, such as local directory or S3 URI.
        output_path: Output path to write output files to, such as local directory or S3 URI.
        output_format: Output format to use for the results.

    Returns:
        A tuple of the Tesseract container name and the port it is serving on.
    """
    if not image_name or not isinstance(image_name, str):
        raise ValueError("Tesseract image name must be provided")

    if output_format == "json+binref" and output_path is None:
        logger.warning(
            "Consider specifying --output-path when using the 'json+binref' output format "
            "to easily retrieve .bin files."
        )

    image = docker_client.images.get(image_name)

    if not image:
        raise ValueError(f"Image ID {image_name} is not a valid Docker image")

    if user is None:
        # Use the current user if not specified
        user = f"{os.getuid()}:{os.getgid()}" if os.name != "nt" else None

    parsed_volumes, volume_environment = _prepare_and_validate_volumes(
        volume_specs=volumes,
        input_path=input_path,
        output_path=output_path,
    )

    if environment is None:
        environment = {}
    environment.update(volume_environment)

    if output_format:
        environment["TESSERACT_OUTPUT_FORMAT"] = output_format

    args = []
    container_api_port = "8000"
    container_debugpy_port = "5678"

    args.extend(["--port", container_api_port])

    if not port:
        port = str(get_free_port())
    else:
        # Convert port ranges to fixed ports
        if "-" in port:
            port_start, port_end = port.split("-")
            port = str(get_free_port(within_range=(int(port_start), int(port_end))))

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
        environment["TESSERACT_DEBUG"] = "1"

    extra_args = [
        "--restart",
        "unless-stopped",
    ]

    if is_podman():
        # This ensures podman behaves like Docker in terms of user namespaces
        # and allows the container to run with the same user ID as the host.
        extra_args.extend(["--userns", "keep-id"])

    if network_alias is not None:
        if network is None:
            raise ValueError("Network must be specified if network_alias is provided")
        extra_args.extend(["--network-alias", network_alias])

    container = docker_client.containers.run(
        image=image_name,
        command=["serve", *args],
        device_requests=gpus,
        ports=port_mappings,
        network=network,
        detach=True,
        volumes=parsed_volumes,
        user=user,
        environment=environment,
        extra_args=extra_args,
    )
    assert isinstance(container, Container)

    logger.info("Waiting for Tesseract to start...")
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

        container_status = docker_client.containers.get(container.id).status

        if timeout < 0 or container_status != "running":
            try:
                container_logs = container.logs(stdout=True, stderr=True)
                logger.error(
                    f"Tesseract container {container.name} failed to start:\n{container_logs.decode()}"
                )
            except APIError as ex:
                logger.warning(
                    f"Failed to get logs for container {container.name}: {ex}"
                )
            try:
                container.stop()
            except APIError as ex:
                logger.warning(f"Failed to stop container {container.name}: {ex}")

            if timeout < 0:
                raise TimeoutError("Tesseract did not start in time")
            else:
                raise RuntimeError("Tesseract failed to start")

    logger.info(f"Serving Tesseract at http://{ping_ip}:{port}")
    logger.info(f"View Tesseract: http://{ping_ip}:{port}/docs")
    if debug:
        logger.info(f"Debugpy server listening at http://{ping_ip}:{debugpy_port}")

    return container.name, container


def _is_local_volume(volume: str) -> bool:
    """Check if a volume is a local path."""
    return "/" in volume or "." in volume


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

        if _is_local_volume(source):
            if not Path(source).exists():
                raise RuntimeError(
                    f"Source path {source} does not exist, "
                    "please provide a valid local path."
                )
            # Docker doesn't like paths like ".", so we convert to absolute path here
            source = str(Path(source).resolve())
        return source, {"bind": target, "mode": mode}

    volumes = {}
    for opt in options:
        source, opt_dict = _parse_option(opt)
        _check_duplicate_volume_source_path(source, volumes)
        volumes[source] = opt_dict
    return volumes


def _check_duplicate_volume_source_path(
    path: Path | str, volumes: dict[str, dict[str, str]]
) -> None:
    """Prevent duplicate source paths in volume mounts."""
    if str(path) in volumes:
        raise ValueError(
            f"Path {path} is already mounted as a volume, please provide a unique path."
        )


def _prepare_and_validate_volumes(
    volume_specs: list[str] | None = None,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    file_inputs: list[tuple[Path, str]] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Parse volumes, validate them, and generate associated env vars for the runtime.

    Args:
        volume_specs: List of volume mount specifications (e.g., ["src:dest:mode"]).
        input_path: Input path to mount.
        output_path: Output path to mount.
        file_inputs: List of (local_path, container_path) tuples for file inputs.

    Returns:
        Tuple of (volumes_dict, environment_dict) ready for Docker.
    """
    environment = {}

    if not volume_specs:
        volumes = {}
    else:
        volumes = _parse_volumes(volume_specs)

    if input_path:
        environment["TESSERACT_INPUT_PATH"] = "/tesseract/input_data"
        if "://" not in str(input_path):
            local_path = _resolve_file_path(input_path)
            _check_duplicate_volume_source_path(local_path, volumes)
            volumes[str(local_path)] = {
                "bind": "/tesseract/input_data",
                "mode": "ro",
            }

    if output_path:
        environment["TESSERACT_OUTPUT_PATH"] = "/tesseract/output_data"
        if "://" not in str(output_path):
            local_path = _resolve_file_path(output_path, make_dir=True)
            _check_duplicate_volume_source_path(local_path, volumes)
            volumes[str(local_path)] = {
                "bind": "/tesseract/output_data",
                "mode": "rw",
            }

    if file_inputs:
        for local_path, container_path in file_inputs:
            _check_duplicate_volume_source_path(local_path, volumes)
            volumes[str(local_path)] = {
                "bind": container_path,
                "mode": "ro",
            }

    return volumes, environment


def run_tesseract(
    image: str,
    command: str,
    args: list[str],
    volumes: list[str] | None = None,
    gpus: list[int | str] | None = None,
    ports: dict[str, str] | None = None,
    environment: dict[str, str] | None = None,
    network: str | None = None,
    user: str | None = None,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    output_format: Literal["json", "json+base64", "json+binref"] | None = None,
    output_file: str | None = None,
) -> tuple[str, str]:
    """Start a Tesseract and execute a given command.

    Args:
        image: string of the Tesseract to run.
        command: Tesseract command to run, e.g. `"apply"`.
        args: arguments for the command.
        volumes: list of paths to mount in the Tesseract container.
        gpus: list of GPUs, as indices or names, to passthrough the container.
        ports: dictionary of ports to bind to the host. Key is the host port,
            value is the container port.
        environment: list of environment variables to set in the container,
            in Docker format: key=value.
        network: name of the Docker network to connect the container to.
        user: user to run the Tesseract as, e.g. '1000' or '1000:1000' (uid:gid).
            Defaults to the current user.
        input_path: Input path to read input files from, such as local directory or S3 URI.
        output_path: Output path to write output files to, such as local directory or S3 URI.
        output_format: Format of the output.
        output_file: If specified, the output will be written to this file within output_path
            instead of stdout.

    Returns:
        Tuple with the stdout and stderr of the Tesseract.
    """
    if output_format == "json+binref" and output_path is None:
        logger.warning(
            "Consider specifying --output-path when using the 'json+binref' output format "
            "to easily retrieve .bin files."
        )

    if user is None:
        # Use the current user if not specified
        user = f"{os.getuid()}:{os.getgid()}" if os.name != "nt" else None

    file_inputs = []
    for arg in args:
        if arg.startswith("@") and "://" not in arg:
            local_path = Path(arg.lstrip("@")).resolve()

            if not local_path.is_file():
                raise RuntimeError(f"Path {local_path} provided as input is not a file")

            path_in_container = os.path.join(
                "/tesseract", f"payload{local_path.suffix}"
            )
            file_inputs.append((local_path, path_in_container))

    parsed_volumes, volume_environment = _prepare_and_validate_volumes(
        volume_specs=volumes,
        input_path=input_path,
        output_path=output_path,
        file_inputs=file_inputs,
    )

    if environment is None:
        environment = {}
    environment.update(volume_environment)

    if output_format:
        environment["TESSERACT_OUTPUT_FORMAT"] = output_format

    if output_file:
        environment["TESSERACT_OUTPUT_FILE"] = output_file

    cmd = []

    if command:
        cmd.append(command)

    file_input_map = {str(local): container for local, container in file_inputs}
    for arg in args:
        # Replace @local_path with @container_path
        if arg.startswith("@") and "://" not in arg:
            local_path_str = str(Path(arg.lstrip("@")).resolve())
            container_path = file_input_map[local_path_str]
            arg = f"@{container_path}"
        cmd.append(arg)

    extra_args = []
    if is_podman():
        extra_args.extend(["--userns", "keep-id"])

    # Run the container
    result = docker_client.containers.run(
        image=image,
        command=cmd,
        volumes=parsed_volumes,
        device_requests=gpus,
        environment=environment,
        network=network,
        ports=ports,
        detach=False,
        remove=True,
        stderr=True,
        user=user,
        extra_args=extra_args,
    )
    assert isinstance(result, tuple)
    stdout, stderr = result
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    return stdout, stderr


def _resolve_file_path(path: str | Path, make_dir: bool = False) -> Path:
    """Resolve a file path, creating the directory if necessary."""
    local_path = Path(path).resolve()
    if make_dir:
        local_path.mkdir(parents=True, exist_ok=True)
    if not local_path.is_dir():
        raise RuntimeError(f"Path {local_path} provided is not a directory")

    return local_path


def logs(container_id: str) -> str:
    """Get logs from a container.

    Args:
        container_id: the ID of the container.

    Returns:
        The logs of the container.
    """
    container = docker_client.containers.get(container_id)
    return container.logs().decode("utf-8")
