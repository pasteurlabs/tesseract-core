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
import tempfile
import time
from collections.abc import Callable, Collection, Sequence
from contextlib import closing
from importlib.metadata import requires
from pathlib import Path
from shutil import copy, copytree, rmtree
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse
from urllib.request import url2pathname

import requests
import yaml
from jinja2 import Environment, PackageLoader, StrictUndefined
from packaging.requirements import Requirement

from .api_parse import TesseractConfig, get_config, validate_tesseract_api
from .docker_client import (
    APIError,
    CLIDockerClient,
    Container,
    ContainerError,
    Image,
    NotFound,
    build_docker_image,
    is_podman,
)
from .exceptions import UserError

if TYPE_CHECKING:
    from pip._internal.index.package_finder import PackageFinder
    from pip._internal.network.session import PipSession

logger = logging.getLogger("tesseract")
docker_client = CLIDockerClient()

# Fixed port the API server binds *inside* the container when port-mapping is
# used (i.e. everything except host networking). The container has its own
# network namespace, so this need not be dynamic -- only the host-side port
# does. Keeping it fixed mirrors how debugpy is handled (fixed 5678 inside,
# dynamic host mapping) and decouples the container port from the host port.
CONTAINER_API_PORT = "8000"
# Fixed port the debugpy server binds inside the container (see runtime serve).
CONTAINER_DEBUGPY_PORT = "5678"

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
        elif _is_local_dependency(parsed_line.requirement):
            local_dependencies.append(line)
        else:
            remote_dependencies.append(line)
    return local_dependencies, remote_dependencies


# Prefixes that mark a requirement as a local filesystem path rather than a
# package name to resolve from an index.
_LOCAL_DEPENDENCY_PREFIXES = (".", "/", "file://")


def _is_local_dependency(spec: str) -> bool:
    """Return whether a requirement spec refers to a local filesystem path."""
    return spec.startswith(_LOCAL_DEPENDENCY_PREFIXES)


def _ignore_pycache(_: Any, names: list[str]) -> list[str]:
    """`copytree` ignore filter that drops ``__pycache__`` directories."""
    return ["__pycache__"] if "__pycache__" in names else []


def _split_local_dependency(line: str) -> tuple[str, str]:
    """Split a local dependency line into its filesystem path and extras suffix.

    A local requirement may carry an extras specifier, e.g. ``./mypkg[extra]``.
    The extras belong to the install spec, not to the path on disk, so they must
    be separated before the path is resolved and staged.

    A ``file://`` scheme is stripped so the returned path is a plain filesystem
    path (``file://`` URLs are always absolute).

    Returns a ``(path, extras)`` tuple where ``extras`` includes the surrounding
    brackets (e.g. ``"[extra]"``) or is empty if none are present.
    """
    # This pattern matches any non-empty string, so a match is always found.
    match = re.match(r"^(?P<path>.+?)(?P<extras>\[[^\]]*\])?\s*\Z", line.strip())
    path = match.group("path")
    if path.startswith("file://"):
        # `Path(...)` does not understand the `file://` scheme, so convert the
        # URL back to a native filesystem path (handles percent-encoding and an
        # optional `localhost` authority).
        path = url2pathname(urlparse(path).path)
    return path, match.group("extras") or ""


def _stage_local_dependency(
    line: str, src_dir: Path, local_requirements_path: Path
) -> str:
    """Copy a local dependency into the build context and return its install spec.

    The source path is resolved relative to ``src_dir`` (so ``.``/``..`` segments
    are collapsed) to derive a valid, unique destination name under
    ``local_requirements/``. Returns the install spec relative to the build
    working directory, with any extras suffix preserved.
    """
    path, extras = _split_local_dependency(line)
    resolved_src = (src_dir / path).resolve()

    if not resolved_src.exists():
        raise RuntimeError(
            f"local dependency not found: {path} (resolved to {resolved_src})"
        )

    # Derive a valid, unique destination name from the resolved path. Using the
    # raw path directly would break for lines like ``../..`` (whose ``.name`` is
    # ``..``, not a real directory name). The collision suffix uses the full
    # name so versioned names like ``pkg-1.0`` are not split on the dot.
    dest_name = resolved_src.name
    dest = local_requirements_path / dest_name
    counter = 1
    while dest.exists():
        dest_name = f"{resolved_src.name}_{counter}"
        dest = local_requirements_path / dest_name
        counter += 1

    if resolved_src.is_file():
        copy(resolved_src, dest)
    else:
        copytree(resolved_src, dest, ignore=_ignore_pycache)

    return f"./local_requirements/{dest_name}{extras}"


def get_runtime_dir() -> Path:
    """Get the source directory for the Tesseract runtime."""
    import tesseract_core

    return Path(tesseract_core.__file__).parent / "runtime"


def get_runtime_dependencies() -> list[str]:
    """Get the runtime dependencies from the installed tesseract-core package.

    This retrieves dependencies declared under the 'runtime' extra without
    requiring that extra to be installed.
    """
    deps = []
    for req_str in sorted(requires("tesseract-core") or []):
        req = Requirement(req_str)
        # Check if this requirement is for the 'runtime' extra
        if req.marker and req.marker.evaluate({"extra": "runtime"}):
            # Reconstruct the requirement string without the marker
            dep_str = req.name
            if req.extras:
                dep_str += f"[{','.join(sorted(req.extras))}]"
            if req.specifier:
                dep_str += str(req.specifier)
            deps.append(dep_str)
    return deps


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

    # Handle package_data paths that reference files outside the Tesseract directory
    # These need to be copied into the build context and their paths rewritten
    package_data_dir = context_dir / "__package_data__"
    resolved_package_data = []
    if user_config.build_config.package_data:
        target_paths = [t for _, t in user_config.build_config.package_data]
        duplicates = {t for t in target_paths if target_paths.count(t) > 1}
        if duplicates:
            raise RuntimeError(
                f"package_data has duplicate target path(s): {', '.join(sorted(duplicates))}"
            )

        for source_path, target_path in user_config.build_config.package_data:
            # Resolve the source path relative to the Tesseract directory
            resolved_source = (src_dir / source_path).resolve()

            # Check if the path goes outside the Tesseract directory
            if resolved_source.is_relative_to(src_dir.resolve()):
                # Path is within src_dir, use as-is
                resolved_package_data.append((source_path, target_path))
            else:
                # Path is outside src_dir, copy to __package_data__ directory
                if not resolved_source.exists():
                    raise RuntimeError(
                        f"package_data source file not found: {source_path} "
                        f"(resolved to {resolved_source})"
                    )

                # Create a unique name for the copied file/directory,
                # using an incrementing counter to avoid collisions
                dest_name = resolved_source.name
                dest_path = package_data_dir / dest_name
                counter = 1
                while dest_path.exists():
                    stem = resolved_source.stem
                    dest_name = f"{stem}_{counter}{resolved_source.suffix}"
                    dest_path = package_data_dir / dest_name
                    counter += 1

                package_data_dir.mkdir(parents=True, exist_ok=True)
                if resolved_source.is_file():
                    copy(resolved_source, dest_path)
                else:
                    copytree(resolved_source, dest_path)

                # Use the path relative to build context for Docker COPY
                resolved_package_data.append(
                    (f"../__package_data__/{dest_name}", target_path)
                )

    template_name = "Dockerfile.base"
    template = ENV.get_template(template_name)

    # Replace the package_data in config with resolved paths
    resolved_config = user_config.model_copy(deep=True)
    if resolved_package_data:
        resolved_config.build_config = resolved_config.build_config.model_copy(
            update={"package_data": tuple(resolved_package_data)}
        )

    template_values = {
        "tesseract_source_directory": "__tesseract_source__",
        "tesseract_runtime_location": "__tesseract_runtime__",
        "config": resolved_config,
        "use_ssh_mount": use_ssh_mount,
    }

    logger.debug(f"Generating Dockerfile from template: {template_name}")
    dockerfile_content = template.render(template_values)
    dockerfile_path = context_dir / "Dockerfile"

    logger.debug(f"Writing Dockerfile to {dockerfile_path}")

    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)

    template_dir = get_template_dir()

    extra_files = [template_dir / "entrypoint.sh"]

    requirement_config = user_config.build_config.requirements
    extra_files.append(template_dir / requirement_config._build_script)

    for path in extra_files:
        copy(path, context_dir / path.relative_to(template_dir))

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

        # Stage each local dependency into the build context and rewrite it to
        # point at the staged copy (preserving any extras suffix). The install
        # specs are written back into the requirements file so pip installs them
        # alongside the remote dependencies.
        staged_dependencies = [
            _stage_local_dependency(dependency, src_dir, local_requirements_path)
            for dependency in local_dependencies
        ]

        # We need to write a new requirements file in the build dir, where the
        # local dependencies are rewritten to their staged locations.
        requirements_file_path = (
            context_dir / "__tesseract_source__" / "tesseract_requirements.txt"
        )
        lines = remote_dependencies + staged_dependencies
        with requirements_file_path.open("w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

    elif requirement_config.provider == "conda":
        # The conda environment file may declare local-path pip dependencies via
        # a `pip:` sub-list (e.g. `- ./mypkg_src`). conda resolves those paths
        # relative to the environment file, but only the file itself is copied
        # into the build stage, not the surrounding Tesseract source. Stage each
        # local path into the build context and rewrite it to point at the
        # staged copy, mirroring the python-pip provider.
        env_file = src_dir / requirement_config._filename
        env_dest = context_dir / "__tesseract_source__" / requirement_config._filename
        if env_file.exists():
            with env_file.open(encoding="utf-8") as f:
                env_spec = yaml.safe_load(f) or {}

            for entry in env_spec.get("dependencies", []) or []:
                if not (isinstance(entry, dict) and "pip" in entry):
                    continue
                rewritten_pip = []
                for pip_dep in entry["pip"] or []:
                    if isinstance(pip_dep, str) and _is_local_dependency(
                        pip_dep.strip()
                    ):
                        rewritten_pip.append(
                            _stage_local_dependency(
                                pip_dep, src_dir, local_requirements_path
                            )
                        )
                    else:
                        rewritten_pip.append(pip_dep)
                entry["pip"] = rewritten_pip

            with env_dest.open("w", encoding="utf-8") as f:
                yaml.safe_dump(env_spec, f, sort_keys=False)

    runtime_source_dir = get_runtime_dir()
    copytree(
        runtime_source_dir,
        context_dir / "__tesseract_runtime__" / "tesseract_core" / "runtime",
        ignore=_ignore_pycache,
    )
    # Copy meta files (except Jinja templates, which we render)
    from tesseract_core import __version__ as tesseract_version

    for metafile in (runtime_source_dir / "meta").glob("*"):
        if metafile.suffix == ".jinja":
            # Render Jinja template
            target_name = metafile.stem  # Remove .jinja suffix
            template_content = metafile.read_text()
            from jinja2 import Template

            template = Template(template_content)
            rendered = template.render(
                runtime_dependencies=get_runtime_dependencies(),
                version=tesseract_version,
            )
            (context_dir / "__tesseract_runtime__" / target_name).write_text(rendered)
        else:
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
    template = ENV.get_template((recipe / template_name).as_posix())

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
    stream_logs: Callable[[str], Any] | bool = False,
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
        stream_logs: if True, stream build logs to stderr. If a callable is provided,
            it will be called with each log line.

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
            stream_logs=stream_logs,
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

    # Validate all container IDs exist before removing any
    containers = {
        # containers.get raises NotFound if any container ID is invalid, preventing partial teardown
        cid: docker_client.containers.get(cid)
        for cid in container_ids
    }

    for container_id, container in containers.items():
        container.remove(force=True)
        logger.info(f"Tesseract is shutdown for Docker container ID: {container_id}")


def get_tesseract_containers() -> list[Container]:
    """Get Tesseract containers."""
    return docker_client.containers.list()


def get_tesseract_images() -> list[Image]:
    """Get Tesseract images."""
    return docker_client.images.list()


# Built-in Docker/Podman networks that can/should not be created.
_BUILTIN_NETWORKS = {"host", "bridge", "none"}


def _ensure_network_exists(network: str) -> None:
    """Create the Docker network if it does not exist yet.

    Params:
        network: The network name to create.
    """
    if network in _BUILTIN_NETWORKS:
        return
    try:
        docker_client.networks.get(network)
    except NotFound:
        create_network = True
    else:
        create_network = False
    if create_network:
        logger.info("Network '%s' not found, creating it.", network)
        docker_client.networks.create(network)


class _PortInUseError(RuntimeError):
    """Container failed to start because its port was already bound.

    Signals that a fresh port should be picked and startup retried. Only raised
    when we chose the port ourselves; a user-supplied port is never retried.

    A port collision surfaces in one of two ways depending on network mode:
    - port-mapping mode: the Docker daemon fails to publish the host port and
      ``containers.run`` raises ``ContainerError`` ("port is already allocated").
    - host networking: the container binds the host port directly, so the
      failure appears in the container logs as uvicorn's "address already in
      use" and is detected in ``_wait_for_health``.
    """


# Substrings container runtimes use to report a host port already being taken.
_PORT_CONFLICT_MARKERS = ("address already in use", "port is already allocated")


def _is_port_conflict(stderr: str) -> bool:
    """Whether runtime stderr/logs indicate a host port collision."""
    lowered = stderr.lower()
    return any(marker in lowered for marker in _PORT_CONFLICT_MARKERS)


def _retry_or_raise_port_conflict(
    port: str, auto_port: bool, attempt: int, max_attempts: int
) -> None:
    """Decide whether a port collision should be retried.

    Returns normally if the caller should retry with a fresh port; raises
    otherwise. A user-supplied fixed port is never retried (we must not
    silently move the Tesseract elsewhere), and auto-selected ports raise once
    the attempt budget is exhausted.
    """
    if not auto_port:
        # User asked for this exact port; surface the collision as-is.
        raise _PortInUseError(f"Port {port} was already in use")
    if attempt + 1 >= max_attempts:
        raise RuntimeError(
            f"Failed to find a free port after {max_attempts} attempts"
        ) from None
    logger.info(f"Port {port} was taken, retrying with a new port...")


def _wait_for_health(
    container: Container, ping_ip: str, port: str, timeout: float = 30
) -> None:
    """Poll a container's /health endpoint until it responds 200 or timeout expires."""
    while True:
        try:
            response = requests.get(f"http://{ping_ip}:{port}/health")
        except requests.exceptions.ConnectionError:
            pass
        else:
            if response.status_code == 200:
                return

        time.sleep(0.1)
        timeout -= 0.1

        container_status = docker_client.containers.get(container.id).status

        if timeout < 0 or container_status != "running":
            logs_text = ""
            try:
                logs_text = container.logs(stdout=True, stderr=True).decode()
                logger.error(
                    f"Tesseract container {container.name} failed to start:\n{logs_text}"
                )
            except APIError as ex:
                logger.warning(
                    f"Failed to get logs for container {container.name}: {ex}"
                )
            try:
                container.stop()
            except APIError as ex:
                logger.warning(f"Failed to stop container {container.name}: {ex}")

            # A port collision is racy and worth retrying with a fresh port;
            # distinguish it from genuine startup failures so those still fail
            # fast.
            if _is_port_conflict(logs_text):
                raise _PortInUseError(f"Port {port} was already in use")

            if timeout < 0:
                raise TimeoutError("Tesseract did not start in time")
            else:
                raise RuntimeError("Tesseract failed to start")


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
    memory: str | None = None,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    output_format: Literal["json", "json+base64", "json+binref"] | None = None,
    docker_args: list[str] | None = None,
    runtime_config: dict[str, Any] | None = None,
    skip_health_check: bool = False,
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
        memory: Memory limit for the container (e.g., "512m", "2g"). Minimum allowed is 6m.
        input_path: Input path to read input files from, such as local directory or S3 URI.
        output_path: Output path to write output files to, such as local directory or S3 URI.
        output_format: Output format to use for the results.
        docker_args: Additional arguments to pass to the container runtime (e.g., Docker).
        runtime_config: Dictionary of runtime configuration options to pass to the Tesseract.
            These are converted to TESSERACT_* environment variables. For example,
            ``{"profiling": True}`` sets ``TESSERACT_PROFILING=1``.
        skip_health_check: If True, skip the startup health check poll. Useful for
            Tesseracts with slow initialization (e.g., Julia runtime startup, large
            model loading). The caller is responsible for ensuring readiness,
            e.g. by polling ``/health``, before calling other endpoints.

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

    # Convert runtime_config to TESSERACT_* environment variables
    if runtime_config is not None:
        for key, value in runtime_config.items():
            env_key = f"TESSERACT_{key.upper()}"
            if isinstance(value, bool):
                env_value = "1" if value else "0"
            else:
                env_value = str(value)
            environment[env_key] = env_value

    if output_format:
        environment["TESSERACT_OUTPUT_FORMAT"] = output_format

    # A port picked by get_free_port can be grabbed by another process between
    # our check and the container binding it (an unavoidable race, since the
    # port must be released before the container can bind it). When we choose
    # the port, retry a few times with a fresh one; a user-supplied fixed port
    # is honored as-is and never retried.
    if not port:
        auto_port = True

        def pick_port() -> str:
            return str(get_free_port())
    elif "-" in port:
        auto_port = True
        port_start, port_end = (int(p) for p in port.split("-"))

        def pick_port() -> str:
            return str(get_free_port(within_range=(port_start, port_end)))
    else:
        auto_port = False
        fixed_port = port

        def pick_port() -> str:
            return fixed_port

    max_attempts = 5 if auto_port else 1
    for attempt in range(max_attempts):
        # `port` is always the host-side port (what we publish and health-check).
        port = pick_port()

        # When using host network there is no port mapping: the container binds
        # the host's namespace directly, so the container port must equal the
        # host port. Otherwise the container binds a fixed internal port and we
        # map the (dynamic) host port onto it.
        if network == "host":
            ping_ip = "127.0.0.1"
            port_mappings = None
            container_api_port = port
        else:
            ping_ip = "127.0.0.1" if host_ip == "0.0.0.0" else host_ip
            container_api_port = CONTAINER_API_PORT
            port_mappings = {f"{host_ip}:{port}": container_api_port}

        args = ["--port", container_api_port]
        if num_workers > 1:
            args.extend(["--num-workers", str(num_workers)])
        # Always bind to all interfaces inside the container
        args.extend(["--host", "0.0.0.0"])

        if debug:
            # debugpy binds a fixed port inside the container; only its host
            # mapping is dynamic. Exclude the host API port so the two host
            # ports never collide (they share the same range).
            debugpy_port = str(get_free_port(exclude=(int(port),)))
            if port_mappings is not None:
                port_mappings[f"{host_ip}:{debugpy_port}"] = CONTAINER_DEBUGPY_PORT
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
                raise ValueError(
                    "Network must be specified if network_alias is provided"
                )
            extra_args.extend(["--network-alias", network_alias])

        if docker_args:
            extra_args.extend(docker_args)

        if network is not None:
            _ensure_network_exists(network)

        try:
            # In port-mapping mode a host-port collision fails here, when the
            # daemon tries to publish the port. In host-network mode it instead
            # surfaces from _wait_for_health (uvicorn's own bind fails).
            container = docker_client.containers.run(
                image=image_name,
                command=["serve", *args],
                device_requests=gpus,
                ports=port_mappings,
                network=network,
                detach=True,
                volumes=parsed_volumes,
                user=user,
                memory=memory,
                environment=environment,
                extra_args=extra_args,
            )
            assert isinstance(container, Container)

            if skip_health_check:
                logger.info("Skipping health check, Tesseract may not be ready yet")
                break

            logger.info("Waiting for Tesseract to start...")
            _wait_for_health(container, ping_ip, port)
        except ContainerError as ex:
            if not _is_port_conflict(ex.stderr.decode("utf-8", errors="ignore")):
                raise
            # Publish failed; no container was created, nothing to clean up.
            _retry_or_raise_port_conflict(port, auto_port, attempt, max_attempts)
            continue
        except _PortInUseError:
            container.remove(force=True)
            _retry_or_raise_port_conflict(port, auto_port, attempt, max_attempts)
            continue
        break

    logger.info(f"Serving Tesseract at http://{ping_ip}:{port}")
    logger.info(f"View Tesseract: http://{ping_ip}:{port}/docs")
    if debug:
        logger.info(f"Debugpy server listening at http://{ping_ip}:{debugpy_port}")

    return container.name, container


def _is_local_volume(volume: str) -> bool:
    """Check if a volume is a local path."""
    # Windows absolute paths like C:\foo
    if (
        len(volume) >= 3
        and volume[0].isalpha()
        and volume[1] == ":"
        and volume[2] in ("/", "\\")
    ):
        return True
    return "/" in volume or "." in volume


def _split_volume_spec(volume_spec: str) -> list[str]:
    r"""Split a volume spec string on colons, respecting Windows drive letters.

    E.g., ``C:\\foo:/bar:ro`` -> ``['C:\\foo', '/bar', 'ro']``
         ``/foo:/bar:ro``    -> ``['/foo', '/bar', 'ro']``
    """
    # Check for Windows drive letter prefix (e.g., "C:")
    if len(volume_spec) >= 2 and volume_spec[0].isalpha() and volume_spec[1] == ":":
        rest = volume_spec[2:]
        parts = rest.split(":")
        parts[0] = volume_spec[:2] + parts[0]
        return parts
    return volume_spec.split(":")


def _parse_volumes(volume_specs: list[str]) -> dict[str, dict[str, str]]:
    """Parses volume mount strings to dict accepted by docker SDK.

    Strings of the form 'source:target:(ro|rw)' are parsed to
    `{source: {'bind': target, 'mode': '(ro|rw)'}}`.
    """

    def _parse_volume_spec(volume_spec: str):
        args = _split_volume_spec(volume_spec)
        if len(args) == 2:
            source, target = args
            mode = "ro"
        elif len(args) == 3:
            source, target, mode = args
        else:
            raise ValueError(
                f"Invalid mount volume specification {volume_spec} "
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
    for spec in volume_specs:
        source, spec_dict = _parse_volume_spec(spec)
        _check_duplicate_volume_source_path(source, volumes)
        volumes[source] = spec_dict
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
    memory: str | None = None,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    output_format: Literal["json", "json+base64", "json+binref"] | None = None,
    output_file: str | None = None,
    docker_args: list[str] | None = None,
    debug: bool = False,
    stream_logs: bool | Callable[[str], None] = False,
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
        memory: Memory limit for the container (e.g., "512m", "2g"). Minimum allowed is 6m.
        input_path: Input path to read input files from, such as local directory or S3 URI.
        output_path: Output path to write output files to, such as local directory or S3 URI.
        output_format: Format of the output.
        output_file: If specified, the output will be written to this file within output_path
            instead of stdout.
        docker_args: Additional arguments to pass to the container runtime (e.g., Docker).
        debug: Enable debug mode. This starts a debugpy server in the Tesseract and
            blocks execution until a debugger attaches to the forwarded port.
        stream_logs: If set, stream logs in real-time. Can be True (streams to stderr)
            or a callable that accepts a string (e.g., logger.info).

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

            path_in_container = f"/tesseract/payload{local_path.suffix}"
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

    if docker_args:
        extra_args.extend(docker_args)

    if network is not None:
        _ensure_network_exists(network)

    if debug:
        environment["TESSERACT_DEBUG"] = "1"
        # `network="host"` binds the container's debugpy port directly on the host,
        # so no explicit port mapping is needed (and would actually be rejected).
        if network == "host":
            debugpy_port = "5678"
        else:
            debugpy_port = str(get_free_port())
            if ports is None:
                ports = {}
            ports[f"127.0.0.1:{debugpy_port}"] = "5678"
        logger.info(
            f"Debug mode enabled. Attach a debugger to localhost:{debugpy_port} "
            "to start execution (see the 'Debug mode' section of the docs for a "
            "sample VSCode launch config)."
        )

    # Run the container, optionally streaming stderr to the terminal
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
        memory=memory,
        extra_args=extra_args,
        stream_stderr=stream_logs,
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
