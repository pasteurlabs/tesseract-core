# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine to power Tesseract commands."""

import datetime
import ipaddress
import linecache
import logging
import optparse
import os
import re
import tempfile
from collections.abc import Callable, Collection
from importlib.metadata import requires
from pathlib import Path
from shutil import copy, copytree, rmtree
from typing import TYPE_CHECKING, Any, Literal, TypeAlias
from urllib.parse import urlparse
from urllib.request import url2pathname

import yaml
from jinja2 import Environment, PackageLoader, StrictUndefined
from packaging.requirements import Requirement
from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from .api_parse import (
    TesseractConfig,
    get_config,
    validate_tesseract_api,
)
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
from .serving import (
    DEFAULT_STARTUP_TIMEOUT,
    PortInUseError,
    get_free_port,
    is_port_conflict,
    retry_or_raise_port_conflict,
    wait_for_health_or_dispose,
)

if TYPE_CHECKING:
    from pip._internal.index.package_finder import PackageFinder
    from pip._internal.network.session import PipSession

logger = logging.getLogger("tesseract")
docker_client = CLIDockerClient()

# Output serialization formats. Single SDK-side source of truth (re-used across
# the SDK, e.g. sdk.tesseract). Mirrors runtime.file_interactions.supported_format_type
# but is defined here so the SDK does not eagerly import the (optional) runtime
# package; a test asserts the two stay in sync.
OutputFormat: TypeAlias = Literal[
    "json", "json+base64", "json+binref", "json+cuda_ipc", "json+nixl"
]

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


def _parse_secret_id(spec: str) -> str:
    """Extract the ``id`` from a BuildKit ``--secret`` spec (``id=name,env=VAR``)."""
    for part in spec.split(","):
        key, _, value = part.partition("=")
        if key.strip() == "id":
            return value.strip()
    raise ValueError(
        f"Invalid --secret spec {spec!r}: expected 'id=<name>,env=<VAR>' "
        "or 'id=<name>,src=<file>'."
    )


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
    secret_ids: list[str] | None = None,
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
        secret_ids: BuildKit secret ids to mount during the dependency install step
            (one per authenticated package index).

    Returns:
        The path to the build context directory.
    """
    secret_ids = list(secret_ids or [])
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
        "secret_ids": secret_ids,
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

    # Shared credential-setup script, sourced by both provider build scripts.
    # Always staged (and COPYed by the Dockerfile) so the build scripts can source
    # it unconditionally; it is a no-op when host_credentials.txt is empty.
    extra_files.append(template_dir / "setup_host_credentials.sh")

    for path in extra_files:
        copy(path, context_dir / path.relative_to(template_dir))

    # Write the declared host credentials (host + secret id + username, never
    # tokens) into a file both build scripts read. At install time the script
    # reads each token from its secret mount and assembles netrc + git-credential
    # entries. This is provider-agnostic: netrc/git auth applies to uv, pip, and
    # conda alike. The file is always written (empty when none are declared) so the
    # Dockerfile can COPY it unconditionally.
    credentials_file_path = context_dir / "host_credentials.txt"
    with credentials_file_path.open("w", encoding="utf-8") as f:
        for credential in user_config.build_config.host_credentials:
            # Tab-separated host, secret id, and username. None of these may
            # contain a tab; the allowlist validators guarantee that.
            f.write(
                f"{credential.host}\t{credential.secret_id}\t{credential.username}\n"
            )

    # When building from a requirements.txt we support local dependencies.
    # We separate local dep. lines from the requirements.txt and copy the
    # corresponding files into the build directory.
    local_requirements_path = context_dir / "local_requirements"
    Path.mkdir(local_requirements_path, parents=True, exist_ok=True)

    if requirement_config.provider == "uv-pip":
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
        # staged copy, mirroring the uv provider.
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


def _coerce_config_override(value: Any, annotation: Any, path: tuple[str, ...]) -> Any:
    """Coerce a config override value to the target field's declared type.

    CLI overrides arrive as raw strings (see ``_parse_config_override``). We first
    interpret the string as YAML so that structured values (lists, dicts, ints,
    bools) work, then fall back to the raw string if that fails to validate. This
    lets string fields like ``python_version=3.12`` work without the user having
    to quote the value, while ``3.10`` is preserved verbatim instead of being
    parsed as the float ``3.1``. Non-string values (e.g. passed via the Python
    SDK) are validated as-is.
    """
    if annotation is None:
        # Unknown field; let the assignment raise a validation error as usual.
        return value

    adapter = TypeAdapter(annotation)

    if not isinstance(value, str):
        return adapter.validate_python(value)

    # Try the YAML-interpreted value first so structured values (lists, dicts,
    # ints, bools) work, then fall back to the raw string so string fields accept
    # unquoted scalars. If both fail, report the error from the YAML-interpreted
    # value: it matches the user's evident intent, whereas the raw-string error
    # is often a misleading "not a valid <type>" for non-string fields.
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed = value

    try:
        return adapter.validate_python(parsed)
    except PydanticValidationError as parsed_error:
        try:
            return adapter.validate_python(value)
        except PydanticValidationError:
            keypath = ".".join(path)
            raise UserError(
                f'Invalid value "{value}" for config override "{keypath}": '
                f"{parsed_error}"
            ) from parsed_error


def build_tesseract(
    src_dir: str | Path,
    image_tag: str | None,
    build_dir: Path | None = None,
    inject_ssh: bool = False,
    secrets: list[str] | None = None,
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
        secrets: BuildKit secret specs (e.g. ``id=name,env=VAR`` or
          ``id=name,src=file``) to forward to the build for authenticated
          package indices. Credentials are mounted, never stored in a layer.
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
            for depth, k in enumerate(path):
                fields = getattr(type(c), "model_fields", None)
                if fields is None or k not in fields:
                    keypath = ".".join(path)
                    reached = ".".join(path[:depth]) or "(top level)"
                    valid = (
                        ", ".join(sorted(fields)) if fields is not None else "(none)"
                    )
                    raise UserError(
                        f'Invalid config override "{keypath}": '
                        f'"{".".join(path[: depth + 1])}" is not a known config '
                        f"option. Valid options under {reached}: {valid}."
                    )
                if depth == len(path) - 1:
                    annotation = fields[k].annotation
                    setattr(c, k, _coerce_config_override(value, annotation, path))
                else:
                    c = getattr(c, k)

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

    # Build secrets are supplied generically on the command line and mounted into
    # the dependency install step. Any secret referenced by a host credential must
    # be backed by a --secret; check that up front.
    secrets = list(secrets or [])
    provided_secret_ids = [_parse_secret_id(spec) for spec in secrets]
    required_secret_ids = [
        credential.secret_id for credential in config.build_config.host_credentials
    ]
    missing = sorted(set(required_secret_ids) - set(provided_secret_ids))
    if missing:
        raise ValueError(
            "Missing build secret(s) for authenticated host credentials: "
            f"{', '.join(missing)}. Provide them with "
            "`tesseract build --secret id=<name>,env=<VAR>` (or `,src=<file>`)."
        )

    context_dir = prepare_build_context(
        src_dir,
        build_dir,
        config,
        use_ssh_mount=inject_ssh,
        secret_ids=provided_secret_ids,
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
            secrets=secrets,
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


def _warn_if_debugger_unreachable(container: Container, expected_port: str) -> None:
    """Warn if the container did not bind the debug port we published a mapping to.

    Which port it binds is decided by the runtime inside the image, and one built
    before the port was configurable ignores it and uses the default, leaving the
    mapping published with nothing behind it. Nothing else fails -- the Tesseract
    is healthy and only the debugger is unreachable -- so it would otherwise look
    like it worked.

    A warning rather than an error, because this reads a log line the runtime
    prints: reformatting it there would make this miss and condemn a working
    setup. Serving is still useful either way.
    """
    logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
    match = re.search(r"Debugger listening on \S+?:(\d+)", logs)
    if match is not None and match.group(1) == expected_port:
        return
    bound = f"port {match.group(1)}" if match else "an unknown port"
    logger.warning(
        f"Tesseract is debugging on {bound}, not the requested {expected_port}, "
        "so no debugger can attach. Rebuild it to choose the port."
    )


def _get_runtime_setting(environment: dict[str, str], setting: str) -> str | None:
    """Read a runtime setting from an environment, under either name it takes.

    Typer binds every config option to ``TESSERACT_RUNTIME_*`` as well, and that
    takes precedence over the ``TESSERACT_*`` the config itself reads.
    """
    return environment.get(f"TESSERACT_RUNTIME_{setting}") or environment.get(
        f"TESSERACT_{setting}"
    )


def _is_loopback(host: str) -> bool:
    """Whether an address only accepts connections from the same host."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _resolve_container_debug_address(
    requested_port: str | None,
    requested_host: str | None,
    *,
    port_mapped: bool,
    reserved_ports: Collection[str],
) -> tuple[dict[str, str], str]:
    """Settle the debugpy address for a container, without touching the caller's.

    A published mapping sends a host port to a port on the container, so the port
    debugpy binds and the container side of that mapping must agree. Nothing is
    special about the default, so a chosen port is honoured and published against.
    Refused only where it cannot work: a port already taken inside the container,
    or, under port mapping, a loopback host -- publishing reaches the container
    over its network interface, so it cannot see a debugger that only accepts
    connections from inside. Any routable address is fine, including the
    container's own.

    Returns:
        Environment entries to apply, and the port debugpy will bind, which the
        caller must publish against.
    """
    port = requested_port or CONTAINER_DEBUGPY_PORT

    if port in reserved_ports:
        raise UserError(
            f"Port {port} is already in use inside the container. Choose another "
            f"debug port, or leave it unset to use {CONTAINER_DEBUGPY_PORT}."
        )

    if port_mapped and requested_host is not None and _is_loopback(requested_host):
        raise UserError(
            f"TESSERACT_DEBUGPY_HOST={requested_host} will not work in a "
            "container: its published port reaches the container from outside, "
            "which an address that only accepts local connections rejects. Leave "
            "it unset."
        )

    # Written under both names, so a value inherited under the other cannot win.
    updates = {f"TESSERACT{p}_DEBUGPY_PORT": port for p in ("", "_RUNTIME")}
    if port_mapped:
        # All interfaces unless asked for something specific, which the runtime
        # would otherwise default to loopback and be unreachable.
        host = requested_host or "0.0.0.0"
        updates.update({f"TESSERACT{p}_DEBUGPY_HOST": host for p in ("", "_RUNTIME")})
    return updates, port


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
    output_format: OutputFormat | None = None,
    docker_args: list[str] | None = None,
    runtime_config: dict[str, Any] | None = None,
    skip_health_check: bool = False,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
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
        startup_timeout: How long to wait for the Tesseract to answer a health
            check, in seconds. Raise it for one that is slow to initialize, in
            preference to skipping the check altogether.

    Returns:
        A tuple of the Tesseract container name and the port it is serving on.
    """
    if not image_name or not isinstance(image_name, str):
        raise ValueError("Tesseract image name must be provided")

    if output_format == "json+binref" and output_path is None:
        raise UserError(
            "The 'json+binref' output format writes array buffers to .bin files, "
            "which are lost when the container is torn down unless an output path "
            "is set. Specify one with --output-path (or output_path=...)."
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

    # Read after runtime_config lands in the environment, which is how the SDK
    # passes it. Only a port the caller asked for needs checking afterwards; the
    # default works whatever the image was built with.
    requested_debugpy_port = _get_runtime_setting(environment, "DEBUGPY_PORT")

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
            environment["TESSERACT_DEBUG"] = "1"
            debug_updates, container_debugpy_port = _resolve_container_debug_address(
                requested_debugpy_port,
                _get_runtime_setting(environment, "DEBUGPY_HOST"),
                port_mapped=port_mappings is not None,
                reserved_ports={container_api_port},
            )
            environment.update(debug_updates)
            # Only the host side of the debugger's mapping is dynamic. Exclude
            # the host API port so the two host ports never collide (they share
            # the same range).
            debugpy_port = str(get_free_port(exclude=(int(port),)))
            if port_mappings is not None:
                port_mappings[f"{host_ip}:{debugpy_port}"] = container_debugpy_port
            else:
                # Host networking: the container binds the host's namespace
                # directly, so there is nothing to map and the port it binds is
                # the one to attach to.
                debugpy_port = container_debugpy_port

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

        # CUDA IPC needs a GPU and a shared IPC namespace between host and
        # container. Wire both up whenever the experimental flag is set (the
        # only reason to enable it is IPC).
        cuda_ipc_enabled = environment.get(
            "TESSERACT_ENABLE_EXPERIMENTAL_CUDA_IPC"
        ) in ("1", "true", "True")

        if cuda_ipc_enabled:
            if not gpus:
                raise ValueError(
                    "enable_experimental_cuda_ipc requires GPU access, but no GPUs "
                    "were requested. Pass gpus=['all'] or specific GPU IDs."
                )
            extra_args.extend(["--ipc=host"])
        elif output_format == "json+cuda_ipc":
            raise ValueError(
                "The 'json+cuda_ipc' output format is experimental and must be "
                "explicitly enabled. Pass "
                "runtime_config={'enable_experimental_cuda_ipc': True}."
            )

        # NIXL needs a GPU, a shared IPC namespace (its same-host backend rides
        # CUDA IPC), and host networking (its UCX agent advertises its own address
        # to the client, which is unreachable across a container network
        # namespace). Wire all three up when the experimental flag is set.
        nixl_enabled = environment.get("TESSERACT_ENABLE_EXPERIMENTAL_CUDA_NIXL") in (
            "1",
            "true",
            "True",
        )
        if nixl_enabled:
            if not gpus:
                raise ValueError(
                    "enable_experimental_cuda_nixl requires GPU access, but no GPUs "
                    "were requested. Pass gpus=['all'] or specific GPU IDs."
                )
            if "--ipc=host" not in extra_args:
                extra_args.append("--ipc=host")
            extra_args.extend(["--network=host"])
        elif output_format == "json+nixl":
            raise ValueError(
                "The 'json+nixl' output format is experimental and must be "
                "explicitly enabled. Pass "
                "runtime_config={'enable_experimental_cuda_nixl': True}."
            )

        if network is not None:
            _ensure_network_exists(network)

        try:
            # In port-mapping mode a host-port collision fails here, when the
            # daemon tries to publish the port. In host-network mode it instead
            # surfaces from wait_for_health_or_dispose (uvicorn's own bind fails).
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
            wait_for_health_or_dispose(container, ping_ip, port, startup_timeout)
        except ContainerError as ex:
            if not is_port_conflict(ex.stderr.decode("utf-8", errors="ignore")):
                raise
            # Publish failed; no container was created, nothing to clean up.
            retry_or_raise_port_conflict(port, auto_port, attempt, max_attempts)
            continue
        except PortInUseError:
            container.remove(force=True)
            retry_or_raise_port_conflict(port, auto_port, attempt, max_attempts)
            continue
        break

    if debug and requested_debugpy_port and not skip_health_check:
        _warn_if_debugger_unreachable(container, container_debugpy_port)

    logger.info(f"Serving Tesseract at http://{ping_ip}:{port}")
    logger.info(f"View Tesseract: http://{ping_ip}:{port}/docs")
    if debug:
        logger.info(
            f"Debug mode enabled. Attach a debugger to {ping_ip}:{debugpy_port}"
        )

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
    output_format: OutputFormat | None = None,
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
        raise UserError(
            "The 'json+binref' output format writes array buffers to .bin files, "
            "which are lost when the container is torn down unless an output path "
            "is set. Specify one with --output-path (or output_path=...)."
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
        requested_debugpy_port = _get_runtime_setting(environment, "DEBUGPY_PORT")
        environment["TESSERACT_DEBUG"] = "1"
        debug_updates, container_debugpy_port = _resolve_container_debug_address(
            requested_debugpy_port,
            _get_runtime_setting(environment, "DEBUGPY_HOST"),
            port_mapped=network != "host",
            reserved_ports=set(),
        )
        environment.update(debug_updates)
        if requested_debugpy_port:
            # This command blocks until a debugger attaches and never hands
            # control back, so unlike `serve` there is no point at which we could
            # check the port was honoured.
            logger.warning(
                "Cannot verify whether the debugpy port is configurable when "
                "using `tesseract run`. Configuration is not possible for some "
                "old Tesseracts. Rebuild the Tesseract if your debugger cannot "
                "attach."
            )
        # `network="host"` binds the container's debugpy port directly on the host,
        # so no explicit port mapping is needed (and would actually be rejected).
        if network == "host":
            debugpy_port = container_debugpy_port
        else:
            debugpy_port = str(get_free_port())
            if ports is None:
                ports = {}
            ports[f"127.0.0.1:{debugpy_port}"] = container_debugpy_port
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
