#!/usr/bin/env python

# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import subprocess
import sys
import time
import webbrowser
from collections import defaultdict
from contextlib import nullcontext
from enum import Enum
from logging import getLogger
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, NoReturn

import click
import docker
import docker.errors
import docker.models
import docker.models.containers
import docker.models.images
import typer
from jinja2 import Environment, PackageLoader, StrictUndefined
from rich.console import Console as RichConsole
from rich.table import Table as RichTable

from . import engine
from .api_parse import (
    EXPECTED_OBJECTS,
    TesseractBuildConfig,
    TesseractConfig,
    get_non_base_fields_in_tesseract_config,
)
from .exceptions import UserError
from .logs import DEFAULT_CONSOLE, set_logger

logger = getLogger("tesseract")

# Jinja2 Template Environment
ENV = Environment(
    loader=PackageLoader("tesseract_core.sdk", "templates"),
    undefined=StrictUndefined,
)


class SpellcheckedTyperGroup(typer.core.TyperGroup):
    """A Typer group that suggests similar commands if a command is not found."""

    def get_command(self, ctx: click.Context, invoked_command: str) -> Any:
        """Get a command from the Typer group, suggesting similar commands if the command is not found."""
        import difflib

        possible_commands = self.list_commands(ctx)
        if invoked_command not in possible_commands:
            close_match = difflib.get_close_matches(
                invoked_command, possible_commands, n=1, cutoff=0.6
            )
            if close_match:
                raise click.UsageError(
                    f"No such command '{invoked_command}'. Did you mean '{close_match[0]}'?",
                    ctx,
                )
        return super().get_command(ctx, invoked_command)


app = typer.Typer(
    # Make -h an alias for --help
    context_settings={"help_option_names": ["-h", "--help"]},
    name="Tesseract",
    pretty_exceptions_show_locals=False,
    cls=SpellcheckedTyperGroup,
)

# Module-wide config
state = SimpleNamespace()
state.print_user_error_tracebacks = False

# Create a list of possible commands based on the ones in api_parse (kebab-cased)
POSSIBLE_CMDS = set(
    re.sub(r"([a-z])([A-Z])", r"\1-\2", object.name).replace("_", "-").lower()
    for object in EXPECTED_OBJECTS
)
POSSIBLE_CMDS.update({"health", "openapi-schema", "check"})

# All fields in TesseractConfig and TesseractBuildConfig for config override
POSSIBLE_KEYPATHS = TesseractConfig.model_fields.keys()
# Check that the only field that has nested fields is build_config
assert len(get_non_base_fields_in_tesseract_config()) == 1
POSSIBLE_BUILD_CONFIGS = TesseractBuildConfig.model_fields.keys()

# Traverse templates folder to seach for recipes
AVAILABLE_RECIPES = set()
for temp_with_path in ENV.list_templates(extensions=["py"]):
    temp_with_path = Path(temp_with_path)
    if temp_with_path.name == "tesseract_api.py" and temp_with_path.parent:
        AVAILABLE_RECIPES.add(str(temp_with_path.parent))
AVAILABLE_RECIPES = sorted(AVAILABLE_RECIPES)


class LogLevel(str, Enum):
    """Available log levels for Tesseract CLI."""

    # Must be an enum to represent a choice in Typer
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"


def _validate_tesseract_name(name: str | None) -> str:
    if name is None:
        if sys.stdout.isatty() and sys.stdin.isatty():
            name = typer.prompt("Enter a name for the Tesseract")
        else:
            raise typer.BadParameter(
                "Name must be provided as an argument or interactively."
            )

    forbidden_characters: str = ":;,.@#$%^&*()[]{}<>?|\\`~"
    if any(char in forbidden_characters for char in name) or any(
        char.isspace() for char in name
    ):
        raise typer.BadParameter(
            f"Name cannot contain whitespace or any of the following characters: {forbidden_characters}"
        )
    return name


def version_callback(value: bool | None) -> None:
    """Typer callback for version option."""
    if value:
        from tesseract_core import __version__

        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main_callback(
    loglevel: Annotated[
        LogLevel,
        typer.Option(
            help="Set the logging level. At debug level, also print tracebacks for user errors.",
            case_sensitive=False,
            show_default=True,
            metavar="LEVEL",
        ),
    ] = LogLevel.info,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Print Tesseract CLI version and exit.",
        ),
    ] = None,
) -> None:
    """Tesseract: A toolkit for re-usable, autodiff-native software components."""
    verbose_tracebacks = loglevel == LogLevel.debug
    state.print_user_error_tracebacks = verbose_tracebacks
    app.pretty_exceptions_show_locals = verbose_tracebacks

    set_logger(loglevel, catch_warnings=True, rich_format=True)


def _parse_config_override(
    options: list[str] | None,
) -> tuple[tuple[list[str], str], ...]:
    """Parse `["path1.path2.path3=value"]` into `[(["path1", "path2", "path3"], "value")]`."""
    if options is None:
        return []

    def _parse_option(option: str):
        bad_param = typer.BadParameter(
            f"Invalid config override {option} (must be `keypath=value`)",
            param_hint="config_override",
        )
        if option.count("=") != 1:
            raise bad_param

        key, value = option.split("=")
        if not key or not value:
            raise bad_param

        path = key.split(".")
        return path, value

    return tuple(_parse_option(option) for option in options)


@app.command("build")
@engine.needs_docker
def build_image(
    src_dir: Annotated[
        Path,
        typer.Argument(
            help=(
                "Source directory for the Tesseract. Must contain `tesseract_api.py` "
                "and `tesseract_config.yaml`."
            ),
            dir_okay=True,
            exists=True,
            file_okay=False,
            readable=True,
        ),
    ],
    tag: Annotated[
        str | None,
        typer.Option(
            "--tag",
            "-t",
            help="Tag for the resulting Tesseract. This can help you distinguish between "
            "Tesseracts with the same base name, like `cfd:v1` and cfd:v2`, "
            "both named `cfd` but tagged as `v1` and `v2` respectively.",
        ),
    ] = None,
    build_dir: Annotated[
        Path | None,
        typer.Option(
            help="Directory to use for the build. Defaults to a temporary directory.",
            dir_okay=True,
            file_okay=False,
            writable=True,
        ),
    ] = None,
    forward_ssh_agent: Annotated[
        bool,
        typer.Option(
            help=(
                "Forward the SSH agent to the Docker build environment. "
                "Has to be provided if requirements.txt contains private dependencies."
            ),
        ),
    ] = False,
    config_override: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "Override a configuration option in the Tesseract. "
                "Format: ``keypath=value`` where ``keypath`` is a dot-separated path to the "
                "attribute in tesseract_config.yaml. "
                "Possible keypaths are: "
                f"{', '.join(POSSIBLE_KEYPATHS)}. \n"
                "\n Possible build_config options are: "
                f"{', '.join(POSSIBLE_BUILD_CONFIGS)}. \n"
                "\nExample: ``--config-override build_config.target_platform=linux/arm64``."
            ),
            metavar="KEYPATH=VALUE",
        ),
    ] = None,
    keep_build_cache: Annotated[
        bool,
        typer.Option(
            help="Keep the Docker build cache (useful for debugging).",
            envvar="TESSERACT_KEEP_BUILD_CACHE",
        ),
    ] = False,
    generate_only: Annotated[
        bool,
        typer.Option(
            help="Only generate the build context and do not actually build the image."
        ),
    ] = False,
) -> None:
    """Build a new Tesseract from a context directory.

    The passed directory must contain the files `tesseract_api.py` and `tesseract_config.yaml`
    (can be created via `tesseract init`).
    """
    if config_override is None:
        config_override = []

    parsed_config_override = _parse_config_override(config_override)

    if generate_only:
        progress_indicator = nullcontext()
    else:
        progress_indicator = DEFAULT_CONSOLE.status(
            "[white]Processing", spinner="dots", spinner_style="white"
        )

    try:
        with progress_indicator:
            build_out = engine.build_tesseract(
                src_dir,
                tag,
                build_dir=build_dir,
                inject_ssh=forward_ssh_agent,
                config_override=parsed_config_override,
                keep_build_cache=keep_build_cache,
                generate_only=generate_only,
            )
    except docker.errors.BuildError as e:
        raise UserError(f"Error building Tesseract: {e}") from e
    except docker.errors.APIError as e:
        raise UserError(f"Docker server error: {e}") from e
    except TypeError as e:
        raise UserError(f"Input error building Tesseract: {e}") from e
    except PermissionError as e:
        raise UserError(f"Permission denied: {e}") from e

    if generate_only:
        # output is the path to the build context
        build_dir = build_out
        typer.echo(build_dir)
    else:
        # output is the built image
        image = build_out
        logger.info(f"Built image {image.short_id}, {image.tags}")


@app.command("init")
def init(
    name: Annotated[
        # Guaranteed to be a string by _validate_tesseract_name
        str | None,
        typer.Option(
            help="Tesseract name as specified in tesseract_config.yaml. Will be prompted if not provided.",
            callback=_validate_tesseract_name,
            show_default=False,
        ),
    ] = None,
    target_dir: Annotated[
        Path,
        typer.Option(
            help="Path to the directory where the Tesseract API module should be created.",
            dir_okay=True,
            file_okay=False,
            writable=True,
            show_default="current directory",
        ),
    ] = Path("."),
    recipe: Annotated[
        str,
        typer.Option(
            click_type=click.Choice(AVAILABLE_RECIPES),
            help="Use a pre-configured template to initialize Tesseract API and configuration.",
        ),
    ] = "base",
) -> None:
    """Initialize a new Tesseract API module."""
    logger.info(f"Initializing Tesseract {name} in directory: {target_dir}")
    engine.init_api(target_dir, name, recipe=recipe)


def _validate_port(port: str | None) -> str:
    """Validate port input."""
    if port is None:
        return ""

    port = port.strip()
    if "-" in port:
        start, end = port.split("-")
    else:
        start = end = port

    try:
        start, end = int(start), int(end)
    except ValueError as ex:
        raise typer.BadParameter(
            (f"Port '{port}' must be single integer or a range (e.g. -p '8000-8080')."),
            param_hint="port",
        ) from ex

    if start > end:
        raise typer.BadParameter(
            (
                f"Start port '{start}' must be less than "
                f"or equal to end port '{end}' (e.g. -p '8000-8080')."
            ),
            param_hint="port",
        )

    if not (1025 <= start <= 65535) or not (1025 <= end <= 65535):
        raise typer.BadParameter(
            f"Ports '{port}' must be between 1025 and 65535.",
            param_hint="port",
        )
    return port


@app.command("serve")
@engine.needs_docker
def serve(
    image_names: Annotated[
        list[str],
        typer.Argument(..., help="One or more Tesseract image names"),
    ],
    volume: Annotated[
        list[str] | None,
        typer.Option(
            "-v",
            "--volume",
            help="Bind mount a volume in all Tesseracts, in Docker format: source:target[:ro|rw]",
            metavar="source:target",
            show_default=False,
        ),
    ] = None,
    port: Annotated[
        str | None,
        typer.Option(
            "--port",
            "-p",
            help="Optional port/port range to serve the Tesseract on (e.g. -p '8080-8082'). "
            "Port must be between 1025 and 65535.",
            callback=_validate_port,
        ),
    ] = None,
    gpus: Annotated[
        list[str] | None,
        typer.Option(
            "--gpus",
            metavar="'all' | int",
            help=(
                "IDs of host Nvidia GPUs to make available in the Tesseract. "
                "You can use all GPUs via `--gpus all` or pass (multiple) IDs: `--gpus 0 --gpus 1`."
            ),
        ),
    ] = None,
) -> None:
    """Serve one or more Tesseract images.

    A successful serve command will display on standard output a JSON object
    with the Docker Compose project ID, which is required to run the teardown
    command, as well as a list of all containers spawned and their respective
    ports.
    """
    if port and len(image_names) > 1:
        # TODO: Docker compose with multiple ports is not supported until
        # docker/compose#7188 is resolved.
        raise typer.BadParameter(
            (
                "Port specification only works if 1 Tesseract is being served."
                f"Currently serving `{len(image_names)}` Tesseracts."
            ),
            param_hint="image_names",
        )

    try:
        project_id = engine.serve(image_names, port, volume, gpus)
        containers = engine.project_containers(project_id)
        _display_container_meta(containers)
        logger.info(
            f"Docker Compose Project ID, use it with 'tesseract teardown' command: {project_id}"
        )

        project_meta = {"project_id": project_id, "containers": []}
        for container in containers:
            project_meta["containers"].append(
                {"name": container.name, "port": _get_container_host_port(container)}
            )

        json_info = json.dumps(project_meta)
        typer.echo(json_info, nl=False)

    except ValueError as ex:
        raise typer.BadParameter(f"{ex}", param_hint="image_names") from ex
    except RuntimeError as ex:
        raise UserError(
            f"Internal Docker error occurred while serving Tesseracts: {ex}"
        ) from ex


@app.command("list")
@engine.needs_docker
def list_tesseract_images() -> None:
    """Display all Tesseract images."""
    tesseract_images = engine.get_tesseract_images()
    _display_tesseract_image_meta(tesseract_images)


@app.command("ps")
@engine.needs_docker
def list_tesseract_containers() -> None:
    """Display all Tesseract containers."""
    tesseract_containers = engine.get_tesseract_containers()
    _display_tesseract_containers_meta(tesseract_containers)


def _display_tesseract_image_meta(
    docker_assets: list[docker.models.images.Image],
) -> None:
    """Display Tesseract image metadata."""
    table = RichTable("ID", "Tags", "Name", "Version", "Description")
    for asset in docker_assets:
        tesseract_vals = _get_tesseract_env_vals(asset)
        if tesseract_vals:
            table.add_row(
                # Checksum Type + First 12 Chars of ID
                asset.id[:19],
                str(asset.attrs["RepoTags"]),
                tesseract_vals["TESSERACT_NAME"],
                tesseract_vals["TESSERACT_VERSION"],
                tesseract_vals.get("TESSERACT_DESCRIPTION", "").replace("\n", " "),
            )
    RichConsole().print(table)


def _display_tesseract_containers_meta(
    docker_assets: list[docker.models.containers.Container],
) -> None:
    """Display Tesseract containers metadata."""
    table = RichTable("ID", "Name", "Version", "Host Port", "Project ID", "Description")
    docker_compose_projects = _docker_compose_projects()

    for asset in docker_assets:
        tesseract_vals = _get_tesseract_env_vals(asset)
        if tesseract_vals:
            tesseract_project = _find_tesseract_project(asset, docker_compose_projects)
            table.add_row(
                asset.id[:12],
                tesseract_vals["TESSERACT_NAME"],
                tesseract_vals["TESSERACT_VERSION"],
                _get_container_host_port(asset),
                tesseract_project,
                tesseract_vals.get("TESSERACT_DESCRIPTION", "").replace("\\n", " "),
            )
    RichConsole().print(table)


def _get_tesseract_env_vals(
    docker_asset: docker.models.images.Image | docker.models.containers.Container,
) -> dict:
    """Convert Tesseract environment variables from list to dictionary."""
    env_vals = [s for s in docker_asset.attrs["Config"]["Env"] if "TESSERACT_" in s]
    return {item.split("=")[0]: item.split("=")[1] for item in env_vals}


def _find_tesseract_project(
    tesseract: docker.models.containers.Container,
    docker_compose_projects: defaultdict[str, list],
) -> str:
    """Find the Tesseract Project ID for a given tesseract."""
    tesseract_id = tesseract.id[:12]

    for project, containers in docker_compose_projects.items():
        if tesseract_id in containers:
            return project

    return "Unknown"


def _docker_compose_projects() -> defaultdict[str, list]:
    """List Docker Compose projects.

    Build a dictionary {project_name: [container ID]}.
    """
    proc = subprocess.run(
        ["docker", "compose", "ls", "--format", "json"],
        check=True,
        capture_output=True,
    )

    out = proc.stdout.decode().strip()
    compose_projects = json.loads(out)

    projects_map = defaultdict(list)

    for project in compose_projects:
        project_name = project["Name"]

        proc = subprocess.run(
            [
                "docker",
                "compose",
                "--project-name",
                project_name,
                "ps",
                "--format",
                "json",
            ],
            check=True,
            capture_output=True,
        )
        # This command outputs a series of JSON documents, one for each
        # container, instead of a JSON with a list of entries.
        compose_ps = proc.stdout.decode().strip().split("\n")
        for asset in compose_ps:
            metadata = json.loads(asset)
            projects_map[project_name].append(metadata["ID"])

    return projects_map


@app.command("apidoc")
@engine.needs_docker
def apidoc(
    image_name: Annotated[
        str,
        typer.Argument(..., help="Tesseract image name"),
    ],
    browser: Annotated[
        bool,
        typer.Option(help="Open the browser after serving the OpenAPI schema"),
    ] = True,
) -> None:
    """Serve the OpenAPI schema for a Tesseract."""
    project_id = None

    try:
        project_id = engine.serve([image_name])
        container = engine.project_containers(project_id)[0]
        host_port = _get_container_host_port(container)
        url = f"http://localhost:{host_port}/docs"
        logger.info(f"Serving OpenAPI docs for Tesseract {image_name} at {url}")
        logger.info("  Press Ctrl+C to stop")
        if browser:
            webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return
    finally:
        if project_id is not None:
            engine.teardown(project_id)


def _display_container_meta(
    containers: list[docker.models.containers.Container],
) -> None:
    """Display container metadata."""
    for container in containers:
        logger.info(f"Container ID: {container.id}")
        logger.info(f"Name: {container.name}")
        entrypoint = container.attrs["Config"]["Entrypoint"]
        logger.info(f"Entrypoint: {entrypoint}")
        port_key = next(iter(container.ports))
        host_port = container.ports[port_key][0]["HostPort"]
        logger.info(f"View Tesseract: http://localhost:{host_port}/docs")


def _get_container_host_port(container: docker.models.containers.Container):
    port_key = next(iter(container.ports))
    host_port = container.ports[port_key][0]["HostPort"]
    return host_port


@app.command("teardown")
@engine.needs_docker
def teardown(
    project_ids: Annotated[
        list[str] | None,
        typer.Argument(..., help="Docker Compose project IDs"),
    ] = None,
    tear_all: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Tear down all Tesseracts currently being served.",
        ),
    ] = False,
) -> None:
    """Tear down one or more Tesseracts that were previously started with `tesseract serve`.

    One or more Tesseract project ids must be specified unless `--all` is set.
    """
    if not project_ids and not tear_all:
        raise typer.BadParameter(
            "Either project IDs or --all flag must be provided",
            param_hint="project_ids",
        )

    if project_ids and tear_all:
        raise typer.BadParameter(
            "Either project IDs or --all flag must be provided, but not both",
            param_hint="project_ids",
        )
    if tear_all:
        project_ids = []
        docker_compose_projects = _docker_compose_projects()
        for project, _ in docker_compose_projects.items():
            if "tesseract-" in project:
                project_ids.append(project)

    for project_id in project_ids:
        try:
            engine.teardown(project_id)
        except ValueError as ex:
            raise UserError(
                f"Input error occurred while tearing down Tesseracts: {ex}"
            ) from ex
        except RuntimeError as ex:
            raise UserError(
                f"Internal Docker error occurred while tearing down Tesseracts: {ex}"
            ) from ex
        logger.info(
            f"Tesseracts are shutdown for Docker Compose project ID: {project_id}"
        )


def _sanitize_error_output(error_output: str, tesseract_image: str) -> str:
    """Remove references to tesseract-runtime and unavailable commands from error output."""
    # Replace references to tesseract-runtime with tesseract run
    error_output = re.sub(
        r"Try 'tesseract-runtime",
        f"Try 'tesseract run {tesseract_image}",
        error_output,
    )

    error_output = re.sub(
        r"Usage: tesseract-runtime",
        f"Usage: tesseract run {tesseract_image}",
        error_output,
    )

    # Hide commands in help strings that users are not supposed to use via tesseract run
    error_output = re.sub(
        r"^│\s+(serve|health)\s+.*?$\n",
        "",
        error_output,
        flags=re.MULTILINE,
    )

    return error_output


@app.command(
    "run",
    # We need to ignore unknown options to forward the args to the Tesseract container
    context_settings={"ignore_unknown_options": True},
    # We implement --help manually to forward the help of the Tesseract container
    add_help_option=False,
)
@engine.needs_docker
def run_container(
    context: click.Context,
    tesseract_image: Annotated[
        str,
        typer.Argument(help="Tesseract image name"),
    ],
    cmd: Annotated[
        str,
        typer.Argument(help="Tesseract command to run"),
    ] = "",
    args: Annotated[
        list[str] | None,
        typer.Argument(help="Arguments for the command"),
    ] = None,
    volume: Annotated[
        list[str] | None,
        typer.Option(
            "-v",
            "--volume",
            help="Bind mount a volume, in Docker format: source:target.",
            metavar="source:target",
            show_default=False,
        ),
    ] = None,
    gpus: Annotated[
        list[str] | None,
        typer.Option(
            "--gpus",
            metavar="'all' | int",
            help=(
                "IDs of host GPUs to make available in the tesseract. "
                "You can use all GPUs via `--gpus all` or pass (multiple) IDs: `--gpus 0 --gpus 1`."
            ),
        ),
    ] = None,
) -> None:
    """Execute a command in a Tesseract.

    This command starts a Tesseract instance and executes the given
    command.
    """
    if args is None:
        args = []

    if cmd == "serve":
        logger.error(
            "You should not serve tesseracts via "
            "`tesseract run <tesseract-name> serve`. "
            "Use `tesseract serve <tesseract-name>` instead."
        )
        raise typer.Exit(1)

    help_args = {"-h", "--help"}

    # When called as `tesseract run --help` -> show generic help
    if tesseract_image in help_args:
        context.get_help()
        return

    invoke_help = any(arg in help_args for arg in args) or cmd in help_args

    if (not cmd or cmd not in POSSIBLE_CMDS) and not invoke_help:
        if not cmd:
            error_string = f"Command is required. Are you sure your Tesseract image name is `{tesseract_image}`?\n"
        else:
            error_string = f"Command `{cmd}` does not exist. \n"

        error_string += (
            f"\nRun `tesseract run {tesseract_image} --help` for more information.\n"
        )

        error_string = (
            error_string + f"\nPossible commands are: {', '.join(POSSIBLE_CMDS)}"
        )
        raise typer.BadParameter(error_string, param_hint="cmd")

    try:
        result_out, result_err = engine.run_tesseract(
            tesseract_image, cmd, args, volumes=volume, gpus=gpus
        )

    except docker.errors.ImageNotFound as e:
        raise UserError(
            "Tesseract image not found. "
            f"Are you sure your tesseract image name is {tesseract_image}?\n\n{e}"
        ) from e

    except (
        docker.errors.APIError,
        docker.errors.ContainerError,
    ) as e:
        if "No such command" in str(e):
            error_string = f"Error running Tesseract '{tesseract_image}' \n\n Error: Unimplemented command '{cmd}'.  "
        else:
            error_string = _sanitize_error_output(
                f"Error running Tesseract. \n\n{e}", tesseract_image
            )

        raise UserError(error_string) from e

    if invoke_help:
        result_err = _sanitize_error_output(result_err, tesseract_image)

    typer.echo(result_err, err=True, nl=False)
    typer.echo(result_out, nl=False)


def entrypoint() -> NoReturn:
    """Entrypoint for the Tesseract CLI."""
    try:
        result = app()
    except UserError as e:
        logger.error(f"{e}", exc_info=state.print_user_error_tracebacks)
        result = 1
    except Exception as e:
        logger.critical(f"Uncaught error: {e}", exc_info=True)
        result = 2

    if result > 0:
        logger.critical("Aborting")

    raise SystemExit(result)


# Expose the underlying click object for doc generation
typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    entrypoint()
