# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""This module provides a command-line interface for interacting with the Tesseract runtime."""

import contextlib
import io
import os
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional

import click
import typer

from tesseract_core.runtime.config import get_config
from tesseract_core.runtime.core import create_endpoints, get_tesseract_api
from tesseract_core.runtime.file_interactions import (
    SUPPORTED_FORMATS,
    guess_format_from_path,
    load_bytes,
    output_to_bytes,
    read_from_path,
    write_to_path,
)
from tesseract_core.runtime.serve import create_rest_api
from tesseract_core.runtime.serve import serve as serve_


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


app = typer.Typer(name="tesseract-runtime", cls=SpellcheckedTyperGroup)


def _prettify_docstring(docstr: str) -> str:
    """Enforce consistent indentation level of docstrings."""
    # First line is not indented, the rest is -> leads to formatting issues
    docstring_lines = docstr.split("\n")
    dedented_lines = dedent("\n".join(docstring_lines[1:]))
    return "\n".join([docstring_lines[0].lstrip(), dedented_lines])


def _parse_arg_callback(
    ctx: Any, param: Any, value: Any
) -> tuple[dict[str, Any], Optional[Path]]:
    """Click callback to parse Tesseract input arguments provided in the CLI.

    Returns a tuple of the parsed value and the base directory of the input path, if any.
    """
    base_dir = None

    if not isinstance(value, str):
        # Passthrough, probably a default value
        return value, base_dir

    if value.startswith("@"):
        base_dir = Path(value[1:]).parent
        value_format = guess_format_from_path(value[1:])
        try:
            value_bytes = read_from_path(value[1:])
        except Exception as e:
            raise click.BadParameter(f"Could not read data from path {value}") from e
    else:
        # Data given directly via the CLI is always in JSON format
        value_format = "json"
        value_bytes = value.encode()

    try:
        decoded_value = load_bytes(value_bytes, value_format)
    except Exception as e:
        raise click.BadParameter(
            f"Could not decode data using format {value_format}"
        ) from e

    return decoded_value, base_dir


@app.callback(
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
def main_callback() -> None:
    """Invoke the Tesseract runtime.

    The Tesseract runtime can be configured via environment variables; for example,
    ``export TESSERACT_RUNTIME_PORT=8080`` sets the port to use for ``tesseract serve`` to 8080.
    """
    pass


tesseract_runtime = typer.main.get_command(app)


def _schema_to_docstring(schema: Any, current_indent: int = 0) -> str:
    """Convert a Pydantic schema to a human-readable docstring."""
    docstring = []
    indent = " " * current_indent

    if not hasattr(schema, "model_fields"):
        return ""

    is_root = schema.model_fields.keys() == {"root"}

    for field, field_info in schema.model_fields.items():
        if is_root:
            docline = f"{indent}{field_info.description}"
        else:
            docline = f"{indent}{field}: {field_info.description}"

        if (
            field_info.default is not None
            and str(field_info.default) != "PydanticUndefined"
        ):
            docline = f"{docline} [default: {field_info.default}]"

        docstring.append(docline)

        if hasattr(field_info.annotation, "model_fields"):
            docstring.append(
                _schema_to_docstring(field_info.annotation, current_indent + 4)
            )

    return "\n".join(docstring)


@tesseract_runtime.command()
def check() -> None:
    """Check whether the Tesseract API can be imported."""
    # raises an exception if the API cannot be imported
    get_tesseract_api()
    typer.echo("✅ Tesseract API check successful ✅")


@tesseract_runtime.command()
@click.option("-p", "--port", default=8000, help="Port number")
@click.option("-h", "--host", default="0.0.0.0", help="Host IP address")
@click.option("-w", "--num-workers", default=1, help="Number of worker processes")
def serve(host: str, port: int, num_workers: int) -> None:
    """Start running this Tesseract's web server."""
    serve_(host=host, port=port, num_workers=num_workers)


def _create_user_defined_cli_command(
    user_function: Callable, out_stream: Optional[io.IOBase]
) -> click.Command:
    """Creates a click command which sends requests to Tesseract endpoints.

    We need to do this dynamically, as we want to generate docs and usage
    from the Tesseract api's signature and docstrings.

    Args:
        user_function: The user-defined function to create a CLI command for.
        out_stream: The default output stream to write to. If None, defaults to
            sys.stdout at the time of invocation.
    """
    InputSchema = user_function.__annotations__.get("payload", None)
    OutputSchema = user_function.__annotations__.get("return", None)

    options = []

    if InputSchema is not None:
        options.append(
            click.Argument(
                ["payload"],
                type=click.STRING,
                required=True,
                metavar="JSON_PAYLOAD",
                callback=_parse_arg_callback,
            )
        )

    options.extend(
        [
            click.Option(
                ["-o", "--output-path"],
                type=click.STRING,
                help=(
                    "Output path to write the result to, such as local directory or S3 URI "
                    "(may be anything supported by fsspec). [default: write to stdout]"
                ),
                default=None,
            ),
            click.Option(
                ["-f", "--output-format"],
                type=click.Choice(SUPPORTED_FORMATS),
                help="Output format to write results in.",
                default="json",
            ),
        ]
    )

    def _callback_wrapper(
        output_path: Optional[str],
        output_format: SUPPORTED_FORMATS,
        **optional_args: Any,
    ):
        if output_format == "json+binref" and output_path is None:
            raise ValueError("--output-path must be specified for json+binref format")

        out_stream_ = out_stream or sys.stdout

        user_function_args = {}

        if InputSchema is not None:
            payload, base_dir = optional_args["payload"]
            user_function_args["payload"] = InputSchema.model_validate(
                payload, context={"base_dir": base_dir}
            )

        result = user_function(**user_function_args)
        result = output_to_bytes(result, output_format, output_path)

        if output_path is None:
            # write raw bytes to out_stream.buffer to support binary data (which may e.g. be piped)
            out_stream_.buffer.write(result)
            out_stream_.flush()
        else:
            format = output_format.split("+", maxsplit=1)[0]
            write_to_path(result, f"{output_path}/results.{format}")

    function_name = user_function.__name__.replace("_", "-")

    # Assemble docstring
    function_doc = [_prettify_docstring(user_function.__doc__)]

    if InputSchema is not None and hasattr(InputSchema, "model_fields"):
        function_doc.append(
            "\nFirst argument is the payload, which should be a JSON object with the following structure."
        )

        # \b\n disables click's automatic formatting
        function_doc.append("\n\n\b\nInput schema:")
        function_doc.append(_schema_to_docstring(InputSchema, current_indent=4))

    if OutputSchema is not None and hasattr(OutputSchema, "model_fields"):
        function_doc.append("\n\n\b\nReturns:")
        function_doc.append(_schema_to_docstring(OutputSchema, current_indent=4))

    command = click.Command(
        function_name,
        short_help=f"Call the Tesseract function {function_name}.",
        help="\n".join(function_doc),
        callback=_callback_wrapper,
        params=options,
    )

    return command


def _add_user_commands_to_cli(
    group: click.Group, out_stream: Optional[io.IOBase]
) -> click.Group:
    tesseract_package = get_tesseract_api()
    endpoints = create_endpoints(tesseract_package)

    openapi_schema_ = create_rest_api(tesseract_package).openapi()

    def openapi_schema() -> dict:
        """Get the openapi.json schema."""
        return openapi_schema_

    endpoints.append(openapi_schema)

    for func in endpoints:
        group.add_command(_create_user_defined_cli_command(func, out_stream))

    return group


@contextlib.contextmanager
def stdout_to_stderr() -> Generator:
    """Redirect stdout to stderr at OS level."""
    orig_stdout = os.dup(sys.stdout.fileno())
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        yield os.fdopen(orig_stdout, "w", closefd=False)
    finally:
        sys.stdout.flush()
        os.dup2(orig_stdout, sys.stdout.fileno())


def main() -> None:
    """Entrypoint for the command line interface."""
    # Redirect stdout to stderr to avoid mixing any output with the JSON response.
    with stdout_to_stderr() as orig_stdout:
        # Fail as fast as possible if the Tesseract API path is not set
        api_path = get_config().tesseract_api_path
        if not api_path.is_file():
            print(
                f"Tesseract API file '{api_path}' does not exist. "
                "Please ensure it is a valid file, or set the TESSERACT_API_PATH "
                "environment variable to the path of your Tesseract API file.\n"
                "\n"
                "Example:\n"
                "    $ export TESSERACT_API_PATH=/path/to/your/tesseract_api.py\n"
                "\n"
                "Aborted.",
                file=sys.stderr,
            )
            sys.exit(1)
        cli = _add_user_commands_to_cli(tesseract_runtime, out_stream=orig_stdout)
        cli(auto_envvar_prefix="TESSERACT_RUNTIME")


if __name__ == "__main__":
    main()
