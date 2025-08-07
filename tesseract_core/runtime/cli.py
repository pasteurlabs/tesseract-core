# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""This module provides a command-line interface for interacting with the Tesseract runtime."""

import inspect
import io
import json
import sys
from collections.abc import Callable, Iterable
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    get_args,
    get_origin,
)

import click
import typer
from pydantic import ValidationError

from tesseract_core.runtime.config import RuntimeConfig, get_config, update_config
from tesseract_core.runtime.core import (
    check_tesseract_api,
    create_endpoints,
    get_tesseract_api,
    redirect_fd,
)
from tesseract_core.runtime.file_interactions import (
    output_to_bytes,
    read_from_path,
    write_to_path,
)
from tesseract_core.runtime.finite_differences import (
    check_gradients as check_gradients_,
)
from tesseract_core.runtime.serve import create_rest_api
from tesseract_core.runtime.serve import serve as serve_

CONFIG_FIELDS = {
    str(field_name): field.annotation
    for field_name, field in RuntimeConfig.model_fields.items()
}


def make_choice_enum(name: str, choices: Iterable[str]) -> type[Enum]:
    """Create a choice enum for Typer."""
    return Enum(name, [(choice.upper(), choice) for choice in choices], type=str)


def _enum_to_val(val: Any) -> Any:
    """Convert an Enum value to its raw representation."""
    if isinstance(val, Enum):
        return val.value
    return val


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
    name="tesseract-runtime",
    cls=SpellcheckedTyperGroup,
    no_args_is_help=True,
    help="Invoke the Tesseract runtime.",
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _prettify_docstring(docstr: str) -> str:
    """Enforce consistent indentation level of docstrings."""
    # First line is not indented, the rest is -> leads to formatting issues
    docstring_lines = docstr.split("\n")
    dedented_lines = dedent("\n".join(docstring_lines[1:]))
    return "\n".join([docstring_lines[0].lstrip(), dedented_lines])


def _parse_payload(value: Any) -> dict[str, Any]:
    """Callback to parse Tesseract input arguments provided in the CLI."""
    if not isinstance(value, str):
        # Passthrough, probably a default value
        return value

    if value.startswith("@"):
        try:
            value = read_from_path(value[1:]).decode("utf-8")
        except Exception as e:
            raise click.BadParameter(f"Could not read data from path {value}.") from e

    return json.loads(value)


def make_callback() -> Callable:
    """Create a callback function for the Tesseract runtime CLI.

    This function dynamically generates a callback function that can be used with the Typer CLI
    that includes all configuration fields as options.
    """

    def main_callback(**kwargs: Any) -> None:
        """Invoke the Tesseract runtime.

        The Tesseract runtime can be configured via environment variables; for example,
        ``export TESSERACT_RUNTIME_PORT=8080`` sets the port to use for ``tesseract serve`` to 8080.
        """
        update_config(
            **{key: _enum_to_val(val) for key, val in kwargs.items() if val is not None}
        )

    params = []
    for field_name, field_type in CONFIG_FIELDS.items():
        if field_name == "api_path":
            # Too late to configure here, as the API path is needed to load the Tesseract API
            continue

        if get_origin(field_type) is Literal:
            field_type = make_choice_enum(f"{field_name}Choices", get_args(field_type))

        params.append(
            inspect.Parameter(
                field_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Annotated[
                    field_type,
                    typer.Option(
                        "--" + field_name.replace("_", "-"),
                        help=f"Overwrite the `{field_name}` config option.",
                        show_default=False,
                    ),
                ],
            )
        )
    main_callback.__signature__ = inspect.Signature(
        parameters=params,
        return_annotation=None,
    )
    return main_callback


main_callback = make_callback()
app.callback()(main_callback)


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


@app.command("check")
def check() -> None:
    """Check whether the Tesseract API is valid."""
    api_module = get_tesseract_api()
    check_tesseract_api(api_module)
    typer.echo("✅ Tesseract API check successful ✅")


@app.command("check-gradients")
def check_gradients(
    payload: Annotated[
        str,
        typer.Argument(
            help="JSON payload to pass to the Tesseract API. If prefixed with '@', it is treated as a file path.",
            metavar="JSON_PAYLOAD",
            show_default=False,
        ),
    ],
    input_paths: Annotated[
        Optional[list[str]],
        typer.Option(
            "--input-paths",
            help="Paths to differentiable inputs to check gradients for.",
            show_default="check all",
        ),
    ] = None,
    output_paths: Annotated[
        Optional[list[str]],
        typer.Option(
            "--output-paths",
            help="Paths to differentiable outputs to check gradients for.",
            show_default="check all",
        ),
    ] = None,
    endpoints: Annotated[
        Optional[list[str]],
        typer.Option(
            "--endpoints",
            help="Endpoints to check gradients for.",
            show_default="check all",
        ),
    ] = None,
    eps: Annotated[
        float,
        typer.Option(
            "--eps",
            help="Step size for finite differences.",
            show_default=True,
        ),
    ] = 1e-4,
    rtol: Annotated[
        float,
        typer.Option(
            "--rtol",
            help="Relative tolerance when comparing finite differences to gradients.",
            show_default=True,
        ),
    ] = 0.1,
    max_evals: Annotated[
        int,
        typer.Option(
            "--max-evals",
            help="Maximum number of evaluations per input.",
            show_default=True,
        ),
    ] = 1000,
    max_failures: Annotated[
        int,
        typer.Option(
            "--max-failures",
            help="Maximum number of failures to report per endpoint.",
            show_default=True,
        ),
    ] = 10,
    seed: Annotated[
        Optional[int],
        typer.Option(
            "--seed",
            help="Seed for random number generator. If not set, a random seed is used.",
            show_default=False,
        ),
    ] = None,
    show_progress: Annotated[
        bool,
        typer.Option(
            "--show-progress",
            help="Show progress bar.",
        ),
    ] = True,
) -> None:
    """Check gradients of endpoints against a finite difference approximation.

    \b
    This is an automated way to check the correctness of the gradients of the different AD endpoints
    (jacobian, jacobian_vector_product, vector_jacobian_product) of a ``tesseract_api.py`` module.
    It will sample random indices and compare the gradients computed by the AD endpoints with the
    finite difference approximation.

    \b
    Warning:
        Finite differences are not exact and the comparison is done with a tolerance. This means
        that the check may fail even if the gradients are correct, and vice versa.

    \b
    Finite difference approximations are sensitive to numerical precision. When finite differences
    are reported incorrectly as 0.0, it is likely that the chosen `eps` is too small, especially for
    inputs that do not use float64 precision.
    """  # noqa: D301
    config = get_config()
    api_module = get_tesseract_api()
    inputs = _parse_payload(payload)

    result_iter = check_gradients_(
        api_module,
        inputs,
        base_dir=Path(config.input_path) if config.input_path else None,
        input_paths=input_paths,
        output_paths=output_paths,
        endpoints=endpoints,
        max_evals=max_evals,
        eps=eps,
        rtol=rtol,
        seed=seed,
        show_progress=show_progress,
    )

    failed = False
    for endpoint, failures, num_evals in result_iter:
        if not failures:
            typer.echo(
                f"✅ Gradient check for {endpoint} passed ✅ ({len(failures)} failures / {num_evals} checks)"
            )
        else:
            failed = True
            typer.echo()
            typer.echo(
                f"⚠️ Gradient check for {endpoint} failed ⚠️ ({len(failures)} failures / {num_evals} checks)"
            )
            printed_failures = min(len(failures), max_failures)
            typer.echo(f"First {printed_failures} failures:")
            for failure in failures[:printed_failures]:
                typer.echo(
                    f"  Input path: '{failure.in_path}', Output path: '{failure.out_path}', Index: {failure.idx}"
                )
                if failure.exception:
                    typer.echo(f"  Encountered exception: {failure.exception}")
                else:
                    typer.echo(f"  {endpoint} value: {failure.grad_val}")
                    typer.echo(f"  Finite difference value: {failure.ref_val}")
                typer.echo()

    if failed:
        typer.echo("❌ Some gradient checks failed ❌")
        sys.exit(1)


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option(help="Host IP address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port number")] = 8000,
    num_workers: Annotated[int, typer.Option(help="Number of worker processes")] = 1,
) -> None:
    """Start running this Tesseract's web server."""
    serve_(host=host, port=port, num_workers=num_workers)


def _create_user_defined_cli_command(
    app: typer.Typer, user_function: Callable, out_stream: Optional[io.TextIOBase]
) -> None:
    """Creates a click command which sends requests to Tesseract endpoints.

    We need to do this dynamically, as we want to generate docs and usage
    from the Tesseract api's signature and docstrings.

    Args:
        app: The Typer application to add the command to.
        user_function: The user-defined function to create a CLI command for.
        out_stream: The default output stream to write to. If None, defaults to
            sys.stdout at the time of invocation.
    """
    InputSchema = user_function.__annotations__.get("payload", None)
    OutputSchema = user_function.__annotations__.get("return", None)

    def _callback_wrapper(**kwargs: Any):
        config = get_config()
        input_path = config.input_path
        output_path = config.output_path
        output_format = config.output_format
        output_file = config.output_file

        out_stream_ = out_stream or sys.stdout

        user_function_args = {}

        if InputSchema is not None:
            payload = kwargs["payload"]
            try:
                user_function_args["payload"] = InputSchema.model_validate(
                    payload,
                    context={"base_dir": input_path},
                )
            except ValidationError as e:
                raise click.BadParameter(
                    str(e),
                    param_hint="payload",
                ) from e

        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)

        result = user_function(**user_function_args)
        result = output_to_bytes(result, output_format, output_path)

        # write raw bytes to out_stream.buffer to support binary data (which may e.g. be piped)
        if not output_file:
            out_stream_.buffer.write(result)
            out_stream_.flush()
        else:
            write_to_path(result, f"{output_path}/{output_file}")

    function_name = user_function.__name__.replace("_", "-")

    # Assemble docstring
    function_doc = [_prettify_docstring(user_function.__doc__)]

    if InputSchema is not None and hasattr(InputSchema, "model_fields"):
        function_doc.append(
            "\nFirst argument is the payload, which should be a JSON object with the following structure."
        )

        # \b\n disables click's automatic formatting
        function_doc.append("\n\nInput schema:")
        function_doc.append(_schema_to_docstring(InputSchema, current_indent=4))

    if OutputSchema is not None and hasattr(OutputSchema, "model_fields"):
        function_doc.append("\n\nReturns:")
        function_doc.append(_schema_to_docstring(OutputSchema, current_indent=4))

    if InputSchema is not None:

        def command_func(payload: str):
            parsed_payload = _parse_payload(payload)
            return _callback_wrapper(payload=parsed_payload)
    else:

        def command_func():
            return _callback_wrapper()

    help_string = "\n".join(function_doc).replace("\n\n", "\n\n\b")
    decorator = app.command(
        function_name,
        short_help=f"Call the Tesseract function {function_name}.",
        help=help_string,
    )
    decorator(command_func)


def _add_user_commands_to_cli(
    app: typer.Typer, out_stream: Optional[io.IOBase]
) -> None:
    tesseract_package = get_tesseract_api()
    endpoints = create_endpoints(tesseract_package)

    def openapi_schema() -> dict:
        """Get the openapi.json schema."""
        openapi_schema_ = create_rest_api(tesseract_package).openapi()
        return openapi_schema_

    endpoints.append(openapi_schema)

    for func in endpoints:
        _create_user_defined_cli_command(app, func, out_stream)


def main() -> None:
    """Entrypoint for the command line interface."""
    # Redirect stdout to stderr to avoid mixing any output with the JSON response.
    with redirect_fd(sys.stdout, sys.stderr) as orig_stdout:
        # Fail as fast as possible if the Tesseract API path is not set
        api_path = get_config().api_path
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

        _add_user_commands_to_cli(app, out_stream=orig_stdout)
        app(auto_envvar_prefix="TESSERACT_RUNTIME")


if __name__ == "__main__":
    main()
