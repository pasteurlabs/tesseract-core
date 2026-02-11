# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An instantiated version of the CLI app with user commands added.

Used only for generating documentation. As such, we perform some additional
formatting on the docstrings to make them more readable.

!! Do not use for anything else !!
"""

import copy
import types
from textwrap import indent
from typing import Any, get_args, get_origin

import typer

from tesseract_core.runtime.cli import (
    _add_user_commands_to_cli,
    _prettify_docstring,
)
from tesseract_core.runtime.cli import (
    app as cli_app,
)
from tesseract_core.runtime.core import create_endpoints, get_tesseract_api


def _format_type_annotation(annotation: type[Any] | types.UnionType) -> str:
    """Format a type annotation as a human-readable string for documentation.

    Args:
        annotation: A type annotation from a Pydantic field. Can be a simple type
            (int, str, etc.), a Union type (str | None), or a generic type (list[str]).

    Returns:
        A human-readable string representation of the type.

    Examples:
        int -> "int"
        str | None -> "optional str"
        str | float -> "str | float"
        float | int | None -> "optional float | int"
    """
    # Handle simple types with __name__
    if hasattr(annotation, "__name__"):
        return annotation.__name__

    # Handle Union types (e.g., str | None, int | float)
    if isinstance(annotation, types.UnionType):
        args = get_args(annotation)
        # Check if None is one of the args
        none_type = type(None)
        if none_type in args:
            # Filter out None and format remaining types
            non_none_args = [arg for arg in args if arg is not none_type]
            if len(non_none_args) == 1:
                # str | None -> "optional str"
                return f"optional {_format_type_annotation(non_none_args[0])}"
            else:
                # float | int | None -> "optional float | int"
                formatted = " | ".join(
                    _format_type_annotation(arg) for arg in non_none_args
                )
                return f"optional {formatted}"
        else:
            # str | float -> "str | float"
            return " | ".join(_format_type_annotation(arg) for arg in args)

    # Handle typing generics (e.g., list[str], dict[str, int])
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if args:
            formatted_args = ", ".join(_format_type_annotation(arg) for arg in args)
            origin_name = getattr(origin, "__name__", str(origin))
            return f"{origin_name}[{formatted_args}]"
        return getattr(origin, "__name__", str(origin))

    # Fallback to string representation
    return str(annotation)


tesseract_api = get_tesseract_api()

# purge dummy docstrings so they don't leak into the docs
for func in tesseract_api.__dict__.values():
    # don't touch non-functions or functions from other modules
    if not callable(func) or func.__module__ != "tesseract_api":
        continue
    func.__doc__ = None

endpoints = create_endpoints(tesseract_api)

# format docstrings to play well with autodocs
for func in endpoints:
    docstring_parts = [_prettify_docstring(func.__doc__)]

    # populate endpoint docstrings with field info
    input_schema = func.__annotations__.get("payload")
    input_docs = []
    if hasattr(input_schema, "model_fields"):
        for field_name, field in input_schema.model_fields.items():
            input_docs.append(
                f"{field_name} ({_format_type_annotation(field.annotation)}): {field.description}"
            )
    if input_docs:
        docstring_parts.append("")
        docstring_parts.append("Arguments:")
        docstring_parts.append(indent("\n".join(input_docs), "  "))

    output_schema = func.__annotations__.get("return")
    output_docs = []
    if hasattr(output_schema, "model_fields"):
        for field_name, field in output_schema.model_fields.items():
            output_docs.append(
                f"{field_name} ({_format_type_annotation(field.annotation)}): {field.description}"
            )
    if output_docs:
        docstring_parts.append("")
        docstring_parts.append("Returns:")
        docstring_parts.append(indent("\n".join(output_docs), "  "))

    func.__doc__ = "\n".join(docstring_parts)

# add all user-defined functions to the global namespace, so we can do
# `from tesseract_core.runtime.app_cli import jacobian`
globals().update({func.__name__: func for func in endpoints})

cli_app = copy.deepcopy(cli_app)
_add_user_commands_to_cli(cli_app, out_stream=None)

# Expose the underlying click object for doc generation
tesseract_runtime_cli = typer.main.get_command(cli_app)
