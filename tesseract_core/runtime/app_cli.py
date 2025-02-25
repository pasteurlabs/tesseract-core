# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An instantiated version of the CLI app with user commands added.

Used only for generating documentation. As such, we perform some additional
formatting on the docstrings to make them more readable.

!! Do not use for anything else !!
"""

import copy
from textwrap import indent

from tesseract_core.runtime.cli import (
    _add_user_commands_to_cli,
    _prettify_docstring,
    tesseract_runtime,
)
from tesseract_core.runtime.core import create_endpoints, get_tesseract_api

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
                f"{field_name} ({field.annotation.__name__}): {field.description}"
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
                f"{field_name} ({field.annotation.__name__}): {field.description}"
            )
    if output_docs:
        docstring_parts.append("")
        docstring_parts.append("Returns:")
        docstring_parts.append(indent("\n".join(output_docs), "  "))

    func.__doc__ = "\n".join(docstring_parts)

# add all user-defined functions to the global namespace, so we can do
# `from tesseract_core.runtime.app_cli import jacobian`
globals().update({func.__name__: func for func in endpoints})

tesseract_runtime_cli = copy.deepcopy(tesseract_runtime)
_add_user_commands_to_cli(tesseract_runtime_cli, out_stream=None)
