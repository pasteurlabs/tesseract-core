# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities for testing runtime functionality."""

from collections.abc import Callable

from pydantic import BaseModel


def get_input_schema(endpoint_function: Callable) -> type[BaseModel]:
    """Get the input schema of an endpoint function."""
    schema = endpoint_function.__annotations__["payload"]
    if not issubclass(schema, BaseModel):
        raise AssertionError(f"Expected BaseModel, got {schema}")
    return schema


def get_output_schema(endpoint_function: Callable) -> type[BaseModel]:
    """Get the output schema of an endpoint function."""
    schema = endpoint_function.__annotations__["return"]
    if not issubclass(schema, BaseModel):
        raise AssertionError(f"Expected BaseModel, got {schema}")
    return schema
