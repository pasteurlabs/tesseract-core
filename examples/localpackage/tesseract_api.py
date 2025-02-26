# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from helloworld import greet
from pydantic import BaseModel, Field


class InputSchema(BaseModel):
    name: str = Field(description="Name of the person you want to greet.")


class OutputSchema(BaseModel):
    greeting: str = Field(description="A greeting!")


def apply(inputs: InputSchema) -> OutputSchema:
    """Greet a person whose name is given as input."""
    return OutputSchema(greeting=greet(inputs.name))
