# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from goodbyeworld import ungreet
from helloworld import greet
from pydantic import BaseModel, Field


class InputSchema(BaseModel):
    name: str = Field(description="Name of the person you want to greet.")


class OutputSchema(BaseModel):
    message: str = Field(description="A message!")


def apply(inputs: InputSchema) -> OutputSchema:
    """Greets and ungreets a person whose name is given as input."""
    message = f"{greet(inputs.name)}\n{ungreet(inputs.name)}"
    return OutputSchema(message=message)
