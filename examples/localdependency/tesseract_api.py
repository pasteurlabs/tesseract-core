# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import cowsay
from pydantic import BaseModel, Field


class InputSchema(BaseModel):
    message: str = Field(description="A message for cowsay.")


class OutputSchema(BaseModel):
    cowsays: str = Field(description="The cowsay output string.")


def apply(inputs: InputSchema) -> OutputSchema:
    """Greet a person whose name is given as input."""
    return OutputSchema(cowsays=cowsay.get_output_string("cow", inputs.message))
