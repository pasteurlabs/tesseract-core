# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field


# SCHEMA START
class InputSchema(BaseModel):
    name: str = Field(description="Name of the person you want to greet.")


class OutputSchema(BaseModel):
    greeting: str = Field(description="A greeting!")


# SCHEMA END


# APPLY START
def apply(inputs: InputSchema) -> OutputSchema:
    """Greet a person whose name is given as input."""
    return OutputSchema(greeting=f"Hello {inputs.name}!")


# APPLY END
