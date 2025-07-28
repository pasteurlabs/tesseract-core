# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

# NOTE: This requires adding `tesseract_core` as a dependency to `tesseract_requirements.txt`
from tesseract_core import Tesseract


class InputSchema(BaseModel):
    name: str = Field(description="Name of the person you want to greet.")
    helloworld_tesseract_url: str = Field(
        description="Url of 'helloworld' target Tesseract."
    )


class OutputSchema(BaseModel):
    greeting: str = Field(description="A greeting!")


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward name to helloworld tesseract and relay its greeting."""
    tess = Tesseract.from_url(inputs.helloworld_tesseract_url)
    greeting = tess.apply({"name": f"{inputs.name}"})["greeting"]
    return OutputSchema(greeting=f"The target Tesseract says: {greeting}")
