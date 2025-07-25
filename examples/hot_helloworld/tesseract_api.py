# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

from tesseract_core import Tesseract


class InputSchema(BaseModel):
    name: str = Field(description="Name of the person you want to greet.")
    tt_ref: str = Field(description="Url of 'helloworld' target Tesseract.")


class OutputSchema(BaseModel):
    greeting: str = Field(description="A greeting!")


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward name to helloworld tesseract and relay its greeting."""
    tess = Tesseract.from_url(inputs.tt_ref)
    greeting = tess.apply({"name": f"{inputs.name}"})["greeting"]
    return OutputSchema(greeting=f"The target Tesseract says: {greeting}")
