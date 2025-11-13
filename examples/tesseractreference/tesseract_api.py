# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

from tesseract_core.runtime.experimental import TesseractReference


class InputSchema(BaseModel):
    target: TesseractReference = Field(description="Tesseract to call.")


class OutputSchema(BaseModel):
    result: str = Field(description="Result of the Tesseract calls.")


def apply(inputs: InputSchema) -> OutputSchema:
    result = inputs.target.apply({"name": "Alice"})["greeting"]
    result += " " + inputs.target.apply({"name": "Bob"})["greeting"]

    return OutputSchema(result=result)
