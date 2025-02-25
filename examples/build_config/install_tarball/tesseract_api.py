# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import cowsay
from pydantic import BaseModel


class InputSchema(BaseModel):
    message: str = "Hello, Tesseractor!"


class OutputSchema(BaseModel):
    out: str


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(out=cowsay.get_output_string("cow", inputs.message))
