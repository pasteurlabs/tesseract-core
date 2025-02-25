# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# This throws an error if VTK is not installed
import pyvista as pv  # noqa: F401
from pydantic import BaseModel


class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    pass


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema()
