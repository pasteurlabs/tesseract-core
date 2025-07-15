# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from pydantic import BaseModel

from tesseract_core.runtime.experimental import IS_BUILDING, require_file

#
# Schemas
#

param_file = require_file("parameters1.json")
if not IS_BUILDING:
    with open(param_file, "rb") as f:
        data = json.load(f)
else:
    # Simulate the data for building purposes
    data = {"a": 1.0, "b": 100.0}


class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    a: float
    b: float


#
# Required endpoints
#

# tested with
# tesseract run required_input_files --input-dir ./input/ apply '{"inputs": {}}'


def apply(inputs: InputSchema) -> OutputSchema:
    assert data == {"a": 1.0, "b": 100.0}
    return OutputSchema(**data)
