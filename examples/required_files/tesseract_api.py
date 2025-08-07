# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from pydantic import BaseModel

from tesseract_core.runtime.experimental import IS_BUILDING, require_file

# required files must be relative to --input-path
param_filepath = require_file("parameters1.json")
if not IS_BUILDING:
    with open(param_filepath, "rb") as f:
        data = json.load(f)
else:
    # No data is present during build time
    data = None

#
# Schemas
#


class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    a: float
    b: float


#
# Required endpoints
#

# execute with
# tesseract run --input-path input required_input_files apply '{"inputs": {}}'


def apply(inputs: InputSchema) -> OutputSchema:
    assert data == {"a": 1.0, "b": 100.0}
    return OutputSchema(**data)
