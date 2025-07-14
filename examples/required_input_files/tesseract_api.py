# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

from pydantic import BaseModel

#
# Schemas
#


class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    pass


#
# Required endpoints
#

# tested with
# tesseract run --input-path=./input required_input_files apply '{"inputs": {}}'


def apply(inputs: InputSchema) -> OutputSchema:
    reqd_files_path = os.environ["TESSERACT_INPUT_PATH"]

    with open(os.path.join(reqd_files_path, "parameters1.json"), "rb") as f:
        data1 = json.load(f)

    assert data1 == {"a": 1.0, "b": 100.0}

    with open(os.path.join(reqd_files_path, "parameters2.json"), "rb") as f:
        data2 = json.load(f)

    assert data2 == {"a": 1.0, "b": 100.0}

    return OutputSchema()
