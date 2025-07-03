# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json

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
# tesseract run required_input_files --input-dir ./input/ apply '{"inputs": {}}'


def apply(inputs: InputSchema) -> OutputSchema:
    # print cwd
    with open("/tesseract-input/parameters1.json", "rb") as f:
        data1 = json.load(f)

    assert data1 == {"a": 1.0, "b": 100.0}

    with open("/tesseract-input/parameters2.json", "rb") as f:
        data2 = json.load(f)

    assert data2 == {"a": 1.0, "b": 100.0}

    return OutputSchema()
