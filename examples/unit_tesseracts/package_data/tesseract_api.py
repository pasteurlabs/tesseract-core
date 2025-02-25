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


def apply(inputs: InputSchema) -> OutputSchema:
    dirname = os.path.dirname(__file__)
    parameter_file = os.path.join(dirname, "parameters.json")

    # check file exists
    assert os.path.exists(parameter_file)

    # check parameters
    with open(parameter_file) as file:
        data = json.load(file)

    assert data == {"a": 1.0, "b": 100.0}

    return OutputSchema()
