# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from pydantic import BaseModel

from tesseract_core.runtime.experimental import required_file


# define function to load a specific required file
# TODO: is it possible/desired to validate correct format of loaded data (at runtime)
# based on return type of loader function denoted here (using pydantic)?
@required_file(require_writable=False)
def load_parameters(filepath: str = "parameters1.json"):
    """Function to load required files.

    Must have `filepath` as first argument.

    Args:
        filepath (str): Path to file, relative to mounted directory.

    Returns:
        data: Loaded data.
    """
    with open(filepath, "rb") as f:
        data = json.load(f)
    return data


# load required file
data = load_parameters()

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

# executed with
# tesseract run -v "./input:/tesseract/input_data:ro" required_input_files apply '{"inputs": {}}'


def apply(inputs: InputSchema) -> OutputSchema:
    assert data == {"a": 1.0, "b": 100.0}
    return OutputSchema(**data)
