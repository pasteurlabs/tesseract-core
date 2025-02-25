# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from tesseract_core.runtime import Differentiable, Float32

#
# Schemas
#


class InputSchema(BaseModel):
    # Make sure that array types work for older Python versions
    foo: Differentiable[Float32]


class OutputSchema(BaseModel):
    bar: Differentiable[Float32]


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    # Ensure that the Python version is what we expect (3.9)
    import sys

    assert sys.version_info[:2] == (3, 9)
    return OutputSchema(bar=0)


#
# Optional endpoints
#


def abstract_eval(abstract_inputs):
    return {"bar": {"shape": (), "dtype": "float32"}}


def jacobian(
    inputs: InputSchema, jac_inputs: list[str], jac_outputs: list[str]
) -> OutputSchema:
    return {out: 0.0 for out in jac_outputs}
