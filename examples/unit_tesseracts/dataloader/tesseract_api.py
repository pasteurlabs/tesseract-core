# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.experimental import LazySequence


# input-schema-label-begin
class InputSchema(BaseModel):
    # NOTE: no file references here
    data: LazySequence[Differentiable[Array[(None, 3), Float32]]] = Field(
        description="Data to be processed."
    )


# input-schema-label-end


class OutputSchema(BaseModel):
    data: Sequence[Array[(None, 3), Float32]] = Field(description="Processed data.")
    data_sum: Differentiable[Array[(3,), Float32]] = Field(
        description="Sum of all data samples."
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Process data samples and compute their sum."""
    out_data = []
    data_sum = np.zeros(3)

    # iterating over inputs.data loads its contents one by one
    for data in inputs.data:
        # we only keep processed data here for demonstration
        out_data.append(data * 2)
        data_sum += data.sum(axis=0)

    return OutputSchema(data=out_data, data_sum=data_sum)


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    assert jac_outputs == {"data_sum"}

    # jac_inputs is a set of keys of the form "data.[i]".
    out = {"data_sum": {}}
    for key in jac_inputs:
        key_parts = key.split(".")
        assert key_parts[0] == "data"
        assert len(key_parts) == 2

        idx = int(key_parts[1][1:-1])
        jac = np.ones(3, *inputs.data[idx].shape)
        jac[0, :, 0] = 1.0
        jac[1, :, 1] = 1.0
        jac[2, :, 2] = 1.0
        out["data_sum"][key] = jac

    return out
