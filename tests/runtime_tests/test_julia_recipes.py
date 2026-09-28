# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Python side of tesseract_core.runtime.julia_recipes."""

import numpy as np
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float32, Float64
from tesseract_core.runtime.julia_recipes import _flatten_inputs


class Block(BaseModel):
    main: Differentiable[Array[(None,), Float64]]
    sup: Differentiable[Array[(None,), Float32]] | None = None


class Inputs(BaseModel):
    blocks: list[Block | None]
    b: Differentiable[Array[(None,), Float64]]
    block_sizes: list[int]
    params: dict[str, float]
    label: str


def test_flatten_inputs_splits_leaves_by_differentiability():
    inputs = Inputs(
        blocks=[
            Block(main=np.array([1.0, 2.0]), sup=np.array([3.0], dtype=np.float32)),
            None,
            Block(main=np.array([4.0])),
        ],
        b=np.array([5.0, 6.0, 7.0]),
        block_sizes=[2, 1],
        params={"alpha": 0.5},
        label="demo",
    )
    flat = _flatten_inputs(inputs)

    diff = dict(zip(flat.diff_paths, flat.diff_args, strict=True))
    assert diff.keys() == {"blocks.[0].main", "blocks.[0].sup", "blocks.[2].main", "b"}
    np.testing.assert_array_equal(diff["blocks.[0].main"], [1.0, 2.0])
    np.testing.assert_array_equal(diff["blocks.[0].sup"], [3.0])
    np.testing.assert_array_equal(diff["blocks.[2].main"], [4.0])
    np.testing.assert_array_equal(diff["b"], [5.0, 6.0, 7.0])
    # Julia receives Vector{Float64}, whatever the schema dtype
    assert all(arr.dtype == np.float64 for arr in flat.diff_args)

    non_diff = dict(zip(flat.non_diff_paths, flat.non_diff_args, strict=True))
    assert non_diff == {
        "block_sizes.[0]": 2,
        "block_sizes.[1]": 1,
        "params.{alpha}": 0.5,
        "label": "demo",
    }
