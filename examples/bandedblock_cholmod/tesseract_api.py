# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract wrapping a sparse CHOLMOD solver for SPD block systems with Enzyme AD.

Solves A * x = b where A is a symmetric positive definite matrix with block
structure. Each nonzero block is tridiagonal (or diagonal), stored compactly
as up to 3 diagonal vectors. Zero blocks are represented as None.

The matrix is assembled as a sparse SparseMatrixCSC and solved via
SuiteSparse CHOLMOD — a sparse Cholesky factorization not available in JAX.
Enzyme provides both forward-mode (JVP) and reverse-mode (VJP) automatic
differentiation through the LinearSolve.jl Enzyme extension, which
implements the implicit function theorem adjoint without differentiating
through CHOLMOD internals.

Example arrow structure (SPD, blocks [0][1]=[1][0]^T, [0][2]=[2][0]^T):

    ┌──────────────────────┐
    │  A00  │  A01  │  A02 │
    ├───────┼───────┼──────┤
    │  A10  │  A11  │   0  │
    ├───────┼───────┼──────┤
    │  A20  │   0   │  A22 │
    └──────────────────────┘
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from juliacall import Main as jl
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.schema_generation import (
    DICT_INDEX_SENTINEL,
    SEQ_INDEX_SENTINEL,
    get_all_model_path_patterns,
)
from tesseract_core.runtime.schema_types import is_differentiable
from tesseract_core.runtime.testing.finite_differences import expand_path_pattern
from tesseract_core.runtime.tree_transforms import get_at_path

# ---------------------------------------------------------------------------
# Julia setup — load solver and Enzyme wrappers from .jl files
# ---------------------------------------------------------------------------

_here = Path(__file__).parent
jl.include(str(_here / "apply.jl"))
jl.include(str(_here / "enzyme_wrappers.jl"))


# ---------------------------------------------------------------------------
# Schemata
# ---------------------------------------------------------------------------


class TridiagBlock(BaseModel):
    """A tridiagonal (or diagonal) block stored as diagonal vectors."""

    sub: Differentiable[Array[(None,), Float64]] | None = Field(
        default=None, description="Sub-diagonal, length n-1. None for diagonal blocks."
    )
    main: Differentiable[Array[(None,), Float64]] = Field(
        description="Main diagonal, length n."
    )
    sup: Differentiable[Array[(None,), Float64]] | None = Field(
        default=None,
        description="Super-diagonal, length n-1. None for diagonal blocks.",
    )


class InputSchema(BaseModel):
    blocks: list[list[TridiagBlock | None]] = Field(
        description="Block structure as nested list. None = zero block. "
        "Example for 3x3 arrow: "
        "[[A00, A01, A02], [A10, A11, None], [A20, None, A22]]",
    )
    b: Differentiable[Array[(None,), Float64]] = Field(
        description="Right-hand side vector, length sum(block_sizes).",
    )
    block_sizes: list[int] = Field(
        description="Size of each block group.",
    )


class OutputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float64]] = Field(
        description="Solution vector, length sum(block_sizes).",
    )


# ---------------------------------------------------------------------------
# Schema-driven path extraction (generic, no need to modify)
# ---------------------------------------------------------------------------


def _path_tuple_to_pattern(path_tuple: tuple) -> str:
    """Convert a path tuple with sentinels to a pattern string with [] and {}."""
    parts = []
    for part in path_tuple:
        if part is SEQ_INDEX_SENTINEL:
            parts.append("[]")
        elif part is DICT_INDEX_SENTINEL:
            parts.append("{}")
        else:
            parts.append(str(part))
    return ".".join(parts)


_ALL_PATTERNS = [
    _path_tuple_to_pattern(p) for p in get_all_model_path_patterns(InputSchema)
]
_DIFF_PATTERNS = [
    _path_tuple_to_pattern(p)
    for p in get_all_model_path_patterns(InputSchema, filter_fn=is_differentiable)
]


def _expand_inputs(
    inputs_dict: dict,
) -> tuple[list[np.ndarray], list[str], list[Any], list[str]]:
    """Expand schema paths against concrete inputs, split into diff and non-diff.

    Returns:
        diff_args: list of differentiable arrays (always numpy float64)
        diff_paths: corresponding Tesseract paths
        non_diff_args: list of non-differentiable leaf values
        non_diff_paths: corresponding Tesseract paths
    """
    all_paths = []
    for pattern in _ALL_PATTERNS:
        all_paths.extend(expand_path_pattern(pattern, inputs_dict))

    diff_path_set = set()
    for pattern in _DIFF_PATTERNS:
        diff_path_set.update(expand_path_pattern(pattern, inputs_dict))

    diff_args, diff_paths = [], []
    non_diff_args, non_diff_paths = [], []

    for path in all_paths:
        value = get_at_path(inputs_dict, path)
        if isinstance(value, (dict, list)):
            continue

        if path in diff_path_set:
            diff_args.append(np.asarray(value, dtype=np.float64))
            diff_paths.append(path)
        else:
            non_diff_args.append(value)
            non_diff_paths.append(path)

    return diff_args, diff_paths, non_diff_args, non_diff_paths


# ---------------------------------------------------------------------------
# Required endpoints — modify evaluate and abstract_eval for your solver
# ---------------------------------------------------------------------------


def evaluate(inputs_dict: dict) -> dict:
    """Call the Julia CHOLMOD solver."""
    diff_args, diff_paths, non_diff_args, non_diff_paths = _expand_inputs(inputs_dict)
    result = jl.apply_jl(diff_args, non_diff_args, diff_paths, non_diff_paths)
    return {"x": np.asarray(result)}


def apply(inputs: InputSchema) -> OutputSchema:
    return evaluate(inputs.model_dump())


def abstract_eval(abstract_inputs):
    """Output x has length sum(block_sizes)."""
    inp = abstract_inputs.model_dump()
    total_size = sum(inp["block_sizes"])
    return {
        "x": {"shape": [total_size], "dtype": "float64"},
    }


# ---------------------------------------------------------------------------
# Gradient endpoints (generic, no need to modify)
# ---------------------------------------------------------------------------


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    inputs_dict = inputs.model_dump()
    diff_args, diff_paths, non_diff_args, non_diff_paths = _expand_inputs(inputs_dict)

    tangent_args = []
    for k, path in enumerate(diff_paths):
        if path in jvp_inputs:
            tangent_args.append(np.asarray(tangent_vector[path], dtype=np.float64))
        else:
            tangent_args.append(np.zeros_like(diff_args[k]))

    jvp_out = np.asarray(
        jl.enzyme_jvp(
            jl.apply_jl,
            diff_args,
            non_diff_args,
            diff_paths,
            non_diff_paths,
            tangent_args,
        )
    )
    return {p: jvp_out for p in jvp_outputs}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    inputs_dict = inputs.model_dump()
    diff_args, diff_paths, non_diff_args, non_diff_paths = _expand_inputs(inputs_dict)

    cotangent_vector = {key: cotangent_vector[key] for key in vjp_outputs}
    combined_cotangent = sum(
        np.asarray(v, dtype=np.float64) for v in cotangent_vector.values()
    )

    grads = jl.enzyme_vjp(
        jl.apply_jl,
        diff_args,
        non_diff_args,
        diff_paths,
        non_diff_paths,
        combined_cotangent,
    )

    result = {}
    for k, path in enumerate(diff_paths):
        if path in vjp_inputs:
            result[path] = np.asarray(grads[k])
    return result
