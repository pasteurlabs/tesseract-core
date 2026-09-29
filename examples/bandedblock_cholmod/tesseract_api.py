# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract wrapping a sparse CHOLMOD solver for SPD block systems with Enzyme AD.

Solves A x = b, where A is symmetric positive definite with block structure.
Each nonzero block is tridiagonal (or diagonal) and stored as up to three
diagonal vectors. Zero blocks are None.

The solve uses SuiteSparse CHOLMOD, a sparse Cholesky factorization that JAX
does not provide. Forward- and reverse-mode gradients come from the Enzyme
extension of LinearSolve.jl, which applies the implicit function theorem
instead of differentiating through CHOLMOD internals.

Example arrow structure (SPD, blocks [0][1]=[1][0]^T, [0][2]=[2][0]^T):

    ┌──────────────────────┐
    │  A00  │  A01  │  A02 │
    ├───────┼───────┼──────┤
    │  A10  │  A11  │   0  │
    ├───────┼───────┼──────┤
    │  A20  │   0   │  A22 │
    └──────────────────────┘
"""

from pathlib import Path
from typing import Any

from juliacall import Main as jl
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.julia_recipes import julia_apply, julia_jvp, julia_vjp

jl.include(str(Path(__file__).parent / "julia" / "apply.jl"))


#
# Schemata
#


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
        description="Size of each block row (and column).",
    )


class OutputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float64]] = Field(
        description="Solution vector, length sum(block_sizes).",
    )


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**julia_apply(jl.apply_jl, inputs))


def abstract_eval(abstract_inputs):
    """Output x has length sum(block_sizes)."""
    inp = abstract_inputs.model_dump()
    total_size = sum(inp["block_sizes"])
    return {
        "x": {"shape": [total_size], "dtype": "float64"},
    }


#
# Enzyme-handled gradient endpoints
#


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return julia_jvp(jl.apply_jl, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return julia_vjp(jl.apply_jl, inputs, vjp_inputs, vjp_outputs, cotangent_vector)
