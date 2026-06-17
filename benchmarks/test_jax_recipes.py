# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for tesseract_core.runtime.jax_recipes' VJP cache.

These benchmarks compare an ``apply -> vjp`` sequence with the cache enabled
against the same sequence with the cache disabled, on a synthetic MLP-style
apply_jit deep enough that the saved forward pass actually shows up against
the per-call overhead of populating/reading the cache.

The accompanying tesseracts in ``examples/`` (e.g. vectoradd_jax) are too
small to demonstrate the win -- their forward pass is microseconds, dwarfed
by the ~hundreds of us cost of ``jax.vjp``'s tracing. This file exists so
the claim made in :func:`set_jax_vjp_cache_size`'s docstring (that enabling
the cache speeds up the standard apply->vjp pattern) has a concrete
artifact backing it.

"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.jax_recipes import (
    jax_apply,
    jax_vjp,
    set_jax_vjp_cache_size,
)

# Shapes chosen near the empirically-measured cache crossover point so the
# benchmark resolves a real difference rather than measuring noise. With
# (depth=32, width=1024) the saved forward pass is ~3-5ms which clearly
# exceeds the ~1ms cost of the cache machinery.
DEPTH = 64
WIDTH = 1024


@pytest.fixture(scope="module")
def synthetic_apply():
    """A synthetic deep MLP apply_jit + matching schema."""
    rng = np.random.default_rng(0)
    weights = [
        jnp.asarray(
            rng.standard_normal((WIDTH, WIDTH)).astype(np.float32) / np.sqrt(WIDTH)
        )
        for _ in range(DEPTH)
    ]

    from pydantic import BaseModel, Field

    class InputSchema(BaseModel):
        x: Differentiable[Array[(WIDTH,), Float32]] = Field(description="input")

    @eqx.filter_jit
    def apply_jit(inputs: dict) -> dict:
        x = inputs["x"]
        for w in weights:
            x = jnp.tanh(w @ x + 0.1 * jnp.sin(x))
        return {"y": x}

    return InputSchema, apply_jit


@pytest.fixture(scope="module")
def warm_inputs(synthetic_apply):
    """Pre-built input + cotangent, with JIT warm-up already triggered.

    Triggers the initial JIT compile so the benchmark measures steady-state
    execution, not first-call compile.
    """
    InputSchema, apply_jit = synthetic_apply
    x = np.random.default_rng(1).standard_normal(WIDTH).astype(np.float32) / np.sqrt(
        WIDTH
    )
    inp = InputSchema(x=x)
    ct = np.ones(WIDTH, dtype=np.float32)
    # Force compile under both cache modes so neither benchmark pays compile cost.
    for cache_size in (0, 1):
        set_jax_vjp_cache_size(cache_size)
        for _ in range(3):
            jax.tree.map(
                lambda a: getattr(a, "block_until_ready", lambda: a)(),
                jax_apply(apply_jit, inp),
            )
            jax.tree.map(
                lambda a: getattr(a, "block_until_ready", lambda: a)(),
                jax_vjp(apply_jit, inp, {"x"}, {"y"}, {"y": ct}),
            )
    set_jax_vjp_cache_size(0)
    return InputSchema, apply_jit, inp, ct


def _apply_then_vjp(apply_jit, inp, ct):
    """One iteration of the workload the cache is designed to speed up."""
    y = jax_apply(apply_jit, inp)
    g = jax_vjp(apply_jit, inp, {"x"}, {"y"}, {"y": ct})
    # block on output so we time real execution rather than dispatch
    jax.block_until_ready(y)
    jax.block_until_ready(g)


def test_apply_then_vjp_cache_off(benchmark, warm_inputs):
    """Baseline: apply + vjp with the cache disabled.

    Every vjp falls through to the freshly-compiled vjp_jit path.
    """
    _, apply_jit, inp, ct = warm_inputs
    set_jax_vjp_cache_size(0)
    try:
        benchmark(_apply_then_vjp, apply_jit, inp, ct)
    finally:
        set_jax_vjp_cache_size(0)


def test_apply_then_vjp_cache_on(benchmark, warm_inputs):
    """Cache enabled: apply fills, vjp reads.

    Should be faster than _off on a forward expensive enough to amortize the
    cache machinery (see DEPTH/WIDTH above).
    """
    _, apply_jit, inp, ct = warm_inputs
    set_jax_vjp_cache_size(1)
    try:
        benchmark(_apply_then_vjp, apply_jit, inp, ct)
    finally:
        set_jax_vjp_cache_size(0)
