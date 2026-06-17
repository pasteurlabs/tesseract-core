# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tesseract_core.runtime.jax_recipes."""

from collections.abc import Hashable

import numpy as np

from tesseract_core.runtime.jax_recipes import hash_tree


class TestHashTree:
    """Tests for hash_tree -- used as the LRUCache key in the jax-cache recipe."""

    def test_deterministic(self):
        tree = {"a": np.array([1.0, 2.0]), "b": 42}
        assert hash_tree(tree) == hash_tree(tree)

    def test_returns_hashable(self):
        h = hash_tree({"x": np.array([1.0])})
        assert isinstance(h, Hashable)

    def test_different_values_differ(self):
        h1 = hash_tree({"x": np.array([1.0, 2.0])})
        h2 = hash_tree({"x": np.array([1.0, 3.0])})
        assert h1 != h2

    def test_different_shape_with_same_bytes_differ(self):
        # int64 [1, 2, 3, 4] and int64 [[1, 2], [3, 4]] have identical
        # .tobytes() output (32 bytes each). Without shape in the hash they
        # would collide and return the wrong cached vjp_func.
        flat = np.array([1, 2, 3, 4], dtype=np.int64)
        reshaped = flat.reshape(2, 2)
        assert flat.tobytes() == reshaped.tobytes()
        assert hash_tree({"x": flat}) != hash_tree({"x": reshaped})

    def test_different_dtype_with_same_bytes_differ(self):
        # int64 [1, 2, 3, 4] and float64 array reinterpretation share buffer
        # patterns at the byte level for some values. Verify the dtype is part
        # of the key by constructing one explicit case.
        a = np.array([1, 2, 3, 4], dtype=np.int64)
        b = a.view(np.float64)  # same bytes, different dtype interpretation
        assert a.tobytes() == b.tobytes()
        assert hash_tree({"x": a}) != hash_tree({"x": b})

    def test_different_treedef_differs(self):
        h1 = hash_tree({"x": np.array([1.0])})
        h2 = hash_tree({"y": np.array([1.0])})
        assert h1 != h2

    def test_nested_structure(self):
        tree = {"outer": {"inner": np.array([1.0]), "scalar": 2.0}}
        assert hash_tree(tree) == hash_tree(tree)

    def test_scalar_leaves(self):
        assert hash_tree({"i": 42, "f": 3.14, "b": True}) == hash_tree(
            {"i": 42, "f": 3.14, "b": True}
        )
        assert hash_tree({"i": 42}) != hash_tree({"i": 43})

    def test_string_leaf_stable_within_call(self):
        tree1 = {"name": "alpha"}
        tree2 = {"name": "alpha"}
        assert hash_tree(tree1) == hash_tree(tree2)
        assert hash_tree({"name": "alpha"}) != hash_tree({"name": "beta"})

    def test_bytes_leaf(self):
        assert hash_tree({"k": b"abc"}) == hash_tree({"k": b"abc"})
        assert hash_tree({"k": b"abc"}) != hash_tree({"k": b"abd"})

    def test_empty_tree(self):
        h = hash_tree({})
        assert isinstance(h, Hashable)


# ---------------------------------------------------------------------------
# Integration test: cache-on and cache-off must produce identical results.
#
# The per-container test_cases in examples/vectoradd_jax/ exercise the
# fallback (cache-miss) path only: each `tesseract run` invocation starts
# with an empty cache so vjp always falls through to vjp_jit. The tests below
# share a single Python process, so cache fill in apply() actually feeds the
# subsequent vjp() call -- which is the case the cache is built for.
# ---------------------------------------------------------------------------


# Minimal pydantic schema + apply_jit, kept inline so the test doesn't
# depend on the vectoradd_jax example wiring.
def _build_api():
    import equinox as eqx
    import jax.numpy as jnp
    from pydantic import BaseModel, Field

    from tesseract_core.runtime import Array, Differentiable, Float32

    class Vec(BaseModel):
        v: Differentiable[Array[(None,), Float32]] = Field(description="vec")
        s: Differentiable[Float32] = Field(default=1.0, description="scale")

    class InputSchema(BaseModel):
        a: Vec
        b: Vec

    @eqx.filter_jit
    def apply_jit(inputs):
        a = inputs["a"]["s"] * inputs["a"]["v"]
        b = inputs["b"]["s"] * inputs["b"]["v"]
        # Genuinely nonlinear so vjp residuals are non-trivial.
        return {"y": jnp.tanh(a + b) * (a - b)}

    return InputSchema, apply_jit


def _make_inputs(InputSchema, offset=0.0):
    return InputSchema(
        a={"v": np.array([1.0, 2.0, 3.0], dtype=np.float32) + offset, "s": 1.5},
        b={"v": np.array([0.5, 1.0, 1.5], dtype=np.float32) + offset, "s": -0.25},
    )


class TestCacheCorrectness:
    """Cache-on must produce identical outputs to cache-off.

    Run after :class:`TestHashTree` so the hash primitive failures (if any)
    surface first with simpler messages.
    """

    def _run_workflow(self, cache_size):
        """Run apply(x) -> vjp(x) -> vjp(x') and return all three outputs.

        With cache_size=1 this exercises both branches of jax_vjp: the second
        call hits the cache (same input as the apply that filled it), the
        third misses (different input) and falls through to vjp_jit.
        """
        from tesseract_core.runtime.jax_recipes import (
            jax_apply,
            jax_vjp,
            set_jax_vjp_cache_size,
        )

        set_jax_vjp_cache_size(cache_size)
        InputSchema, apply_jit = _build_api()
        inp_a = _make_inputs(InputSchema, offset=0.0)
        inp_b = _make_inputs(InputSchema, offset=0.5)

        ct = np.ones(3, dtype=np.float32)
        y = jax_apply(apply_jit, inp_a)
        # Second call on the SAME input as apply -> cache hit when enabled.
        grad_hit = jax_vjp(apply_jit, inp_a, {"a.v", "b.v"}, {"y"}, {"y": ct})
        # Third call on a DIFFERENT input -> cache miss -> fallback path.
        grad_miss = jax_vjp(apply_jit, inp_b, {"a.v", "b.v"}, {"y"}, {"y": ct})

        # Reset to disabled so this test doesn't leak module-level state.
        set_jax_vjp_cache_size(0)
        return y, grad_hit, grad_miss

    def test_cache_on_matches_cache_off(self):
        # Run with cache disabled (every vjp through the fallback path).
        y_off, grad_hit_off, grad_miss_off = self._run_workflow(0)
        # Run with cache enabled (first vjp hits cache; second misses).
        y_on, grad_hit_on, grad_miss_on = self._run_workflow(1)

        np.testing.assert_allclose(np.asarray(y_off["y"]), np.asarray(y_on["y"]))
        for k in grad_hit_off:
            # Cache-hit branch must produce the same result as the fallback.
            np.testing.assert_allclose(
                np.asarray(grad_hit_off[k]), np.asarray(grad_hit_on[k])
            )
            # Cache-miss branch must also match (sanity: fallback is identical
            # whether cache was on or off, since on a miss we re-enter the
            # same vjp_jit codepath).
            np.testing.assert_allclose(
                np.asarray(grad_miss_off[k]), np.asarray(grad_miss_on[k])
            )
        # Sanity-check that the two inputs really did exercise different
        # gradients -- otherwise the hit/miss distinction would be vacuous.
        for k in grad_hit_off:
            assert not np.allclose(
                np.asarray(grad_hit_off[k]), np.asarray(grad_miss_off[k])
            ), "hit and miss inputs produced identical gradients; test is vacuous"

    def test_cache_hit_actually_engages(self):
        # Sanity check that the cache fills and is read back, not silently
        # bypassed.
        from tesseract_core.runtime.jax_recipes import (
            jax_apply,
            jax_vjp,
            set_jax_vjp_cache_size,
        )

        set_jax_vjp_cache_size(1)
        try:
            from tesseract_core.runtime import jax_recipes

            assert jax_recipes.jax_vjp_cache is not None
            assert jax_recipes.jax_vjp_cache.size == 0

            InputSchema, apply_jit = _build_api()
            inp = _make_inputs(InputSchema)

            jax_apply(apply_jit, inp)
            assert jax_recipes.jax_vjp_cache.size == 1

            # vjp on the SAME inputs should hit cache (get is non-destructive).
            jax_vjp(
                apply_jit,
                inp,
                {"a.v"},
                {"y"},
                {"y": np.ones(3, dtype=np.float32)},
            )
            assert jax_recipes.jax_vjp_cache.size == 1
        finally:
            set_jax_vjp_cache_size(0)
