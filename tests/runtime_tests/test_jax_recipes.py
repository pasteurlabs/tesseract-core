# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tesseract_core.runtime.jax_recipes."""

import os
import subprocess
import sys
import textwrap
from collections.abc import Hashable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32, jax_recipes
from tesseract_core.runtime.experimental import set_jax_vjp_cache_size
from tesseract_core.runtime.jax_recipes import (
    _cache_key,
    as_jax_arrays,
    jax_apply,
    jax_jacobian,
    jax_jvp,
    jax_vjp,
)


def test_as_jax_arrays_preserves_structure_and_static_leaves():
    host = np.array([1.0, 2.0], dtype=np.float32)
    host.setflags(write=False)
    existing = jnp.asarray([3.0])
    tree = {"arrays": [host, np.float32(4), existing], "static": ("tag", None, 2)}
    result = as_jax_arrays(tree)
    assert all(isinstance(x, jax.Array) for x in result["arrays"])
    np.testing.assert_array_equal(result["arrays"][0], host)
    assert result["arrays"][1].item() == 4
    assert result["arrays"][2] is existing
    assert result["static"] == tree["static"]
    assert tree["arrays"][0] is host


def test_jax_apply_converts_readonly_host_inputs():
    class Inputs(BaseModel):
        x: Array[(2,), Float32]

    host = np.array([1.0, 2.0], dtype=np.float32)
    host.setflags(write=False)

    def apply(inputs):
        assert isinstance(inputs["x"], jax.Array)
        return {"y": inputs["x"] * 2}

    np.testing.assert_array_equal(jax_apply(apply, Inputs(x=host))["y"], [2, 4])


def test_jax_endpoints_convert_device_inputs(monkeypatch):
    class DeviceInput:
        # Only the metadata/dispatch boundary is simulated; the differentiation
        # below runs through real JAX arrays on CPU.
        def __init__(self, values):
            self.__cuda_array_interface__ = {
                "shape": (2,),
                "typestr": "<f4",
                "data": (0, False),
                "version": 3,
            }
            self.array = jnp.asarray(values, dtype=jnp.float32)

    class Inputs(BaseModel):
        x: Any

    monkeypatch.setattr(jnp, "from_dlpack", lambda x: x.array)
    inputs = Inputs(x=DeviceInput([2, 3]))
    vector = DeviceInput([1, 1])
    apply = lambda x: {"y": x["x"] ** 2}
    set_jax_vjp_cache_size(0)
    np.testing.assert_array_equal(jax_apply(apply, inputs)["y"], [4, 9])
    np.testing.assert_array_equal(
        jax_jvp(apply, inputs, {"x"}, {"y"}, {"x": vector})["y"], [4, 6]
    )
    np.testing.assert_array_equal(
        jax_jacobian(apply, inputs, {"x"}, {"y"})["y"]["x"], [[4, 0], [0, 6]]
    )
    for cache_size in (0, 1):
        set_jax_vjp_cache_size(cache_size)
        try:
            jax_apply(apply, inputs)
            np.testing.assert_array_equal(
                jax_vjp(apply, inputs, {"x"}, {"y"}, {"y": vector})["x"], [4, 6]
            )
        finally:
            set_jax_vjp_cache_size(0)


class TestCacheKey:
    """Tests for _cache_key -- the LRUCache key used by the jax-cache recipe."""

    def test_deterministic(self):
        tree = {"a": np.array([1.0, 2.0]), "b": 42}
        assert _cache_key(tree) == _cache_key(tree)

    def test_returns_hashable(self):
        h = _cache_key({"x": np.array([1.0])})
        assert isinstance(h, Hashable)

    def test_different_values_differ(self):
        h1 = _cache_key({"x": np.array([1.0, 2.0])})
        h2 = _cache_key({"x": np.array([1.0, 3.0])})
        assert h1 != h2

    def test_different_shape_with_same_bytes_differ(self):
        # int64 [1, 2, 3, 4] and int64 [[1, 2], [3, 4]] have identical
        # .tobytes() output (32 bytes each). Without shape in the hash they
        # would collide and return the wrong cached vjp_func.
        flat = np.array([1, 2, 3, 4], dtype=np.int64)
        reshaped = flat.reshape(2, 2)
        assert flat.tobytes() == reshaped.tobytes()
        assert _cache_key({"x": flat}) != _cache_key({"x": reshaped})

    def test_different_dtype_with_same_bytes_differ(self):
        # int64 [1, 2, 3, 4] and float64 array reinterpretation share buffer
        # patterns at the byte level for some values. Verify the dtype is part
        # of the key by constructing one explicit case.
        a = np.array([1, 2, 3, 4], dtype=np.int64)
        b = a.view(np.float64)  # same bytes, different dtype interpretation
        assert a.tobytes() == b.tobytes()
        assert _cache_key({"x": a}) != _cache_key({"x": b})

    def test_different_treedef_differs(self):
        h1 = _cache_key({"x": np.array([1.0])})
        h2 = _cache_key({"y": np.array([1.0])})
        assert h1 != h2

    def test_nested_structure(self):
        tree = {"outer": {"inner": np.array([1.0]), "scalar": 2.0}}
        assert _cache_key(tree) == _cache_key(tree)

    def test_scalar_leaves(self):
        assert _cache_key({"i": 42, "f": 3.14, "b": True}) == _cache_key(
            {"i": 42, "f": 3.14, "b": True}
        )
        assert _cache_key({"i": 42}) != _cache_key({"i": 43})

    def test_colliding_scalar_hashes_stay_distinct(self):
        # CPython reserves -1 as an error sentinel, so hash(-1) == hash(-2).
        # A key collapsed with hash() cannot tell these two apart, and the
        # LRUCache dict has no second key to compare against, so a Tesseract
        # taking an integer parameter would serve the wrong backward pass.
        assert hash(-1) == hash(-2)
        assert _cache_key({"i": -1}) != _cache_key({"i": -2})

    def test_numerically_equal_leaves_of_different_types_stay_distinct(self):
        # The other direction: 1 == 1.0 == True compare equal *and* hash
        # equal, so the value alone is not enough to discriminate a leaf whose
        # field is typed as a union.
        assert hash(1) == hash(1.0) == hash(True)
        keys = [_cache_key({"x": v}) for v in (1, 1.0, True)]
        assert len(set(keys)) == 3

    def test_string_leaf_stable_within_call(self):
        tree1 = {"name": "alpha"}
        tree2 = {"name": "alpha"}
        assert _cache_key(tree1) == _cache_key(tree2)
        assert _cache_key({"name": "alpha"}) != _cache_key({"name": "beta"})

    def test_bytes_leaf(self):
        assert _cache_key({"k": b"abc"}) == _cache_key({"k": b"abc"})
        assert _cache_key({"k": b"abc"}) != _cache_key({"k": b"abd"})

    def test_empty_tree(self):
        h = _cache_key({})
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
    class Vec(BaseModel):
        v: Differentiable[Array[(None,), Float32]] = Field(description="vec")
        s: Differentiable[Float32] = Field(default=1.0, description="scale")

    class InputSchema(BaseModel):
        a: Vec
        b: Vec
        # Non-array leaves: an int reaches the cache key, a str is a type
        # jax.vjp refuses to trace. Both are static.
        norm_ord: int = Field(default=2, description="order of norm")
        mode: str = Field(default="fast", description="non-JAX leaf")

    @eqx.filter_jit
    def apply_jit(inputs):
        a = inputs["a"]["s"] * inputs["a"]["v"]
        b = inputs["b"]["s"] * inputs["b"]["v"]
        # Genuinely nonlinear so vjp residuals are non-trivial.
        y = jnp.tanh(a + b) * (a - b)
        scale = 2.0 if inputs["mode"] == "fast" else 3.0
        return {"y": y * scale, "n": jnp.linalg.norm(y, ord=inputs["norm_ord"])}

    return InputSchema, apply_jit


def _make_inputs(InputSchema, offset=0.0, norm_ord=2):
    return InputSchema(
        a={"v": np.array([1.0, 2.0, 3.0], dtype=np.float32) + offset, "s": 1.5},
        b={"v": np.array([0.5, 1.0, 1.5], dtype=np.float32) + offset, "s": -0.25},
        norm_ord=norm_ord,
        mode="fast",
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
        set_jax_vjp_cache_size(1)
        try:
            assert jax_recipes._jax_vjp_cache is not None
            assert jax_recipes._jax_vjp_cache.size == 0

            InputSchema, apply_jit = _build_api()
            inp = _make_inputs(InputSchema)

            jax_apply(apply_jit, inp)
            assert jax_recipes._jax_vjp_cache.size == 1

            # vjp on the SAME inputs should hit cache (get is non-destructive).
            jax_vjp(
                apply_jit,
                inp,
                {"a.v"},
                {"y"},
                {"y": np.ones(3, dtype=np.float32)},
            )
            assert jax_recipes._jax_vjp_cache.size == 1
        finally:
            set_jax_vjp_cache_size(0)


class TestCacheWithNonArrayInputs:
    """Non-array leaves must not collide in the key, or break the forward pass.

    Both paths are invisible to ``test_cache_on_matches_cache_off``: array
    leaves are keyed on their bytes and discriminate correctly, and that test
    never varies a non-array leaf between the ``apply`` and the ``vjp``.
    """

    @staticmethod
    def _vjp(cache_size, norm_ord, prime_with=None):
        """Optionally prime the cache at ``prime_with``, then vjp at ``norm_ord``."""
        InputSchema, apply_jit = _build_api()
        set_jax_vjp_cache_size(cache_size)
        try:
            if prime_with is not None:
                jax_apply(apply_jit, _make_inputs(InputSchema, norm_ord=prime_with))
            inp = _make_inputs(InputSchema, norm_ord=norm_ord)
            ct = {"y": np.ones(3, dtype=np.float32)}
            return jax_vjp(apply_jit, inp, {"a.v", "a.s"}, {"y"}, ct)
        finally:
            set_jax_vjp_cache_size(0)

    def test_int_input_does_not_serve_a_colliding_entry(self):
        # hash(-1) == hash(-2), so priming at norm_ord=-1 used to satisfy a
        # lookup at norm_ord=-2 and return the wrong gradient.
        expected = self._vjp(0, norm_ord=-2)
        got = self._vjp(4, norm_ord=-2, prime_with=-1)
        for k in expected:
            np.testing.assert_allclose(
                np.asarray(got[k]), np.asarray(expected[k]), rtol=1e-6
            )

    def test_non_jax_input_does_not_break_apply(self):
        # jax.vjp traces every leaf it is handed, so the str leaf aborted the
        # call outright and merely enabling the cache broke the Tesseract.
        InputSchema, apply_jit = _build_api()
        set_jax_vjp_cache_size(4)
        try:
            out = jax_apply(apply_jit, _make_inputs(InputSchema))
        finally:
            set_jax_vjp_cache_size(0)
        assert np.all(np.isfinite(np.asarray(out["y"])))

    def test_scalar_float_field_keeps_its_gradient(self):
        # A scalar Differentiable[Float32] validates to a numpy scalar, so it
        # must stay on the dynamic side of the input partition.
        expected = self._vjp(0, norm_ord=2)
        got = self._vjp(4, norm_ord=2, prime_with=2)
        np.testing.assert_allclose(
            np.asarray(got["a.s"]), np.asarray(expected["a.s"]), rtol=1e-6
        )
        assert np.asarray(expected["a.s"]).item() != 0.0


class TestCacheWithDeviceArrays:
    """Array leaves are compared where they live instead of copied to the host.

    Every JAX array takes the same path, so these run on CPU-only machines too.
    """

    @staticmethod
    def _store(tree, value):
        jax_recipes._jax_vjp_cache.put(_cache_key(tree), value)

    @staticmethod
    def _lookup(tree):
        return jax_recipes._jax_vjp_cache.get(_cache_key(tree))

    def test_key_fingerprints_arrays_without_copying_them(self):
        a = {"x": jnp.array([1.0, 2.0])}
        b = {"x": jnp.array([1.0, 3.0])}
        # The fingerprint separates keys by contents before __eq__ compares them.
        assert hash(_cache_key(a)) != hash(_cache_key(b))
        assert _cache_key(a) != _cache_key(b)
        assert _cache_key(a) == _cache_key({"x": jnp.array([1.0, 2.0])})
        assert _cache_key(a) != _cache_key({"x": jnp.array([1.0, 2.0, 3.0])})
        assert _cache_key(a)[-1].arrays[0] is a["x"]

    def test_lookup_compares_contents(self):
        jax_recipes._set_jax_vjp_cache_size(1)
        try:
            x = {"x": jnp.array([0.0, 1.0, jnp.nan])}
            self._store(x, "cached")
            assert self._lookup({"x": jnp.array([0.0, 1.0, jnp.nan])}) == "cached"
            # Comparison is bitwise, so -0.0 is a new input.
            assert self._lookup({"x": jnp.array([-0.0, 1.0, jnp.nan])}) is None
            assert self._lookup({"x": jnp.array([0.0, 2.0, jnp.nan])}) is None
        finally:
            jax_recipes._set_jax_vjp_cache_size(0)

    def test_same_shape_inputs_are_cached_side_by_side(self):
        jax_recipes._set_jax_vjp_cache_size(2)
        try:
            first = {"x": jnp.array([1.0, 2.0])}
            second = {"x": jnp.array([3.0, 4.0])}
            self._store(first, "first")
            self._store(second, "second")
            assert jax_recipes._jax_vjp_cache.size == 2
            assert self._lookup(first) == "first"
            assert self._lookup(second) == "second"

            # Storing equal inputs again replaces their entry.
            self._store({"x": jnp.array([1.0, 2.0])}, "again")
            assert jax_recipes._jax_vjp_cache.size == 2
            assert self._lookup(first) == "again"
        finally:
            jax_recipes._set_jax_vjp_cache_size(0)

    def test_arrays_on_different_devices_miss(self):
        # Needs two devices, which a JAX process only gets at startup.
        code = textwrap.dedent(
            """
            import jax
            import jax.numpy as jnp
            from tesseract_core.runtime import jax_recipes

            cache = jax_recipes.LRUCache(maxsize=2)
            key = lambda device: jax_recipes._cache_key({"x": jax.device_put(x, device)})
            cpu0, cpu1 = jax.devices("cpu")[:2]
            x = jnp.array([1.0, 2.0])
            cache.put(key(cpu0), "cached")
            assert cache.get(key(cpu1)) is None
            assert cache.get(key(cpu0)) == "cached"
            """
        )
        env = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
        subprocess.run([sys.executable, "-c", code], env=env, check=True)

    def test_key_holds_arrays_on_different_devices(self):
        # Needs two devices, which a JAX process only gets at startup.
        code = textwrap.dedent(
            """
            import jax
            import jax.numpy as jnp
            from tesseract_core.runtime import jax_recipes

            cache = jax_recipes.LRUCache(maxsize=2)
            cpu0, cpu1 = jax.devices("cpu")[:2]
            x = jnp.array([1.0, 2.0])
            key = lambda d0, d1: jax_recipes._cache_key(
                {"a": jax.device_put(x, d0), "b": jax.device_put(x + 1, d1)}
            )
            cache.put(key(cpu0, cpu1), "cached")
            assert cache.get(key(cpu0, cpu1)) == "cached"
            assert cache.get(key(cpu1, cpu0)) is None
            assert cache.get(key(cpu0, cpu0)) is None
            """
        )
        env = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
        subprocess.run([sys.executable, "-c", code], env=env, check=True)

    def test_bitwise_equal_handles_dtypes(self):
        for arr in (
            jnp.array([True, False]),
            jnp.array([1, 2], dtype=jnp.int8),
            jnp.array([1.0, 2.0], dtype=jnp.bfloat16),
            jnp.array([1 + 2j, 3 - 4j], dtype=jnp.complex64),
        ):
            assert jax_recipes._bitwise_equal((arr,), (arr + 0,))
            assert not jax_recipes._bitwise_equal((arr,), (jnp.flip(arr),))

    def test_cache_on_matches_cache_off(self):
        InputSchema, apply_jit = _build_api()

        def inputs(offset):
            inp = _make_inputs(InputSchema, offset=offset)
            inp.a.v = jnp.asarray(inp.a.v)
            inp.b.v = jnp.asarray(inp.b.v)
            return inp

        ct = {"y": jnp.ones(3, dtype=jnp.float32)}
        expected = jax_vjp(apply_jit, inputs(0.5), {"a.v", "b.v"}, {"y"}, ct)

        jax_recipes._set_jax_vjp_cache_size(1)
        try:
            jax_apply(apply_jit, inputs(0.0))
            hit = self._lookup(as_jax_arrays(inputs(0.0).model_dump()))
            miss = self._lookup(as_jax_arrays(inputs(0.5).model_dump()))
            got = jax_vjp(apply_jit, inputs(0.5), {"a.v", "b.v"}, {"y"}, ct)
        finally:
            jax_recipes._set_jax_vjp_cache_size(0)

        assert hit is not None
        assert miss is None
        for k in expected:
            np.testing.assert_allclose(np.asarray(got[k]), np.asarray(expected[k]))


def _fmix32(h: int) -> int:
    """Reference MurmurHash3 finalizer (fmix32 in smhasher's MurmurHash3.cpp)."""
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    return h ^ (h >> 16)


def _fingerprint(*arrays: jax.Array) -> int:
    return int(jax_recipes._fingerprint_jit(arrays))


class TestFingerprint:
    """The fingerprint that narrows cache lookups of device arrays."""

    # MurmurHash3_x86_32 of empty input reduces to fmix32(seed), so its
    # published empty-input vectors pin down the finalizer.
    @pytest.mark.parametrize(
        "seed,expected", [(0, 0x00000000), (1, 0x514E28B7), (0xFFFFFFFF, 0x81F16F39)]
    )
    def test_mix_matches_murmur3_vectors(self, seed, expected):
        assert int(jax.jit(jax_recipes._mix)(jnp.uint32(seed))) == expected

    def test_mix_matches_reference_and_wraps_in_uint32(self):
        xs = np.random.default_rng(0).integers(0, 2**32, 10_000, dtype=np.uint64)
        got = np.asarray(jax.jit(jax_recipes._mix)(jnp.asarray(xs.astype(np.uint32))))
        assert got.dtype == np.uint32
        assert [int(g) for g in got] == [_fmix32(int(x)) for x in xs]

    def test_depends_only_on_contents(self):
        a = np.arange(10, dtype=np.float32)
        assert _fingerprint(jnp.asarray(a)) == _fingerprint(jnp.asarray(a.copy()))

    @pytest.mark.parametrize(
        "a,b",
        [
            # Each word is salted with its position.
            (jnp.array([1.0, 2.0]), jnp.array([2.0, 1.0])),
            # Bitwise, like __eq__.
            (jnp.array([0.0]), jnp.array([-0.0])),
            (jnp.array([1 + 2j]), jnp.array([2 + 1j])),
            (jnp.array([True, False]), jnp.array([False, True])),
        ],
    )
    def test_distinguishes(self, a, b):
        assert _fingerprint(a) != _fingerprint(b)

    def test_order_of_arrays_matters(self):
        a, b = jnp.array([1.0]), jnp.array([2.0])
        assert _fingerprint(a, b) != _fingerprint(b, a)

    def test_sees_high_word_of_64_bit_values(self):
        # 64-bit arrays need JAX_ENABLE_X64, which is read at startup.
        code = textwrap.dedent(
            """
            import jax.numpy as jnp
            from tesseract_core.runtime import jax_recipes

            fingerprint = lambda x: int(jax_recipes._fingerprint_jit((x,)))
            a = jnp.array([1], dtype=jnp.uint64)
            assert a.dtype == jnp.uint64
            assert fingerprint(a) != fingerprint(a + (1 << 40))
            """
        )
        env = {**os.environ, "JAX_ENABLE_X64": "1"}
        subprocess.run([sys.executable, "-c", code], env=env, check=True)

    def test_spreads_similar_inputs(self):
        # With a well-mixed 32-bit hash, 2000 one-hot arrays collide with
        # probability ~5e-4, so any collision points at poor mixing.
        fps = {
            _fingerprint(jnp.zeros(2000, dtype=jnp.float32).at[i].set(1.0))
            for i in range(2000)
        }
        assert len(fps) == 2000
