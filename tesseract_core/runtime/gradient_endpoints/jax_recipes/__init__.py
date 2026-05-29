# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX gradient endpoint helpers.

These functions remove the boilerplate from a JAX-backed Tesseract
``tesseract_api.py`` by providing one-line implementations of the
``apply``, ``jacobian``, ``jacobian_vector_product``,
``vector_jacobian_product`` and ``abstract_eval`` endpoints. The user
supplies a single ``apply_jit`` (already wrapped in ``@eqx.filter_jit``);
the helpers compose the JAX transforms around it.

VJP residual caching is enabled by default: :func:`jax_apply` stashes the
backward function it constructs and :func:`jax_vjp` reuses it on the next
matching call — typically a ~10-20% speedup on the standard tesseract-jax
``apply → vjp`` pattern (which is hit by ``jax.value_and_grad``,
``jax.jacrev``, and even plain ``jax.grad`` since tesseract-jax's
``custom_vjp.fwd`` always runs ``apply()`` first). Call
:func:`set_jax_vjp_cache_size` with ``0`` to disable. See its docstring
for the full taxonomy.
"""

import hashlib
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel

from tesseract_core.runtime.tree_transforms import (
    LRUCache,
    filter_func,
    flatten_with_paths,
    set_at_path,
)

#: Module-level VJP residual cache. ``jax_apply`` populates it; ``jax_vjp``
#: reads from it. Set to ``None`` (via :func:`set_jax_vjp_cache_size` with
#: ``0``) to disable caching — both helpers then bypass the cache machinery
#: entirely.
jax_vjp_cache: LRUCache | None = LRUCache(maxsize=1)


def set_jax_vjp_cache_size(size: int) -> None:
    """Enable or resize the VJP residual cache, discarding existing entries.

    When :func:`jax_apply` and :func:`jax_vjp` are called in sequence on the
    same inputs (the typical pattern under tesseract-jax's ``custom_vjp``,
    including under ``jax.value_and_grad`` and ``jax.jacrev``), caching the
    backward function produced by :func:`jax_apply`'s internal ``jax.vjp``
    call lets the subsequent :func:`jax_vjp` skip the redundant forward
    pass. Typical speedup ~10-20% on moderate-to-deep networks.

    Best fit:
      - tesseract-jax workflows that go through ``custom_vjp`` (every
        gradient evaluation runs ``apply()`` first via ``fwd``, so the
        cache reliably hits on the subsequent ``vjp()`` — true even
        under plain ``jax.grad``).
      - Manual ``apply()`` followed by multiple ``vjp()`` calls on the
        same inputs (e.g. iterative solvers, CG-style inverse problems).

    Poor fit:
      - Very small models where the forward pass is microseconds: the
        cache machinery overhead exceeds the saved work.

    Args:
        size: Number of cache slots. ``1`` (the default) covers the standard
            apply → vjp pattern. ``0`` disables caching entirely (both
            :func:`jax_apply` and :func:`jax_vjp` then bypass the cache
            machinery). Increase for workflows that interleave multiple
            ``apply()`` calls before their corresponding ``vjp()`` calls.
    """
    global jax_vjp_cache
    jax_vjp_cache = LRUCache(maxsize=size) if size > 0 else None


def hash_tree(tree: Any) -> bytes:
    """Compute a SHA-256 digest over a pytree's structure and leaves.

    Suitable for use as an :class:`LRUCache` key. For array leaves, the digest
    incorporates dtype + shape + raw bytes so leaves with identical bytes but
    different interpretations (e.g. ``int64[4]`` vs ``int64[2,2]``) don't
    collide. Scalar and primitive leaves use Python's ``hash()`` except for
    ``str``/``bytes`` (whose hash is randomized across processes by Python's
    PYTHONHASHSEED — we encode them directly instead).
    """
    leaves, treedef = jax.tree.flatten(tree)
    h = hashlib.sha256()
    h.update(str(treedef).encode())
    for leaf in leaves:
        if hasattr(leaf, "tobytes"):
            h.update(str(leaf.dtype).encode())
            h.update(str(leaf.shape).encode())
            h.update(leaf.tobytes())
        elif isinstance(leaf, (str, bytes)):
            h.update(leaf if isinstance(leaf, bytes) else leaf.encode())
        else:
            h.update(hash(leaf).to_bytes(8, "big", signed=True))
    return h.digest()


def jax_apply(apply_jit: Callable, inputs: BaseModel) -> dict:
    """Run ``apply_jit`` and, if caching is enabled, populate the VJP cache.

    ``apply_jit`` is assumed to already be JIT-compiled (e.g. wrapped with
    ``@eqx.filter_jit``); this helper does not jit it. The user-facing
    ``apply`` endpoint may want to do pre/post-processing around the call,
    so we cannot wrap it in a jit internally.

    When :data:`jax_vjp_cache` is set (see :func:`set_jax_vjp_cache_size`),
    the forward pass is run via ``jax.vjp`` so the resulting backward
    function can be stashed and reused by a later :func:`jax_vjp` call.
    Otherwise this is just ``apply_jit(inputs.model_dump())``.
    """
    inputs_dict = inputs.model_dump()

    if jax_vjp_cache is None:
        return apply_jit(inputs_dict)

    # Compute forward pass via jax.vjp to cache residuals for a potential
    # subsequent vector_jacobian_product call. eqx.partition separates
    # array (differentiable) from non-array outputs; has_aux tells jax.vjp
    # to only differentiate through the array outputs.
    def _apply_for_vjp(inputs_dict: dict) -> tuple:
        out = apply_jit(inputs_dict)
        diff_out, static_out = eqx.partition(out, eqx.is_array)
        return diff_out, static_out

    diff_primals, vjp_func, static_primals = jax.vjp(
        _apply_for_vjp, inputs_dict, has_aux=True
    )
    out = eqx.combine(diff_primals, static_primals)

    cotangent_template = jax.tree.map(jnp.zeros_like, diff_primals)
    jax_vjp_cache.put(hash_tree(inputs_dict), (vjp_func, cotangent_template))
    return out


def jax_vjp(
    apply_jit: Callable,
    inputs: BaseModel,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute the vector-Jacobian product.

    Reuses the cached backward from a prior :func:`jax_apply` call when one
    is available (see :func:`set_jax_vjp_cache_size`); otherwise falls
    through to a freshly JIT-compiled ``jax.vjp`` evaluation. The JIT
    compilation happens internally on the first miss for a given
    (input shape/dtype, path subset) combination and is cached for reuse.
    """
    inputs_dict = inputs.model_dump()

    # Use get (not pop) so the cached residuals can serve multiple sequential
    # vjp calls on the same inputs — for example, when tesseract-jax's
    # value_and_grad is followed by jax.jacrev, which decomposes into many
    # vjp calls per output basis vector.
    if (
        jax_vjp_cache is not None
        and (cached := jax_vjp_cache.get(hash_tree(inputs_dict))) is not None
    ):
        vjp_func, cotangent_template = cached
        full_cotangent = jax.tree.map(jnp.zeros_like, cotangent_template)
        full_cotangent = set_at_path(full_cotangent, cotangent_vector)
        (all_input_cotangents,) = vjp_func(full_cotangent)
        return flatten_with_paths(all_input_cotangents, include_paths=vjp_inputs)

    # Cache disabled or cache miss: fall back to original JIT-compiled path.
    return _vjp_jit(
        apply_jit,
        inputs_dict,
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


def jax_jvp(
    apply_jit: Callable,
    inputs: BaseModel,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute the Jacobian-vector product via :func:`jax.jvp`.

    JIT compilation is applied internally and cached per
    ``(input shape/dtype, jvp_inputs, jvp_outputs)`` combination.
    """
    return _jvp_jit(
        apply_jit,
        inputs.model_dump(),
        tuple(jvp_inputs),
        tuple(jvp_outputs),
        tangent_vector,
    )


def jax_jacobian(
    apply_jit: Callable,
    inputs: BaseModel,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> dict[str, dict[str, Any]]:
    """Compute the Jacobian via :func:`jax.jacrev`.

    JIT compilation is applied internally and cached per
    ``(input shape/dtype, jac_inputs, jac_outputs)`` combination.
    """
    return _jac_jit(
        apply_jit, inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs)
    )


def jax_abstract_eval(apply_jit: Callable, abstract_inputs: BaseModel) -> dict:
    """Calculate the output shape of ``apply_jit`` from the shape of its inputs."""
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs: dict) -> dict:
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


#
# Internal jit-compiled fallbacks (used on cache miss or when caching is disabled).
#


@eqx.filter_jit
def _jac_jit(
    apply_jit: Callable,
    inputs: dict,
    jac_inputs: tuple[str, ...],
    jac_outputs: tuple[str, ...],
) -> dict:
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )


@eqx.filter_jit
def _jvp_jit(
    apply_jit: Callable,
    inputs: dict,
    jvp_inputs: tuple[str, ...],
    jvp_outputs: tuple[str, ...],
    tangent_vector: dict,
) -> dict:
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


@eqx.filter_jit
def _vjp_jit(
    apply_jit: Callable,
    inputs: dict,
    vjp_inputs: tuple[str, ...],
    vjp_outputs: tuple[str, ...],
    cotangent_vector: dict,
) -> dict:
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]
