# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX gradient endpoint helpers.

These functions remove the boilerplate from a JAX-backed Tesseract
``tesseract_api.py`` by providing one-line implementations of the
``apply``, ``jacobian``, ``jacobian_vector_product``,
``vector_jacobian_product`` and ``abstract_eval`` endpoints, along with
shared VJP residual caching across ``apply`` and ``vector_jacobian_product``.

Cache behaviour
---------------
By default :func:`jax_apply` runs ``jax.vjp`` internally and stashes the
resulting backward function in :data:`jax_vjp_cache`. A subsequent
:func:`jax_vjp` call on the same inputs reuses that backward, skipping the
redundant forward pass — typically a 10-20% saving on moderate-to-deep
networks. Call :func:`set_jax_cache_size` with ``0`` to disable caching
entirely; both :func:`jax_apply` and :func:`jax_vjp` then bypass the cache
machinery and run their forward passes directly.
"""

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
    hash_pytree_leaves,
    set_at_path,
)

# Default size of the VJP residual cache. Use :func:`set_jax_cache_size` to
# change at runtime; do not mutate this directly.
_jax_cache_size = 1

#: Module-level VJP residual cache. ``jax_apply`` populates it; ``jax_vjp``
#: reads from it.
jax_vjp_cache = LRUCache(maxsize=_jax_cache_size)


def set_jax_cache_size(size: int) -> None:
    """Resize the VJP cache. Discards any existing entries."""
    global _jax_cache_size, jax_vjp_cache
    _jax_cache_size = size
    jax_vjp_cache = LRUCache(maxsize=size)


def _hash_inputs(inputs_dict: dict) -> bytes:
    """Compute a SHA-256 hash of a pytree of inputs for cache key comparison."""
    leaves, treedef = jax.tree.flatten(inputs_dict)
    return hash_pytree_leaves(leaves, treedef)


def jax_apply(apply_jit: Callable, inputs: BaseModel) -> dict:
    """Run ``apply_jit`` and populate the VJP cache.

    A subsequent :func:`jax_vjp` call on the same input reuses the cached
    backward. When caching is disabled (see :func:`set_jax_cache_size`),
    this is just ``apply_jit(inputs.model_dump())``.
    """
    inputs_dict = inputs.model_dump()

    if _jax_cache_size <= 0:
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
    jax_vjp_cache.put(_hash_inputs(inputs_dict), (vjp_func, cotangent_template))
    return out


def jax_vjp(
    apply_jit: Callable,
    inputs: BaseModel,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute the vector-Jacobian product.

    Reuses the cached backward from a prior :func:`jax_apply` call if one is
    available; otherwise falls through to a fresh ``jax.vjp`` evaluation.
    """
    inputs_dict = inputs.model_dump()

    # Use get (not pop) so the cached residuals can serve multiple sequential
    # vjp calls on the same inputs — for example, when tesseract-jax's
    # value_and_grad is followed by jax.jacrev, which decomposes into many
    # vjp calls per output basis vector.
    if (
        _jax_cache_size > 0
        and (cached := jax_vjp_cache.get(_hash_inputs(inputs_dict))) is not None
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
    """Compute the Jacobian-vector product via :func:`jax.jvp`."""
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
    """Compute the Jacobian via :func:`jax.jacrev`."""
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
