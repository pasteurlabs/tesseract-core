# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable implementations of gradient endpoints for Tesseract APIs.

The helpers in this package keep ``tesseract_api.py`` focused on the
user's ``apply_jit`` and schemas by absorbing the boilerplate of the
gradient endpoints (``apply``, ``jacobian``, ``jacobian_vector_product``,
``vector_jacobian_product``, ``abstract_eval``).

See ``tesseract_core.runtime.gradient_recipes.jax`` for JAX-backed helpers.
"""

from tesseract_core.runtime.gradient_recipes.jax import (
    jax_abstract_eval,
    jax_apply,
    jax_jacobian,
    jax_jvp,
    jax_vjp,
    jax_vjp_cache,
    set_jax_cache_size,
)

__all__ = [
    "jax_abstract_eval",
    "jax_apply",
    "jax_jacobian",
    "jax_jvp",
    "jax_vjp",
    "jax_vjp_cache",
    "set_jax_cache_size",
]
