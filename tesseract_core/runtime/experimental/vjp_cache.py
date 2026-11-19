# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


def set_jax_vjp_cache_size(size: int) -> None:
    """Enable or resize the VJP residual cache, discarding existing entries.

    **Experimental.** The cache is opt-in and disabled by default; its
    performance characteristics are problem-dependent (see below) and the
    interface may change.

    When ``jax_apply`` and ``jax_vjp`` (from
    :mod:`tesseract_core.runtime.jax_recipes`) are called in sequence on the
    same inputs (as tesseract-jax does under ``jax.grad`` and
    ``jax.value_and_grad``), caching the backward function produced by
    ``jax_apply``'s internal ``jax.vjp`` call lets the subsequent ``jax_vjp``
    skip the redundant forward pass. The win scales with the cost of the
    forward pass and goes the wrong way on trivially-small ones, so benchmark
    your own workload before enabling.

    Best fit:
      - tesseract-jax reverse mode: ``jax.grad`` and ``jax.value_and_grad``
        call ``apply()`` and then ``vector_jacobian_product()`` on the same
        inputs, so the second call hits the cache.
      - Manual ``apply()`` followed by multiple ``vjp()`` calls on the
        same inputs (e.g. iterative solvers, CG-style inverse problems).

    Poor fit:
      - Very small models where the forward pass is microseconds: the
        cache machinery overhead exceeds the saved work.
      - ``jax.jit`` where nothing the compiled function returns depends
        on ``apply``'s outputs (e.g. ``jax.grad`` of a loss linear in
        them): JAX drops the ``apply()`` call, so nothing fills the cache.

    **Supported outputs.** With the cache enabled, ``apply`` may only nest its
    outputs in dicts, lists, tuples and NamedTuples, because a cache hit
    rebuilds the cotangent in the same structure. Any other pytree container
    (for example an ``equinox.Module`` or a dataclass registered as a pytree)
    makes ``jax_apply`` raise a ``TypeError``; such outputs work with the
    cache disabled.

    Args:
        size: Number of cache slots. ``0`` (the default) disables caching
            entirely (both ``jax_apply`` and ``jax_vjp`` then bypass the cache
            machinery). ``1`` covers the standard apply -> vjp pattern.
            Increase for workflows that interleave multiple ``apply()`` calls
            before their corresponding ``vjp()`` calls.
    """
    # Imported lazily: jax_recipes pulls in optional jax/equinox dependencies
    # that the core runtime (which eagerly imports this package) must not require.
    from tesseract_core.runtime.jax_recipes import _set_jax_vjp_cache_size

    _set_jax_vjp_cache_size(size)
