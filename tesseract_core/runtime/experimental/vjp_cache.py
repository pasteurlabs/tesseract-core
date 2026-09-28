# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


def set_jax_vjp_cache_size(size: int) -> None:
    """Enable or resize the VJP residual cache, discarding existing entries.

    **Experimental.** The cache is opt-in and disabled by default; its
    performance characteristics are problem-dependent (see below) and the
    interface may change.

    When ``jax_apply`` and ``jax_vjp`` (from
    :mod:`tesseract_core.runtime.jax_recipes`) are called in sequence on the
    same inputs (the typical pattern under tesseract-jax's ``custom_vjp``,
    including under ``jax.value_and_grad`` and ``jax.jacrev``), caching the
    backward function produced by ``jax_apply``'s internal ``jax.vjp`` call
    lets the subsequent ``jax_vjp`` skip the redundant forward pass. The win
    scales with the cost of the forward pass and goes the wrong way on
    trivially-small ones, so benchmark your own workload before enabling.

    Best fit:
      - tesseract-jax workflows that go through ``custom_vjp`` (every
        gradient evaluation runs ``apply()`` first via ``fwd``, so the
        cache reliably hits on the subsequent ``vjp()`` -- true even
        under plain ``jax.grad``).
      - Manual ``apply()`` followed by multiple ``vjp()`` calls on the
        same inputs (e.g. iterative solvers, CG-style inverse problems).

    Poor fit:
      - Very small models where the forward pass is microseconds: the
        cache machinery overhead exceeds the saved work.

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
