# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable PyTorch wrapper for Tesseract operations.

This module registers a Tesseract as a first-class differentiable primitive in
PyTorch's autograd graph.  The forward pass dispatches to ``tesseract.apply()``,
the backward pass to ``tesseract.vector_jacobian_product()``, and the
forward-mode JVP to ``tesseract.jacobian_vector_product()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch is required to use tesseract_core.torch_compat. "
        "Install it with:  pip install tesseract-core[torch]"
    ) from exc

from ..sdk.tesseract import Tesseract


def _to_tensor(arr: Any) -> torch.Tensor:
    """Convert a numpy array to a float32 tensor, copying if read-only."""
    a = np.asarray(arr)
    if not a.flags.writeable:
        a = a.copy()
    return torch.as_tensor(a, dtype=torch.float32)


def _get_differentiable_arrays(
    openapi_schema: dict,
    component: str,
) -> set[str]:
    """Extract differentiable array dotted-paths from the OpenAPI schema."""
    schema = openapi_schema["components"]["schemas"].get(component, {})
    return set(schema.get("differentiable_arrays", {}))


# ---------------------------------------------------------------------------
# Pytree helpers - flatten / unflatten nested dicts using dotted paths
# ---------------------------------------------------------------------------


def _flatten_pytree(
    tree: dict[str, Any],
    prefix: str = "",
    *,
    recurse_into: set[str] | None = None,
) -> list[tuple[str, Any]]:
    """Flatten a nested dict into ``(dotted_path, leaf_value)`` pairs.

    Only recurses into sub-dicts whose dotted prefix is a strict prefix of at
    least one path in *recurse_into*.  All other dicts are treated as opaque
    leaf values (e.g. ``dict[str, Array]`` schema fields).

    If *recurse_into* is ``None``, every nested dict is recursed into.
    """
    items: list[tuple[str, Any]] = []
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and _should_recurse(path, value, recurse_into):
            items.extend(_flatten_pytree(value, path, recurse_into=recurse_into))
        else:
            items.append((path, value))
    return items


def _should_recurse(
    path: str,
    value: dict,
    known_paths: set[str] | None,
) -> bool:
    """Return True when *path* is a prefix of a known leaf path."""
    if not value:
        return False
    if known_paths is None:
        return True
    dot_prefix = path + "."
    return any(p.startswith(dot_prefix) for p in known_paths)


def _unflatten_pytree(flat: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a nested dict from ``{dotted_path: value}``."""
    tree: dict[str, Any] = {}
    for path, value in flat.items():
        parts = path.split(".")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree


# ---------------------------------------------------------------------------
# Core autograd function
# ---------------------------------------------------------------------------


class _TesseractFunction(torch.autograd.Function):
    """Low-level autograd function wrapping a Tesseract.

    This is an implementation detail.  Users should call :func:`apply_tesseract`.
    """

    @staticmethod
    def forward(
        tesseract: Tesseract,
        diff_input_names: list[str],
        diff_output_names: list[str],
        all_paths: set[str],
        static_inputs: dict[str, Any],
        non_diff_result_holder: list[dict[str, Any]],
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the Tesseract forward pass, returning differentiable outputs.

        The full (flat) result dict is stashed in *non_diff_result_holder*
        so the caller can reconstruct non-differentiable outputs without a
        second ``apply()`` call.
        """
        flat_inputs = dict(static_inputs)
        for name, tensor in zip(diff_input_names, tensors, strict=True):
            flat_inputs[name] = tensor.detach().cpu().numpy()

        result = tesseract.apply(_unflatten_pytree(flat_inputs))
        flat_result = dict(_flatten_pytree(result, recurse_into=all_paths))

        # Stash full result for the caller
        non_diff_result_holder.append(flat_result)

        # Return only the differentiable output tensors (in sorted order)
        return tuple(_to_tensor(flat_result[name]) for name in diff_output_names)

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> None:
        """Save forward-pass metadata for use in backward / jvp."""
        (
            tesseract,
            diff_input_names,
            diff_output_names,
            all_paths,  # noqa: RUF059
            static_inputs,
            _holder,
            *tensors,
        ) = inputs
        ctx.tesseract = tesseract
        ctx.diff_input_names = diff_input_names
        ctx.diff_output_names = diff_output_names

        saved_inputs: dict[str, Any] = dict(static_inputs)
        for name, tensor in zip(diff_input_names, tensors, strict=True):
            saved_inputs[name] = tensor.detach().cpu().numpy()
        ctx.saved_inputs = saved_inputs

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[None | torch.Tensor, ...]:
        """Reverse-mode AD via the Tesseract's VJP endpoint."""
        cotangent_vector = {
            name: grad.detach().cpu().numpy()
            for name, grad in zip(ctx.diff_output_names, grad_outputs, strict=True)
        }

        vjp_result = ctx.tesseract.vector_jacobian_product(
            inputs=_unflatten_pytree(ctx.saved_inputs),
            vjp_inputs=list(ctx.diff_input_names),
            vjp_outputs=list(ctx.diff_output_names),
            cotangent_vector=cotangent_vector,
        )

        grad_inputs: list[torch.Tensor | None] = []
        for name in ctx.diff_input_names:
            g = vjp_result.get(name)
            grad_inputs.append(_to_tensor(g) if g is not None else None)

        # None for (tesseract, diff_input_names, diff_output_names,
        #           all_paths, static_inputs, non_diff_result_holder)
        return (None, None, None, None, None, None, *grad_inputs)

    @staticmethod
    def jvp(
        ctx: Any,
        *tangents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Forward-mode AD via the Tesseract's JVP endpoint."""
        # tangents: (tesseract, diff_input_names, diff_output_names,
        #            all_paths, static_inputs, holder, *tensor_tangents)
        tensor_tangents = tangents[6:]

        tangent_vector: dict[str, Any] = {}
        jvp_inputs: list[str] = []
        for name, t in zip(ctx.diff_input_names, tensor_tangents, strict=True):
            if t is not None:
                tangent_vector[name] = t.detach().cpu().numpy()
                jvp_inputs.append(name)

        if not jvp_inputs:
            return tuple(
                torch.zeros_like(_to_tensor(ctx.saved_inputs.get(name, 0.0)))
                for name in ctx.diff_output_names
            )

        jvp_result = ctx.tesseract.jacobian_vector_product(
            inputs=_unflatten_pytree(ctx.saved_inputs),
            jvp_inputs=jvp_inputs,
            jvp_outputs=list(ctx.diff_output_names),
            tangent_vector=tangent_vector,
        )

        return tuple(_to_tensor(jvp_result[name]) for name in ctx.diff_output_names)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_tesseract(
    tesseract: Tesseract,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Call a Tesseract as a differentiable PyTorch operation.

    Infers which inputs/outputs are differentiable from the Tesseract's schema.
    Torch tensors provided for differentiable fields participate in autograd;
    all other values are passed through as static inputs.

    Supports both reverse-mode (``.backward()``) and forward-mode
    (``torch.autograd.forward_ad``) differentiation.

    Args:
        tesseract: A Tesseract instance.
        inputs: Nested dict matching the Tesseract's input schema.  Provide
            ``torch.Tensor`` for array fields you want gradients through,
            and plain Python / NumPy values for everything else.

    Returns:
        Nested dict matching the Tesseract's output schema, with
        differentiable array outputs as ``torch.Tensor`` (with ``grad_fn``
        when inputs require grad) and non-differentiable outputs as-is
        (NumPy arrays or scalars).

    Example::

        # Flat schema
        result = apply_tesseract(quadratic, {"x": x, "A": A, "b": b})
        result["y"].sum().backward()

        # Nested schema
        result = apply_tesseract(meshstats, {
            "mesh": {"n_points": 3, ..., "points": points_tensor}
        })
        result["statistics"]["barycenter"].sum().backward()
    """
    openapi = tesseract.openapi_schema
    diff_in_paths = _get_differentiable_arrays(openapi, "ApplyInputSchema")
    diff_out_paths = _get_differentiable_arrays(openapi, "ApplyOutputSchema")
    diff_out_names = sorted(diff_out_paths)

    # All known dotted paths guide pytree flattening so we recurse into
    # sub-models but not into opaque dict fields.
    all_paths = diff_in_paths | diff_out_paths

    flat_inputs = _flatten_pytree(inputs, recurse_into=all_paths)

    # Partition into differentiable tensors vs static values
    diff_names: list[str] = []
    diff_tensors: list[torch.Tensor] = []
    static: dict[str, Any] = {}

    for path, value in flat_inputs:
        if path in diff_in_paths and isinstance(value, torch.Tensor):
            diff_names.append(path)
            diff_tensors.append(value)
        elif isinstance(value, torch.Tensor):
            static[path] = value.detach().cpu().numpy()
        else:
            static[path] = value

    # Mutable holder so forward() can pass the full result dict back to us
    # without going through autograd's return values.
    result_holder: list[dict[str, Any]] = []

    output_tensors = _TesseractFunction.apply(
        tesseract,
        diff_names,
        diff_out_names,
        all_paths,
        static,
        result_holder,
        *diff_tensors,
    )

    # Reconstruct full output pytree
    flat_result = dict(result_holder[0])
    for name, tensor in zip(diff_out_names, output_tensors, strict=True):
        flat_result[name] = tensor

    return _unflatten_pytree(flat_result)
