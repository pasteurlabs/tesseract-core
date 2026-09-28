# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Julia endpoint helpers, used in the Julia recipe.

These functions remove the boilerplate from a Julia-backed Tesseract
``tesseract_api.py`` by providing one-line implementations of the
``apply``, ``jacobian_vector_product`` and ``vector_jacobian_product``
endpoints. Gradients are computed with Enzyme.

The wrapped Julia function must follow this contract::

    apply_jl(diff_args, non_diff_args, diff_paths, non_diff_paths) -> NamedTuple

- ``diff_args::Vector{Vector{Float64}}`` holds every differentiable input array.
- ``non_diff_args::Vector{Any}`` holds every other input leaf (ints, strings, ...).
- ``diff_paths`` and ``non_diff_paths`` hold the Tesseract path of each value,
  e.g. ``"blocks.[0].[1].main"``.
- The returned NamedTuple has one entry per top-level field of the output
  schema, e.g. ``(; x = solution)``.

Importing this module starts Julia via ``juliacall``. The active Julia project
must provide ``Enzyme`` and ``PythonCall``.
"""

from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from juliacall import Main as jl
from pydantic import BaseModel

from tesseract_core.runtime.schema_generation import (
    _path_tuple_to_str,
    get_all_model_path_patterns,
)
from tesseract_core.runtime.schema_types import is_differentiable
from tesseract_core.runtime.tree_transforms import expand_path_pattern, get_at_path

_recipes = jl.include(str(Path(__file__).with_suffix(".jl")))


class _FlatInputs(NamedTuple):
    diff_args: list[np.ndarray]
    non_diff_args: list[Any]
    diff_paths: list[str]
    non_diff_paths: list[str]


def _expand_paths(
    schema: type[BaseModel],
    inputs: dict,
    filter_fn: Callable[[type], bool] | None = None,
) -> list[str]:
    return [
        path
        for pattern in get_all_model_path_patterns(schema, filter_fn=filter_fn)
        for path in expand_path_pattern(_path_tuple_to_str(pattern), inputs)
    ]


def _flatten_inputs(inputs: BaseModel) -> _FlatInputs:
    """Split the leaves of ``inputs`` into differentiable arrays and everything else."""
    schema = type(inputs)
    inputs_dict = inputs.model_dump()
    diff_paths = set(_expand_paths(schema, inputs_dict, filter_fn=is_differentiable))

    flat = _FlatInputs([], [], [], [])
    for path in _expand_paths(schema, inputs_dict):
        value = get_at_path(inputs_dict, path)
        if isinstance(value, (dict, list)):
            # Container patterns expand too, but only leaves are passed to Julia.
            continue
        if path in diff_paths:
            flat.diff_args.append(np.asarray(value, dtype=np.float64))
            flat.diff_paths.append(path)
        else:
            flat.non_diff_args.append(value)
            flat.non_diff_paths.append(path)
    return flat


def _from_julia(julia_dict: Any, keys: Collection[str] | None = None) -> dict[str, Any]:
    out = {}
    for key, value in julia_dict.items():
        if keys is None or key in keys:
            out[key] = np.array(value) if hasattr(value, "__array__") else value
    return out


def julia_apply(apply_fn: Any, inputs: BaseModel) -> dict:
    """Call ``apply_fn`` on ``inputs`` and return its outputs as a dict."""
    return _from_julia(_recipes.apply(apply_fn, *_flatten_inputs(inputs)))


def julia_jvp(
    apply_fn: Any,
    inputs: BaseModel,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute the Jacobian-vector product of ``apply_fn`` with forward-mode Enzyme."""
    jvp_paths = list(jvp_inputs)
    tangents = [np.asarray(tangent_vector[p], dtype=np.float64) for p in jvp_paths]
    out = _recipes.jvp(apply_fn, *_flatten_inputs(inputs), jvp_paths, tangents)
    return _from_julia(out, keys=jvp_outputs)


def julia_vjp(
    apply_fn: Any,
    inputs: BaseModel,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute the vector-Jacobian product of ``apply_fn`` with reverse-mode Enzyme."""
    output_names = list(vjp_outputs)
    cotangents = [
        np.asarray(cotangent_vector[name], dtype=np.float64) for name in output_names
    ]
    out = _recipes.vjp(
        apply_fn,
        *_flatten_inputs(inputs),
        list(vjp_inputs),
        output_names,
        cotangents,
    )
    return _from_julia(out)
