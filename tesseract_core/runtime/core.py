# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Union

from pydantic import BaseModel

from .config import get_config
from .schema_generation import (
    create_abstract_eval_schema,
    create_apply_schema,
    create_autodiff_schema,
)
from .tree_transforms import get_at_path


def load_module_from_path(path: Union[Path, str]) -> ModuleType:
    """Load a module from a file path."""
    path = Path(path)

    if not path.is_file():
        raise ImportError(f"Could not load module from {path} (is not a file)")

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(f"Could not load module from {path}") from exc
    return module


def get_supported_endpoints(api_module: ModuleType) -> tuple[str, ...]:
    """Get available Tesseract functions.

    Returns:
        All optional function names defined by the Tesseract.
    """
    optional_funcs = {
        "jacobian",
        "jacobian_vector_product",
        "vector_jacobian_product",
        "abstract_eval",
    }
    return tuple(func for func in optional_funcs if hasattr(api_module, func))


def get_tesseract_api() -> ModuleType:
    """Import tesseract_api.py file."""
    return load_module_from_path(get_config().tesseract_api_path)


def _validate_ad_input(
    inputs: BaseModel,
    input_keys: set[str],
    output_keys: set[str],
    endpoint_name: str,
    tangent_cotangent_vector: Optional[dict[str, Any]] = None,
):
    """Raise an exception if the input of an autodiff function does not conform to given input keys."""
    # Could be moved to a model validator
    for input_key in input_keys:
        if not re.match(r"^[a-zA-Z0-9_.\[\]{}]+$", input_key):
            raise RuntimeError(
                f"Error when validating input of function {endpoint_name}:\n"
                f"Invalid input key {input_key}."
            )

        try:
            get_at_path(inputs, input_key)
        except Exception as exc:
            raise RuntimeError(
                f"Error when validating input of function {endpoint_name}:\n"
                f"Could not find input path {input_key} in inputs."
            ) from exc

    if endpoint_name == "jacobian_vector_product":
        # Tangent vector needs same keys as input_keys
        if set(tangent_cotangent_vector.keys()) != input_keys:
            raise RuntimeError(
                f"Error when validating input of function {endpoint_name}:\n"
                f"Expected tangent vector with keys {input_keys}, got {set(tangent_cotangent_vector.keys())}."
            )

    if endpoint_name == "vector_jacobian_product":
        # Cotangent vector needs same keys as output_keys
        if set(tangent_cotangent_vector.keys()) != output_keys:
            raise RuntimeError(
                f"Error when validating input of function {endpoint_name}:\n"
                f"Expected cotangent vector with keys {output_keys}, got {set(tangent_cotangent_vector.keys())}."
            )


def _validate_ad_output(
    input: Any,
    output: Any,
    input_keys: set[str],
    output_keys: set[str],
    endpoint_name: str,
):
    """Raise an exception if the output structure of an autodiff function does not conform to given keys."""
    if not isinstance(output, dict):
        raise RuntimeError(
            f"Error when validating output of function {endpoint_name}:\n"
            f"Expected output to be a dictionary, got {type(output)}"
        )

    if endpoint_name == "jacobian":
        if output_keys != output.keys():
            raise RuntimeError(
                "Error when validating output of jacobian:\n"
                f"Expected keys {output_keys} in output; got {set(output.keys())}"
            )
        for subkey, subout in output.items():
            if not isinstance(subout, dict):
                raise RuntimeError(
                    "Error when validating output of jacobian:\n"
                    f"Expected output with structure {{{tuple(output_keys)}: {{{tuple(input_keys)}: ...}}}}, "
                    f"got unexpected type {type(subout)} for output key {subkey}."
                )
            if input_keys != subout.keys():
                raise RuntimeError(
                    "Error when validating output of jacobian:\n"
                    f"Expected output with structure {{{tuple(output_keys)}: {{{tuple(input_keys)}: ...}}}}, "
                    f"got {set(subout.keys())} for output key {subkey}."
                )

    elif endpoint_name == "jacobian_vector_product":
        if output_keys != output.keys():
            raise RuntimeError(
                "Error when validating output of jacobian_vector_product:\n"
                f"Expected keys {output_keys} in output; got {set(output.keys())}"
            )

    elif endpoint_name == "vector_jacobian_product":
        if input_keys != output.keys():
            raise RuntimeError(
                "Error when validating output of vector_jacobian_product:\n"
                f"Expected keys {input_keys} in output; got {set(output.keys())}"
            )
    else:
        raise RuntimeError(f"Unknown endpoint name {endpoint_name}")


def create_endpoints(api_module: ModuleType) -> list[Callable]:
    """Create the Tesseract API endpoints.

    This ensures proper type annotations, signatures, and validation for the external-facing API.

    Args:
        api_module: The Tesseract API module.

    Returns:
        A tuple of all Tesseract API endpoints as callables.
    """
    supported_functions = get_supported_endpoints(api_module)

    endpoints = []

    def assemble_docstring(wrapped_func: Callable):
        """Decorator to assemble a docstring from multiple functions."""

        def inner(otherfunc: Callable):
            doc_parts = []
            if otherfunc.__doc__:
                doc_parts.append(otherfunc.__doc__)

            if wrapped_func.__doc__:
                doc_parts.append(wrapped_func.__doc__)

            otherfunc.__doc__ = "\n\n".join(doc_parts)
            return otherfunc

        return inner

    ApplyInputSchema, ApplyOutputSchema = create_apply_schema(
        api_module.InputSchema, api_module.OutputSchema
    )

    @assemble_docstring(api_module.apply)
    def apply(payload: ApplyInputSchema) -> ApplyOutputSchema:
        """Apply the Tesseract to the input data."""
        out = api_module.apply(payload.inputs)
        if isinstance(out, api_module.OutputSchema):
            out = out.model_dump()
        return ApplyOutputSchema.model_validate(out)

    endpoints.append(apply)

    if "jacobian" in supported_functions:
        JacobianInputSchema, JacobianOutputSchema = create_autodiff_schema(
            api_module.InputSchema, api_module.OutputSchema, ad_flavor="jacobian"
        )

        @assemble_docstring(api_module.jacobian)
        def jacobian(payload: JacobianInputSchema) -> JacobianOutputSchema:
            """Computes the Jacobian of the Tesseract.

            Differentiates ``jac_outputs`` with respect to ``jac_inputs``, at the point ``inputs``.
            """
            _validate_ad_input(
                payload.inputs, payload.jac_inputs, payload.jac_outputs, "jacobian"
            )
            out = api_module.jacobian(
                payload.inputs, payload.jac_inputs, payload.jac_outputs
            )
            _validate_ad_output(
                payload.inputs, out, payload.jac_inputs, payload.jac_outputs, "jacobian"
            )
            return JacobianOutputSchema.model_validate(out)

        endpoints.append(jacobian)

    if "jacobian_vector_product" in supported_functions:
        JVPInputSchema, JVPOutputSchema = create_autodiff_schema(
            api_module.InputSchema, api_module.OutputSchema, ad_flavor="jvp"
        )

        @assemble_docstring(api_module.jacobian_vector_product)
        def jacobian_vector_product(payload: JVPInputSchema) -> JVPOutputSchema:
            """Compute the Jacobian vector product of the Tesseract at the input data.

            Evaluates the Jacobian vector product between the Jacobian given by ``jvp_outputs``
            with respect to ``jvp_inputs`` at the point ``inputs`` and the given tangent vector.
            """
            _validate_ad_input(
                payload.inputs,
                payload.jvp_inputs,
                payload.jvp_outputs,
                "jacobian_vector_product",
                tangent_cotangent_vector=payload.tangent_vector,
            )
            out = api_module.jacobian_vector_product(
                payload.inputs,
                payload.jvp_inputs,
                payload.jvp_outputs,
                payload.tangent_vector,
            )
            _validate_ad_output(
                payload.inputs,
                out,
                payload.jvp_inputs,
                payload.jvp_outputs,
                "jacobian_vector_product",
            )
            return JVPOutputSchema.model_validate(out)

        endpoints.append(jacobian_vector_product)

    if "vector_jacobian_product" in supported_functions:
        VJPInputSchema, VJPOutputSchema = create_autodiff_schema(
            api_module.InputSchema, api_module.OutputSchema, ad_flavor="vjp"
        )

        @assemble_docstring(api_module.vector_jacobian_product)
        def vector_jacobian_product(payload: VJPInputSchema) -> VJPOutputSchema:
            """Compute the Jacobian vector product of the Tesseract at the input data.

            Computes the vector Jacobian product between the Jacobian given by ``vjp_outputs``
            with respect to ``vjp_inputs`` at the point ``inputs`` and the given cotangent vector.
            """
            _validate_ad_input(
                payload.inputs,
                payload.vjp_inputs,
                payload.vjp_outputs,
                "vector_jacobian_product",
                tangent_cotangent_vector=payload.cotangent_vector,
            )
            out = api_module.vector_jacobian_product(
                payload.inputs,
                payload.vjp_inputs,
                payload.vjp_outputs,
                payload.cotangent_vector,
            )
            _validate_ad_output(
                payload.inputs,
                out,
                payload.vjp_inputs,
                payload.vjp_outputs,
                "vector_jacobian_product",
            )
            return VJPOutputSchema.model_validate(out)

        endpoints.append(vector_jacobian_product)

    def health() -> dict[str, Any]:
        """Get health status of the Tesseract instance."""
        return {"status": "ok"}

    endpoints.append(health)

    def input_schema() -> dict[str, Any]:
        """Get input schema for tesseract apply function."""
        return api_module.InputSchema.model_json_schema()

    endpoints.append(input_schema)

    def output_schema() -> dict[str, Any]:
        """Get output schema for tesseract apply function."""
        return api_module.OutputSchema.model_json_schema()

    endpoints.append(output_schema)

    if "abstract_eval" in supported_functions:
        AbstractEvalInputSchema, AbstractEvalOutputSchema = create_abstract_eval_schema(
            api_module.InputSchema, api_module.OutputSchema
        )

        @assemble_docstring(api_module.abstract_eval)
        def abstract_eval(payload: AbstractEvalInputSchema) -> AbstractEvalOutputSchema:
            """Perform abstract evaluation of the Tesseract on the input data."""
            out = api_module.abstract_eval(payload.inputs)
            return AbstractEvalOutputSchema.model_validate(out)

        endpoints.append(abstract_eval)

    return endpoints
