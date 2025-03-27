# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Union

from .config import get_config
from .schema_generation import (
    create_abstract_eval_schema,
    create_apply_schema,
    create_autodiff_schema,
)


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
            out = api_module.jacobian(**dict(payload))
            return JacobianOutputSchema.model_validate(
                out,
                context={
                    "output_keys": payload.jac_outputs,
                    "input_keys": payload.jac_inputs,
                },
            )

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
            out = api_module.jacobian_vector_product(**dict(payload))
            return JVPOutputSchema.model_validate(
                out, context={"output_keys": payload.jvp_outputs}
            )

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
            out = api_module.vector_jacobian_product(**dict(payload))
            return VJPOutputSchema.model_validate(
                out, context={"input_keys": payload.vjp_inputs}
            )

        endpoints.append(vector_jacobian_product)

    def health() -> dict[str, Any]:
        """Get health status of the Tesseract instance."""
        return {"status": "ok"}

    endpoints.append(health)

    def input_schema() -> dict[str, Any]:
        """Get input schema for tesseract apply function."""
        return ApplyInputSchema.model_json_schema()

    endpoints.append(input_schema)

    def output_schema() -> dict[str, Any]:
        """Get output schema for tesseract apply function."""
        return ApplyOutputSchema.model_json_schema()

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
