# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from io import TextIOBase
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, TextIO

from pydantic import BaseModel

from .config import get_config
from .schema_generation import (
    create_abstract_eval_schema,
    create_apply_schema,
    create_autodiff_schema,
)


class RegressOutputSchema(BaseModel):
    """Output schema for the regress endpoint."""

    status: Literal["passed", "failed", "error"]
    message: str
    endpoint: str


@contextmanager
def redirect_fd(from_: TextIO, to_: TextIO | int) -> Generator[TextIO, None, None]:
    """Redirect a file descriptor at OS level.

    Args:
        from_: The file object to redirect from.
        to_: The file descriptor or file object to redirect to.

    Yields:
        A writable file object connected to the original file descriptor.
    """
    orig_fd = os.dup(from_.fileno())
    from_.flush()
    if isinstance(to_, TextIOBase):
        to_ = to_.fileno()
    assert isinstance(to_, int)
    os.dup2(to_, from_.fileno())
    orig_fd_file = os.fdopen(orig_fd, "w", closefd=True)
    try:
        yield orig_fd_file
    finally:
        from_.flush()
        os.dup2(orig_fd, from_.fileno())
        orig_fd_file.close()


def load_module_from_path(path: Path | str) -> ModuleType:
    """Load a module from a file path.

    Temporarily puts the module's parent folder on PYTHONPATH to ensure local imports work as expected.
    """
    path = Path(path)

    if not path.is_file():
        raise ImportError(f"Could not load module from {path} (is not a file)")

    module_name = path.stem
    module_dir = path.parent.resolve()

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)

    try:
        old_path = sys.path
        sys.path = [str(module_dir), *old_path]
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(f"Could not load module from {path}") from exc
    finally:
        sys.path = old_path

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
    return load_module_from_path(get_config().api_path)


def get_input_schema(endpoint_function: Callable) -> type[BaseModel]:
    """Get the input schema of an endpoint function."""
    schema = endpoint_function.__annotations__["payload"]
    if not issubclass(schema, BaseModel):
        raise AssertionError(f"Expected BaseModel, got {schema}")
    return schema


def get_output_schema(endpoint_function: Callable) -> type[BaseModel]:
    """Get the output schema of an endpoint function."""
    schema = endpoint_function.__annotations__["return"]
    if not issubclass(schema, BaseModel):
        raise AssertionError(f"Expected BaseModel, got {schema}")
    return schema


def check_tesseract_api(api_module: ModuleType) -> None:
    """Performs basic checks on the Tesseract API module."""
    required_schemas = ("InputSchema", "OutputSchema")
    required_endpoints = ("apply",)

    for schema_name in required_schemas:
        try:
            schema = getattr(api_module, schema_name)
            if not issubclass(schema, BaseModel):
                raise TypeError(
                    f"{schema_name} is not a subclass of pydantic.BaseModel",
                )
        except AttributeError as err:
            raise TypeError(
                f"{schema_name} is not defined in Tesseract API module {api_module.__name__}",
            ) from err

    for endpoint_name in required_endpoints:
        try:
            _ = getattr(api_module, endpoint_name)
        except AttributeError as err:
            raise TypeError(
                f"{endpoint_name} is not defined in Tesseract API module {api_module.__name__}",
            ) from err


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

    from tesseract_core.runtime.testing.regression import TestSpec, regress_test_case

    def regress(payload: TestSpec) -> RegressOutputSchema:
        """Run a single regression test against a Tesseract endpoint.

        Tests an endpoint by calling it with specified inputs and comparing outputs
        against expected values or verifying expected exceptions are raised.

        Args:
            payload: Test specification containing:
                - endpoint: Name of endpoint to test (e.g., "apply", "jacobian")
                - inputs: Input data for the endpoint
                - expected_outputs: Expected output data (mutually exclusive with expected_exception)
                - expected_exception: Expected exception type or name (mutually exclusive with expected_outputs)
                - expected_exception_regex: Optional regex pattern for exception message
                - atol: Absolute tolerance for numeric comparisons (default: 1e-8)
                - rtol: Relative tolerance for numeric comparisons (default: 1e-5)

        Returns:
            RegressOutputSchema with:
                - status: "passed" | "failed" | "error"
                - message: Empty for passed tests, error details for failed/error
                - endpoint: Name of the tested endpoint

        Note:
            This endpoint is designed for testing and CI/CD workflows.
            All outcomes return HTTP 200 with status in the response body regardless of success/failure.
        """
        config = get_config()
        base_dir = Path(config.input_path) if config.input_path else None

        try:
            regress_test_case(
                api_module,
                endpoint_functions={func.__name__: func for func in endpoints},
                test_spec=payload,
                base_dir=base_dir,
                threshold=100,
            )
            return RegressOutputSchema(
                status="passed",
                message="",
                endpoint=payload.endpoint,
            )
        except AssertionError as e:
            return RegressOutputSchema(
                status="failed",
                message=str(e),
                endpoint=payload.endpoint,
            )
        except Exception as e:
            return RegressOutputSchema(
                status="error",
                message=f"{type(e).__name__}: {e}",
                endpoint=payload.endpoint,
            )

    endpoints.append(regress)

    return endpoints
