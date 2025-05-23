# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for cuda-image
# Generated by tesseract 0.8.2.dev14+g494ed54.d20250401 on 2025-04-04T18:55:51.909676


from pydantic import BaseModel

#
# Schemas
#


class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    pass


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Assert that CUDA is available."""
    from ctypes.util import find_library

    cudart = find_library("cudart")
    assert cudart is not None, "CUDA runtime library not found"
    return OutputSchema()


#
# Optional endpoints
#

# import numpy as np

# def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
#     return {}

# def jacobian_vector_product(
#     inputs: InputSchema,
#     jvp_inputs: set[str],
#     jvp_outputs: set[str],
#     tangent_vector: dict[str, np.typing.ArrayLike]
# ) -> dict[str, np.typing.ArrayLike]:
#     return {}

# def vector_jacobian_product(
#     inputs: InputSchema,
#     vjp_inputs: set[str],
#     vjp_outputs: set[str],
#     cotangent_vector: dict[str, np.typing.ArrayLike]
# ) -> dict[str, np.typing.ArrayLike]:
#     return {}

# def abstract_eval(abstract_inputs):
#     return {}
