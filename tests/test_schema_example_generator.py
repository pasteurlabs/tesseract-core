# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import jsonschema
from pydantic import BaseModel, Field
from schema_example_generator import generate_example

from tesseract_core.runtime import Array, Differentiable, Float32, Int32

# NOTE: This is a "test for a test" - not something we do often, but it gives us confidence
# that the example generator is producing valid examples for real-world schemas.


class VolumetricMeshData(BaseModel):
    """Nested model with multiple array types, dict-of-arrays, and plain scalars."""

    n_points: int
    n_cells: int
    points: Differentiable[Array[(None, 3), Float32]]
    num_points_per_cell: Array[(None,), Float32]
    cell_connectivity: Array[(None,), Int32]
    cell_data: dict[str, Array[(None, None), Float32]]
    point_data: dict[str, Array[(None, None), Float32]]


class InputSchema(BaseModel):
    mesh: VolumetricMeshData = Field(
        description="The mesh you want summary statistics of"
    )


def test_generate_example_validates_against_real_schema():
    """Generated example from a real Tesseract schema must pass JSON Schema validation."""
    schema = InputSchema.model_json_schema()
    example = generate_example(schema)
    jsonschema.validate(instance=example, schema=schema)


def test_generate_example_structure():
    """Spot-check structural properties of the generated example."""
    schema = InputSchema.model_json_schema()
    example = generate_example(schema)

    mesh = example["mesh"]

    # Plain integer fields
    assert isinstance(mesh["n_points"], int)
    assert isinstance(mesh["n_cells"], int)

    # Array fields have the expected encoded structure
    for array_field in ("points", "num_points_per_cell", "cell_connectivity"):
        arr = mesh[array_field]
        assert arr["object_type"] == "array"
        assert isinstance(arr["shape"], list)
        assert all(isinstance(d, int) and d > 0 for d in arr["shape"])
        assert isinstance(arr["dtype"], str)
        assert arr["data"]["encoding"] in ("binref", "base64", "json")

    # points shape is 2D (None, 3) -> generated as [1, 1] or similar
    assert len(mesh["points"]["shape"]) == 2

    # 1D arrays
    assert len(mesh["num_points_per_cell"]["shape"]) == 1
    assert len(mesh["cell_connectivity"]["shape"]) == 1

    # dict[str, Array] fields produce empty dicts (no required keys)
    assert mesh["cell_data"] == {}
    assert mesh["point_data"] == {}
