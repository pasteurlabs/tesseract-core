# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import numpy as np
import pytest

from tesseract_core import Tesseract
from tesseract_core.sdk.schema_tree import (
    AnySchema,
    ArraySchema,
    LiteralSchema,
    ObjectSchema,
    ScalarSchema,
    SequenceSchema,
    UnionSchema,
    map_arrays,
    parse_schema,
)

TESSERACT_API = """
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, Float64, Int32


class Sub(BaseModel):
    x: Differentiable[Array[(None,), Float32]]
    label: str = "a"


class Cat(BaseModel):
    kind: Literal["cat"]
    x: Array[(None,), Float32]


class Dog(BaseModel):
    kind: Literal["dog"]
    x: Array[(None,), Float64]


class NT(NamedTuple):
    a: Array[(2,), Float64]
    n: int


class InputSchema(BaseModel):
    vec: Differentiable[Array[(None, 3), Float32]]
    scalar_arr: Array[(), Int32]
    anydim: Array[..., None]
    opt: Array[(None,), Float32] | None = None
    seq: list[Array[(None,), Float32]] = []
    tup: tuple[Array[(None,), Float32], int]
    nt: NT
    table: dict[str, list[Sub]] = {}
    either: Sub | Differentiable[Array[(None,), Float32]]
    mode: Literal["fast", "slow"] = "fast"
    count: int = 1
    ratio: float = 0.5
    flag: bool = True
    name: str = "x"
    extra: Any = None
    pet: Cat | Dog | None = None
    tagged_pet: Annotated[Cat | Dog, Field(discriminator="kind")] | None = None


class OutputSchema(BaseModel):
    y: Array[(None, 3), Float32]


def apply(inputs):
    return OutputSchema(y=inputs.vec)
"""

F32 = ArraySchema(shape=(None,), dtype="float32", differentiable=False)
DIFF_F32 = ArraySchema(shape=(None,), dtype="float32", differentiable=True)
SUB = ObjectSchema(
    properties={"x": DIFF_F32, "label": ScalarSchema("string")},
    required=frozenset({"x"}),
    additional=None,
)


@pytest.fixture(scope="module")
def input_schema(tmp_path_factory):
    api = tmp_path_factory.mktemp("schema_tree") / "tesseract_api.py"
    api.write_text(TESSERACT_API)
    schemas = Tesseract.from_tesseract_api(api).openapi_schema["components"]["schemas"]
    return parse_schema(schemas["Apply_InputSchema"], schemas)


@pytest.mark.parametrize(
    "field, expected",
    [
        ("vec", ArraySchema((None, 3), "float32", differentiable=True)),
        ("scalar_arr", ArraySchema((), "int32", differentiable=False)),
        ("anydim", ArraySchema(None, None, differentiable=False)),
        ("opt", UnionSchema((F32, ScalarSchema("null")))),
        ("seq", SequenceSchema(prefix=(), items=F32)),
        (
            "tup",
            SequenceSchema(prefix=(F32, ScalarSchema("integer")), items=None),
        ),
        (
            "nt",
            SequenceSchema(
                prefix=(ArraySchema((2,), "float64", False), ScalarSchema("integer")),
                items=None,
            ),
        ),
        (
            "table",
            ObjectSchema(
                properties={},
                required=frozenset(),
                additional=SequenceSchema(prefix=(), items=SUB),
            ),
        ),
        ("either", UnionSchema((SUB, DIFF_F32))),
        ("mode", LiteralSchema(("fast", "slow"))),
        ("count", ScalarSchema("integer")),
        ("ratio", ScalarSchema("number")),
        ("flag", ScalarSchema("boolean")),
        ("name", ScalarSchema("string")),
        ("extra", AnySchema()),
    ],
)
def test_parse_schema_reconstructs_field_types(input_schema, field, expected):
    assert input_schema.properties[field] == expected


def test_map_arrays_reaches_every_array_and_keeps_containers(input_schema):
    NT = namedtuple("NT", "a n")
    x = np.ones(3)
    payload = {
        "vec": np.ones((2, 3)),
        "scalar_arr": 1.0,
        "anydim": x,
        "opt": x,
        "seq": [x, x],
        "tup": (x, 5),
        "nt": NT(x, 5),
        "table": {"k": [{"x": x, "label": "b"}]},
        "either": {"x": x},
        "mode": "slow",
        "count": 3,
        "extra": {"not": "an array"},
    }
    seen = {}

    def record(leaf, array_schema, path):
        seen[path] = array_schema.dtype
        return ("mapped", leaf)

    out = map_arrays(record, payload, input_schema)

    assert seen == {
        ("vec",): "float32",
        ("scalar_arr",): "int32",
        ("anydim",): None,
        ("opt",): "float32",
        ("seq", 0): "float32",
        ("seq", 1): "float32",
        ("tup", 0): "float32",
        ("nt", 0): "float64",
        ("table", "k", 0, "x"): "float32",
        ("either", "x"): "float32",
    }
    assert type(out["nt"]) is NT
    assert out["nt"].n == 5
    assert type(out["tup"]) is tuple
    assert out["table"]["k"][0]["label"] == "b"
    assert out["either"]["x"][0] == "mapped"
    assert out["extra"] == {"not": "an array"}


def test_map_arrays_picks_union_member_by_value(input_schema):
    def mark(leaf, array_schema, path):
        return "mapped"

    either = input_schema.properties["either"]
    assert map_arrays(mark, np.ones(3), either) == "mapped"
    assert map_arrays(mark, {"x": np.ones(3)}, either) == {"x": "mapped"}
    # None is a valid value of an optional array, not an array
    assert map_arrays(mark, None, input_schema.properties["opt"]) is None


@pytest.mark.parametrize(
    "dtype_property, expected",
    [
        ({"examples": ["int8"], "enum": ["int8"]}, "int8"),
        # an open field also lists float64 first, but accepts complex input
        ({"examples": ["float64"], "enum": ["float64", "complex128"]}, None),
        ({"examples": ["float64"], "enum": ["float32", "float64"]}, "float64"),
        ({"type": "string"}, None),
    ],
)
def test_array_dtype_from_schemas_without_array_dtype(dtype_property, expected):
    """Tesseracts built before ``array_dtype`` existed only list it in ``examples``."""
    schema = {"array_flags": [], "properties": {"dtype": dtype_property}}
    assert parse_schema(schema).dtype == expected


@pytest.mark.parametrize("array_dtype", ["float32", None])
def test_array_dtype_takes_precedence(array_dtype):
    schema = {
        "array_flags": [],
        "array_dtype": array_dtype,
        "properties": {"dtype": {"examples": ["int8"], "enum": ["int8"]}},
    }
    assert parse_schema(schema).dtype == array_dtype


@pytest.mark.parametrize("field", ["pet", "tagged_pet"])
@pytest.mark.parametrize("kind, dtype", [("cat", "float32"), ("dog", "float64")])
def test_map_arrays_picks_model_by_literal_tag(input_schema, field, kind, dtype):
    """Models with the same keys are told apart by their ``Literal`` field."""
    out = map_arrays(
        lambda leaf, array_schema, path: array_schema.dtype,
        {"kind": kind, "x": np.ones(3)},
        input_schema.properties[field],
    )
    assert out == {"kind": kind, "x": dtype}


def _union_fields_schema():
    from typing import Any, Literal

    from pydantic import BaseModel

    from tesseract_core.runtime import Array, Float32, Float64

    class Sub(BaseModel):
        x: Array[(None,), Float32]

    class Fields(BaseModel):
        rank: Array[(None,), Float32] | Array[(None, None), Float64]
        tuples: (
            tuple[Array[(None,), Float32], int]
            | tuple[Array[(None,), Float64], int, int]
        )
        sub_or_any: Sub | Any
        scalar_or_arr: float | Array[(), Float32]
        lit_or_arr: Literal["auto"] | Array[(None,), Float32]

    return parse_schema(Fields.model_json_schema())


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("rank", np.ones(3), "float32"),  # told apart by number of dimensions
        ("rank", np.ones((2, 3)), "float64"),
        ("tuples", (np.ones(3), 1), ("float32", 1)),  # by length
        ("tuples", (np.ones(3), 1, 2), ("float64", 1, 2)),
        ("sub_or_any", {"x": np.ones(3)}, {"x": "float32"}),  # Any is the fallback
        ("scalar_or_arr", 1.5, 1.5),  # a Python float is a JSON number
        (
            "scalar_or_arr",
            np.float64(1.5),
            "float32",
        ),  # a numpy scalar is sent as an array
        ("lit_or_arr", "auto", "auto"),
        ("lit_or_arr", np.ones(3), "float32"),
    ],
)
def test_map_arrays_picks_union_member_that_accepts_value(field, value, expected):
    schema = _union_fields_schema()
    out = map_arrays(
        lambda leaf, array_schema, path: array_schema.dtype,
        {field: value},
        schema,
    )
    assert out[field] == expected


def test_unsupported_part_is_treated_as_any_and_the_rest_still_parsed():
    """A schema the parser can't handle shouldn't stop other arrays being found."""
    array = {"array_flags": [], "array_dtype": "float32"}
    defs = {"Loop": {"properties": {"again": {"$ref": "#/$defs/Loop"}}}}
    schema = {
        "properties": {
            "x": array,
            "loop": {"$ref": "#/$defs/Loop"},
            "both": {"allOf": [{"type": "object"}, {"type": "object"}]},
        }
    }
    with (
        pytest.warns(UserWarning, match="cannot parse the recursive definition 'Loop'"),
        pytest.warns(UserWarning, match="cannot parse the 'allOf' with 2 members"),
    ):
        node = parse_schema(schema, defs)

    assert node.properties["both"] == AnySchema()
    assert node.properties["loop"].properties["again"] == AnySchema()
    out = map_arrays(lambda leaf, a, path: a.dtype, {"x": np.ones(2)}, node)
    assert out == {"x": "float32"}
