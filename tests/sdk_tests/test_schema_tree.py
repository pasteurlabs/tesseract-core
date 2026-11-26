# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import namedtuple

import jsonschema
import numpy as np
import pytest

from tesseract_core import Tesseract
from tesseract_core.sdk import schema_tree
from tesseract_core.sdk.schema_tree import (
    AnySchema,
    ArraySchema,
    LiteralSchema,
    ObjectSchema,
    ScalarSchema,
    SequenceSchema,
    UnionSchema,
    UnsupportedSchema,
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
def openapi(tmp_path_factory):
    api = tmp_path_factory.mktemp("schema_tree") / "tesseract_api.py"
    api.write_text(TESSERACT_API)
    return Tesseract.from_tesseract_api(api).openapi_schema


@pytest.fixture(scope="module")
def input_schema(openapi):
    return schema_tree.parse(openapi, "#/components/schemas/Apply_InputSchema")


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
def test_parse_reconstructs_field_types(input_schema, field, expected):
    assert input_schema.properties[field] == expected


def _array_dtypes(node, leaf):
    """Replace arrays with their declared dtype, so tests can see which node each got."""
    return ("array", node.dtype) if isinstance(node, ArraySchema) else leaf


def test_map_reaches_every_array_and_keeps_containers(input_schema):
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
    out = schema_tree.map(_array_dtypes, input_schema, payload)

    f32 = ("array", "float32")
    assert out == {
        "vec": f32,
        "scalar_arr": ("array", "int32"),
        "anydim": ("array", None),
        "opt": f32,
        "seq": [f32, f32],
        "tup": (f32, 5),
        "nt": NT(("array", "float64"), 5),
        "table": {"k": [{"x": f32, "label": "b"}]},
        "either": {"x": f32},
        "mode": "slow",
        "count": 3,
        "extra": {"not": "an array"},
    }
    assert type(out["nt"]) is NT


def test_map_with_path_gives_the_keys_and_indices_to_each_leaf(input_schema):
    payload = {"table": {"k": [{"x": np.ones(3)}]}, "tup": (np.ones(3), 5)}
    out = schema_tree.map_with_path(
        lambda path, node, leaf: path if isinstance(node, ArraySchema) else leaf,
        input_schema,
        payload,
    )
    assert out == {"table": {"k": [{"x": "table.{k}.[0].x"}]}, "tup": ("tup.[0]", 5)}


def test_map_with_path_escapes_dict_keys(input_schema):
    """A dict key can hold dots and braces; Tesseract's path format escapes it."""
    out = schema_tree.map_with_path(
        lambda path, node, leaf: path,
        input_schema,
        {"table": {"a.b{c}": [{"x": np.ones(3)}]}},
    )
    path = out["table"]["a.b{c}"][0]["x"]
    assert path == r"table.{a.b\{c\}}.[0].x"


def test_map_with_path_matches_the_runtime_path_format(input_schema):
    """The runtime's own path lookup finds each leaf from the path it was given."""
    from tesseract_core.runtime.tree_transforms import get_at_path

    payload = {
        "vec": np.ones((2, 3)),
        "tup": (np.ones(3), 5),
        "table": {"a.b{c}": [{"x": np.ones(3), "label": "b"}]},
    }
    paths = []
    schema_tree.map_with_path(
        lambda path, node, leaf: paths.append((path, leaf)), input_schema, payload
    )
    assert len(paths) == 5
    for path, leaf in paths:
        assert get_at_path(payload, path) is leaf


def test_map_pairs_every_leaf_with_its_node(input_schema):
    payload = {
        "mode": "slow",
        "count": 3,
        "extra": {"not": "an array"},
        "table": {"k": [{"x": np.ones(3), "label": "b"}]},
        "undeclared": 1,
    }
    out = schema_tree.map(lambda node, leaf: type(node).__name__, input_schema, payload)
    assert out == {
        "mode": "LiteralSchema",
        "count": "ScalarSchema",
        "extra": "AnySchema",  # not walked into: the schema doesn't describe it
        "table": {"k": [{"x": "ArraySchema", "label": "ScalarSchema"}]},
        "undeclared": "NoneType",  # a key the schema doesn't declare
    }


def test_map_is_leaf_stops_the_walk(input_schema):
    out = schema_tree.map(
        lambda node, leaf: (type(node).__name__, len(leaf)),
        input_schema,
        {"seq": [np.ones(3), np.ones(3)]},
        is_leaf=lambda value: isinstance(value, list),
    )
    assert out == {"seq": ("SequenceSchema", 2)}


def test_map_picks_union_member_by_value(input_schema):
    either = input_schema.properties["either"]
    assert schema_tree.map(_array_dtypes, either, np.ones(3)) == (
        "array",
        "float32",
    )
    assert schema_tree.map(_array_dtypes, either, {"x": np.ones(3)}) == {
        "x": ("array", "float32")
    }
    # None is a valid value of an optional array, not an array
    opt = input_schema.properties["opt"]
    assert schema_tree.map(_array_dtypes, opt, None) is None


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
    assert schema_tree.parse(schema).dtype == expected


@pytest.mark.parametrize("array_dtype", ["float32", None])
def test_array_dtype_takes_precedence(array_dtype):
    schema = {
        "array_flags": [],
        "array_dtype": array_dtype,
        "properties": {"dtype": {"examples": ["int8"], "enum": ["int8"]}},
    }
    assert schema_tree.parse(schema).dtype == array_dtype


@pytest.mark.parametrize("field", ["pet", "tagged_pet"])
@pytest.mark.parametrize("kind, dtype", [("cat", "float32"), ("dog", "float64")])
def test_map_picks_model_by_literal_tag(input_schema, field, kind, dtype):
    """Models with the same keys are told apart by their ``Literal`` field."""
    out = schema_tree.map(
        _array_dtypes, input_schema.properties[field], {"kind": kind, "x": np.ones(3)}
    )
    assert out == {"kind": kind, "x": ("array", dtype)}


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

    return schema_tree.parse(Fields.model_json_schema())


F32_LEAF, F64_LEAF = ("array", "float32"), ("array", "float64")


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("rank", np.ones(3), F32_LEAF),  # told apart by number of dimensions
        ("rank", np.ones((2, 3)), F64_LEAF),
        ("tuples", (np.ones(3), 1), (F32_LEAF, 1)),  # by length
        ("tuples", (np.ones(3), 1, 2), (F64_LEAF, 1, 2)),
        ("sub_or_any", {"x": np.ones(3)}, {"x": F32_LEAF}),  # Any is the fallback
        ("scalar_or_arr", 1.5, 1.5),  # a Python float is a JSON number
        (
            "scalar_or_arr",
            np.float64(1.5),
            F32_LEAF,
        ),  # numpy scalars are sent as arrays
        ("lit_or_arr", "auto", "auto"),
        ("lit_or_arr", np.ones(3), F32_LEAF),
    ],
)
def test_map_picks_union_member_that_accepts_value(field, value, expected):
    out = schema_tree.map(_array_dtypes, _union_fields_schema(), {field: value})
    assert out[field] == expected


ARRAY = {"array_flags": [], "array_dtype": "float32"}
INT, STR, NULL = ScalarSchema("integer"), ScalarSchema("string"), ScalarSchema("null")


@pytest.mark.parametrize(
    "schema, expected",
    [
        ({"type": ["string", "null"]}, UnionSchema((STR, NULL))),
        (
            {"type": "integer", "nullable": True},
            UnionSchema((INT, NULL)),
        ),  # OpenAPI 3.0
        ({"oneOf": [{"type": "integer"}, {"type": "string"}]}, UnionSchema((INT, STR))),
        (
            {"allOf": [{"$ref": "#/components/schemas/Arr"}], "description": "x"},
            ArraySchema((None,), "float32", False),
        ),
        ({"type": "array", "items": {"type": "integer"}}, SequenceSchema((), INT)),
        (
            {"type": "array", "prefixItems": [{"type": "integer"}], "items": False},
            SequenceSchema((INT,), None),
        ),
        ({"type": "object"}, ObjectSchema({}, frozenset(), AnySchema())),
        (
            {"type": "object", "additionalProperties": {"type": "integer"}},
            ObjectSchema({}, frozenset(), INT),
        ),
        ({"const": 3}, LiteralSchema((3,))),
        ({}, AnySchema()),
        (True, AnySchema()),
        (
            {
                "array_flags": [],
                "array_dtype": "int8",
                "properties": {
                    "shape": {
                        "prefixItems": [
                            {"const": 1},
                            {"enum": [1, 3]},
                            {"type": "integer"},
                        ]
                    }
                },
            },
            ArraySchema((1, 3, None), "int8", False),
        ),
        (  # an array from a Tesseract built before array_flags/array_dtype
            {
                "properties": {
                    "object_type": {"const": "array"},
                    "shape": {"items": {"type": "integer"}},
                    "dtype": {"examples": ["int8"], "enum": ["int8"]},
                }
            },
            ArraySchema(None, "int8", False),
        ),
    ],
)
def test_parse_handles_schemas_pydantic_does_not_emit(schema, expected):
    """OpenAPI and JSON Schema written by hand, or by other tools than pydantic."""
    arr = {**ARRAY, "properties": {"shape": {"prefixItems": [{"type": "integer"}]}}}
    document = schema
    if isinstance(schema, dict):
        document = {**schema, "components": {"schemas": {"Arr": arr}}}
    assert schema_tree.parse(document) == expected


def test_union_member_picked_by_json_type_and_closed_keys():
    ints_or_bools = schema_tree.parse({"type": ["integer", "boolean"]})
    pick = lambda node, leaf: node.type if node else None
    assert schema_tree.map(pick, ints_or_bools, 1) == "integer"
    assert schema_tree.map(pick, ints_or_bools, True) == "boolean"

    closed = {"type": "object", "additionalProperties": False}
    a_or_b = schema_tree.parse(
        {
            "anyOf": [
                {**closed, "properties": {"a": {"type": "integer"}}},
                {**closed, "properties": {"b": {"type": "string"}}},
            ]
        }
    )
    assert schema_tree.map(pick, a_or_b, {"a": 1}) == {"a": "integer"}


def test_iter_nodes_lists_every_node(input_schema):
    kinds = {type(n).__name__ for n in schema_tree.iter_nodes(input_schema)}
    assert kinds == {
        "ObjectSchema",
        "ArraySchema",
        "UnionSchema",
        "SequenceSchema",
        "LiteralSchema",
        "ScalarSchema",
        "AnySchema",
    }
    arrays = [
        n for n in schema_tree.iter_nodes(input_schema) if isinstance(n, ArraySchema)
    ]
    # vec, scalar_arr, anydim, opt, seq, tup, nt, table, 2 in either, 2 in each pet
    assert len(arrays) == 14


def test_unsupported_part_is_reported_and_the_rest_still_parsed():
    """A schema the parser can't handle shouldn't stop other arrays being found."""
    schema = {
        "properties": {
            "x": ARRAY,
            "loop": {"$ref": "#/$defs/Loop"},
            "both": {"allOf": [{"type": "object"}, {"type": "object"}]},
        },
        "$defs": {"Loop": {"properties": {"again": {"$ref": "#/$defs/Loop"}}}},
    }
    node = schema_tree.parse(schema)

    unsupported = [
        n for n in schema_tree.iter_nodes(node) if isinstance(n, UnsupportedSchema)
    ]
    assert [n.reason for n in unsupported] == [
        "cannot parse the recursive $ref '#/$defs/Loop'",
        "cannot parse the 'allOf' with 2 members",
    ]
    out = schema_tree.map(_array_dtypes, node, {"x": np.ones(2)})
    assert out == {"x": ("array", "float32")}


def _every_field_kind_model():
    from typing import Any, Literal, NamedTuple

    from pydantic import BaseModel, Field

    from tesseract_core.runtime import Array, Differentiable, Float32, Float64, Int32

    class Pair(NamedTuple):
        a: Array[(2,), Float64]
        n: int

    class Sub(BaseModel):
        x: Differentiable[Array[(None,), Float32]]
        label: str

    class EveryFieldKind(BaseModel):
        scalar_arr: Array[(), Int32]
        anydim: Array[..., None]
        opt: Array[(None,), Float32] | None
        tup: tuple[Array[(None,), Float32], int]
        pair: Pair
        table: dict[str, list[Sub]]
        subs: list[Sub]
        either: Sub | Array[(None,), Float32]
        mode: Literal["fast", "slow"]
        ratio: float = Field(gt=0)
        scale: float
        count: int = Field(gt=0)
        name: str = Field(examples=["a name"])
        flag: bool
        extra: Any

    return EveryFieldKind


def test_example_validates_against_its_model():
    model = _every_field_kind_model()
    schema = model.model_json_schema()
    example = schema_tree._example(schema_tree.parse(schema))
    jsonschema.validate(instance=example, schema=schema)
    model.model_validate_json(json.dumps(example))


SUB = {"properties": {"x": ARRAY}}


@pytest.mark.parametrize(
    "document",
    [
        pytest.param({"properties": {"sub": SUB}}, id="inline"),
        pytest.param(
            {"properties": {"sub": {"$ref": "#/$defs/Sub"}}, "$defs": {"Sub": SUB}},
            id="$defs",  # JSON Schema 2020-12, pydantic v2
        ),
        pytest.param(
            {
                "properties": {"sub": {"$ref": "#/definitions/Sub"}},
                "definitions": {"Sub": SUB},
            },
            id="definitions",  # draft-07, pydantic v1 and many other tools
        ),
        pytest.param(
            {
                "properties": {"sub": {"$ref": "#/components/schemas/Sub"}},
                "components": {"schemas": {"Sub": SUB}},
            },
            id="components",  # OpenAPI
        ),
        pytest.param(
            {
                "properties": {
                    "sub": {"$ref": "#/properties/sub/$defs/Sub", "$defs": {"Sub": SUB}}
                }
            },
            id="nested $defs",
        ),
        pytest.param(
            {
                "properties": {"sub": {"$ref": "#/$defs/Holder/properties/inner"}},
                "$defs": {"Holder": {"properties": {"inner": SUB}}},
            },
            id="ref into a definition",
        ),
        pytest.param(
            {
                "properties": {"sub": {"$ref": "#/$defs/a~1b~0c"}},
                "$defs": {"a/b~c": SUB},
            },
            id="escaped pointer",
        ),
    ],
)
def test_parse_follows_refs_wherever_definitions_live(document):
    expected = ObjectSchema(
        {"x": ArraySchema((), "float32", False)}, frozenset(), AnySchema()
    )
    assert schema_tree.parse(document).properties["sub"] == expected


def test_refs_that_cannot_be_followed_are_reported():
    node = schema_tree.parse(
        {
            "properties": {
                "external": {"$ref": "other.json#/$defs/Sub"},
                "missing": {"$ref": "#/$defs/Missing"},
                "itself": {"$ref": "#"},
            }
        }
    )
    reasons = {k: v.reason for k, v in node.properties.items()}
    assert reasons == {
        "external": "cannot parse the $ref 'other.json#/$defs/Sub', which doesn't point into the document",
        "missing": "cannot parse the $ref '#/$defs/Missing', which doesn't point into the document",
        "itself": "cannot parse the recursive $ref '#'",
    }


@pytest.mark.parametrize(
    "pointer", ["#/components/schemas/Missing", "Apply_InputSchema"]
)
def test_parse_rejects_a_pointer_outside_the_document(pointer):
    with pytest.raises(ValueError, match="does not point into the document"):
        schema_tree.parse({"components": {"schemas": {}}}, pointer)


def _differentiable_paths(node, results):
    """Path patterns of the differentiable arrays below ``node``, as a ``fold`` function."""
    if isinstance(node, ArraySchema):
        return {""} if node.differentiable else set()
    if isinstance(node, UnionSchema):
        return set().union(*results.values())
    paths = set()
    for key, below in results.items():
        if isinstance(node, SequenceSchema):
            segment = "[]"
        else:
            segment = "{}" if key is ... else key
        paths |= {f"{segment}.{p}" if p else segment for p in below}
    return paths


def test_fold_derives_the_published_differentiable_arrays(openapi, input_schema):
    """A use of fold besides examples: the paths tesseract-core publishes."""
    published = openapi["components"]["schemas"]["ApplyInputSchema"]
    assert schema_tree.fold(_differentiable_paths, input_schema) == set(
        published["differentiable_arrays"]
    )


def test_fold_passes_children_results_by_key():
    seen = []

    def record(node, results):
        seen.append((type(node).__name__, sorted(map(repr, results))))
        return type(node).__name__

    schema_tree.fold(
        record,
        schema_tree.parse(
            {
                "properties": {"a": {"type": "integer"}},
                "additionalProperties": {"type": "string"},
                "required": ["a"],
            }
        ),
    )
    assert seen[-1] == ("ObjectSchema", ["'a'", "Ellipsis"])
