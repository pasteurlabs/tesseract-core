# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal JSON Schema example data generator.

Only supports the subset of JSON Schema features used by Pydantic-generated
schemas in this project.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

import numpy as np

from tesseract_core.sdk.schema_tree import (
    ArraySchema,
    LiteralSchema,
    ObjectSchema,
    ScalarSchema,
    SchemaNode,
    SequenceSchema,
    UnionSchema,
    parse_schema,
)


def generate_example(
    schema: Mapping, definitions: Mapping | None = None, seed: int = 42
) -> Any:
    """Generate an example object that validates against a JSON Schema.

    Arrays get random data in [0, 100) of their declared dtype (float64 if
    open), base64-encoded, with every dimension of size 1 so it broadcasts to
    any fixed size. ``definitions`` are the schemas ``$ref`` entries point to,
    ``schema["$defs"]`` by default.
    """
    rng = np.random.RandomState(seed)
    return _generate(parse_schema(schema, definitions), rng)


def _generate(node: SchemaNode, rng: np.random.RandomState) -> Any:
    raw = node.schema

    if isinstance(node, ArraySchema):
        return _generate_array(node, rng)
    if isinstance(node, LiteralSchema):
        return node.values[0]
    if raw.get("examples"):
        return raw["examples"][0]

    if isinstance(node, UnionSchema):
        for member in node.members:
            if not (isinstance(member, ScalarSchema) and member.type == "null"):
                return _generate(member, rng)
        return None

    if isinstance(node, ObjectSchema):
        return {k: _generate(v, rng) for k, v in node.properties.items()}

    if isinstance(node, SequenceSchema):
        if node.prefix:
            return [_generate(item, rng) for item in node.prefix]
        return [_generate(node.items, rng) for _ in range(raw.get("minItems", 1))]

    if isinstance(node, ScalarSchema):
        if node.type == "string":
            return "a" * max(raw.get("minLength", 1), 1)
        if node.type == "integer":
            if "exclusiveMinimum" in raw:
                return raw["exclusiveMinimum"] + 1
            return raw.get("minimum", 0)
        if node.type == "number":
            if "exclusiveMinimum" in raw:
                return raw["exclusiveMinimum"] + 0.1
            return float(raw.get("minimum", 0.0))
        if node.type == "boolean":
            return True

    return None


def _generate_array(node: ArraySchema, rng: np.random.RandomState) -> dict:
    shape = (1,) if node.shape is None else (1,) * len(node.shape)
    dtype = node.dtype or "float64"
    data = rng.uniform(0, 100, shape).astype(dtype)
    return {
        "object_type": "array",
        "shape": list(shape),
        "dtype": dtype,
        "data": {
            "buffer": base64.b64encode(data.tobytes()).decode(),
            "encoding": "base64",
        },
    }
