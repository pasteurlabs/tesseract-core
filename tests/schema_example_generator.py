# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal JSON Schema example data generator.

Only supports the subset of JSON Schema features used by Pydantic-generated
schemas in this project.
"""

from __future__ import annotations

from typing import Any


def generate_example(schema: dict) -> Any:
    """Generate a structurally valid example object from a JSON Schema."""
    defs = schema.get("$defs", {})
    return _generate(schema, defs)


def _resolve(schema: dict, defs: dict) -> dict:
    if "$ref" in schema:
        ref_path = schema["$ref"]
        # Assume ref_path is in the format #/$defs/Name
        ref_name = ref_path.split("/")[-1]
        return defs[ref_name]
    return schema


def _generate(schema: dict, defs: dict) -> Any:
    schema = _resolve(schema, defs)

    # const / enum take priority
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]

    # Composition keywords
    if "oneOf" in schema:
        return _generate(schema["oneOf"][0], defs)

    if "anyOf" in schema:
        for variant in schema["anyOf"]:
            resolved = _resolve(variant, defs)
            if resolved.get("type") != "null":
                return _generate(variant, defs)
        return None

    # Type dispatch
    typ = schema.get("type")

    if typ == "object" or (typ is None and "properties" in schema):
        result = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            result[prop_name] = _generate(prop_schema, defs)
        return result

    if typ == "array":
        if "prefixItems" in schema:
            return [_generate(item, defs) for item in schema["prefixItems"]]
        items_schema = schema.get("items", {})
        count = schema.get("minItems", 1)
        return [_generate(items_schema, defs) for _ in range(count)]

    if typ == "string":
        min_len = max(schema.get("minLength", 1), 1)
        return "a" * min_len

    if typ == "integer":
        if "exclusiveMinimum" in schema:
            return schema["exclusiveMinimum"] + 1
        return schema.get("minimum", 0)

    if typ == "number":
        if "exclusiveMinimum" in schema:
            return schema["exclusiveMinimum"] + 0.1
        return float(schema.get("minimum", 0.0))

    if typ == "boolean":
        return True

    if typ == "null":
        return None

    # Fallback
    return None
