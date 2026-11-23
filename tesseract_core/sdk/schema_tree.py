# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A typed view of the JSON Schema a Tesseract publishes for its endpoints.

Clients only receive a Tesseract's OpenAPI schema, not the pydantic models it
was generated from. :func:`parse_schema` turns that schema into a tree of
plain dataclasses, one per kind of type pydantic emits, so client code can ask
where a payload's arrays are and what shape and dtype they should have without
re-implementing JSON Schema traversal. :func:`schema_map` maps a function
over a payload's leaves alongside that tree, and :func:`iter_nodes` lists its
nodes.

Every node keeps the (resolved) JSON Schema it was parsed from as ``schema``,
for constraints that aren't modelled here, such as ``minimum`` or
``examples``.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Literal

import numpy as np

__all__ = [
    "AnySchema",
    "ArraySchema",
    "LiteralSchema",
    "ObjectSchema",
    "ScalarSchema",
    "SchemaNode",
    "SequenceSchema",
    "UnionSchema",
    "UnsupportedSchema",
    "iter_nodes",
    "parse_schema",
    "schema_map",
    "schema_map_with_path",
]

ScalarType = Literal["string", "number", "integer", "boolean", "null"]


@dataclass(frozen=True)
class SchemaNode:
    """Base class of the nodes :func:`parse_schema` returns.

    Each subclass says which schemas it describes (``_matches``) and how to
    build itself from one (``_from_schema``). :func:`parse_schema` tries them
    in the order of ``_NODE_TYPES``.
    """

    schema: Mapping = field(
        default_factory=dict, repr=False, compare=False, kw_only=True
    )

    @classmethod
    def _matches(cls, schema: Mapping) -> bool:
        """Whether ``schema`` (with any ``$ref`` resolved) is of this kind."""
        raise NotImplementedError

    @classmethod
    def _from_schema(cls, schema: Mapping, parser: _SchemaParser) -> SchemaNode:
        """Build the node for ``schema``, using ``parser`` for its sub-schemas."""
        raise NotImplementedError

    def _accepts(self, value: Any) -> bool:
        """Whether ``value`` could be meant for this node, judged by its top level.

        Used to pick a union's member. Only checks what is cheap to check
        without descending into ``value``.
        """
        raise NotImplementedError

    def _children(self) -> dict[Any, SchemaNode]:
        """The nodes directly below this one, by property name or position.

        ``...`` stands for any other key or item (``additionalProperties`` or
        ``items``).
        """
        return {}


@dataclass(frozen=True)
class ArraySchema(SchemaNode):
    """An ``Array[...]`` field.

    ``shape`` is ``None`` for an ``Array[..., ...]`` field, with ``None`` for
    each dimension of any size. ``dtype`` is ``None`` for a field that accepts
    any dtype.
    """

    shape: tuple[int | None, ...] | None
    dtype: str | None
    differentiable: bool

    @classmethod
    def _matches(cls, schema: Mapping) -> bool:
        """``array_flags``, or ``object_type: "array"`` in older schemas."""
        if "array_flags" in schema:
            return True
        object_type = schema.get("properties", {}).get("object_type", {})
        return object_type.get("const") == "array"

    @classmethod
    def _from_schema(cls, schema: Mapping, parser: _SchemaParser) -> ArraySchema:
        return cls(
            shape=_array_shape(schema),
            dtype=_array_dtype(schema),
            differentiable="DIFFERENTIABLE" in schema.get("array_flags", ()),
            schema=schema,
        )

    def _accepts(self, value: Any) -> bool:
        if not (hasattr(value, "shape") and hasattr(value, "dtype")):
            return False
        return self.shape is None or len(value.shape) == len(self.shape)


def _array_shape(schema: Mapping) -> tuple[int | None, ...] | None:
    shape_schema = schema.get("properties", {}).get("shape", {})
    if "items" in shape_schema:  # Array[..., dtype]
        return None
    dims = []
    for dim in shape_schema.get("prefixItems", ()):
        if "const" in dim:
            dims.append(dim["const"])
        elif "enum" in dim:  # [1, N]: N, or 1 to broadcast to it
            dims.append(max(dim["enum"]))
        else:
            dims.append(None)
    return tuple(dims)


def _array_dtype(schema: Mapping) -> str | None:
    """The dtype an array field declares, or ``None`` if it accepts any.

    Read from ``array_dtype`` where the schema has it (tesseract-core#898).
    Older schemas only carry the dtype in the ``dtype`` property's
    ``examples``, with ``float64`` there for an open field too; an open field
    also accepts complex input, which a ``Float64`` field does not.
    """
    if "array_dtype" in schema:
        return schema["array_dtype"]
    dtype_schema = schema.get("properties", {}).get("dtype", {})
    name = (dtype_schema.get("examples") or [None])[0]
    if name == "float64" and "complex128" in dtype_schema.get("enum", ()):
        return None
    return name


@dataclass(frozen=True)
class LiteralSchema(SchemaNode):
    """A ``const`` or ``enum``: one of a fixed set of values."""

    values: tuple

    @classmethod
    def _matches(cls, schema: Mapping) -> bool:
        return "const" in schema or "enum" in schema

    @classmethod
    def _from_schema(cls, schema: Mapping, parser: _SchemaParser) -> LiteralSchema:
        values = (schema["const"],) if "const" in schema else tuple(schema["enum"])
        return cls(values, schema=schema)

    def _accepts(self, value: Any) -> bool:
        # Literals are JSON values; anything else (an array) can't be one,
        # and comparing an array or tracer with `in` isn't safe anyway
        return _is_json_scalar(value) and value in self.values


@dataclass(frozen=True)
class UnionSchema(SchemaNode):
    """``anyOf``, ``oneOf`` or a list of types: a value matching any of ``members``."""

    members: tuple[SchemaNode, ...]

    @classmethod
    def _matches(cls, schema: Mapping) -> bool:
        return (
            "anyOf" in schema
            or "oneOf" in schema
            or isinstance(schema.get("type"), list)
            or schema.get("nullable") is True  # OpenAPI 3.0
        )

    @classmethod
    def _from_schema(cls, schema: Mapping, parser: _SchemaParser) -> UnionSchema:
        base = {k: v for k, v in schema.items() if k != "nullable"}
        alternatives = [*base.get("anyOf", ()), *base.get("oneOf", ())]
        if not alternatives and isinstance(base.get("type"), list):
            alternatives = [{**base, "type": t} for t in base["type"]]
        if not alternatives:  # only nullable
            alternatives = [base]
        if schema.get("nullable") is True:
            alternatives.append({"type": "null"})
        members = []
        for node in map(parser.parse, alternatives):
            # Flatten nested unions, such as an optional discriminated union
            members.extend(node.members if isinstance(node, UnionSchema) else [node])
        return cls(tuple(members), schema=schema)

    def _children(self) -> dict[Any, SchemaNode]:
        return dict(enumerate(self.members))


@dataclass(frozen=True)
class ObjectSchema(SchemaNode):
    """A model (``properties``), a dict (``additionalProperties``) or both.

    ``additional`` is the schema of keys not in ``properties``, or ``None`` if
    no other keys are allowed.
    """

    properties: Mapping[str, SchemaNode]
    required: frozenset[str]
    additional: SchemaNode | None

    @classmethod
    def _matches(cls, schema: Mapping) -> bool:
        return (
            schema.get("type") == "object"
            or "properties" in schema
            or "additionalProperties" in schema
        )

    @classmethod
    def _from_schema(cls, schema: Mapping, parser: _SchemaParser) -> ObjectSchema:
        additional = schema.get("additionalProperties", True)
        return cls(
            properties={
                k: parser.parse(v) for k, v in schema.get("properties", {}).items()
            },
            required=frozenset(schema.get("required", ())),
            additional=None if additional is False else parser.parse(additional),
            schema=schema,
        )

    def _accepts(self, value: Any) -> bool:
        if not isinstance(value, Mapping) or not self.required <= value.keys():
            return False
        if self.additional is None and not value.keys() <= self.properties.keys():
            return False
        # A tag (Literal field) the value sets must be one this model allows
        return all(
            prop._accepts(value[key])
            for key, prop in self.properties.items()
            if isinstance(prop, LiteralSchema) and key in value
        )

    def _children(self) -> dict[Any, SchemaNode]:
        extra = {} if self.additional is None else {...: self.additional}
        return {**self.properties, **extra}


@dataclass(frozen=True)
class SequenceSchema(SchemaNode):
    """A list (``items``), or a tuple or NamedTuple (``prefixItems``).

    ``prefix`` holds the schemas of the leading positional items, and
    ``items`` that of any item after them, or ``None`` if there can't be any.
    """

    prefix: tuple[SchemaNode, ...]
    items: SchemaNode | None

    @classmethod
    def _matches(cls, schema: Mapping) -> bool:
        return (
            schema.get("type") == "array"
            or "items" in schema
            or "prefixItems" in schema
        )

    @classmethod
    def _from_schema(cls, schema: Mapping, parser: _SchemaParser) -> SequenceSchema:
        prefix = tuple(parser.parse(m) for m in schema.get("prefixItems", ()))
        items = schema.get("items", True)
        max_items = schema.get("maxItems")
        # pydantic closes a tuple with maxItems rather than items: false
        closed = items is False or (max_items is not None and max_items <= len(prefix))
        return cls(
            prefix=prefix,
            items=None if closed else parser.parse(items),
            schema=schema,
        )

    def _accepts(self, value: Any) -> bool:
        if not _is_sequence(value) or len(value) < self.schema.get("minItems", 0):
            return False
        return self.items is not None or len(value) <= len(self.prefix)

    def _children(self) -> dict[Any, SchemaNode]:
        extra = {} if self.items is None else {...: self.items}
        return {**dict(enumerate(self.prefix)), **extra}


@dataclass(frozen=True)
class ScalarSchema(SchemaNode):
    """A JSON scalar: string, number, integer, boolean or null."""

    type: ScalarType

    @classmethod
    def _matches(cls, schema: Mapping) -> bool:
        return schema.get("type") in ("string", "number", "integer", "boolean", "null")

    @classmethod
    def _from_schema(cls, schema: Mapping, parser: _SchemaParser) -> ScalarSchema:
        return cls(schema["type"], schema=schema)

    def _accepts(self, value: Any) -> bool:
        if not _is_json_scalar(value):
            return False
        if isinstance(value, bool):  # bool is an int subclass, but not a JSON number
            return self.type == "boolean"
        return isinstance(value, _SCALAR_PYTHON_TYPES[self.type])


@dataclass(frozen=True)
class AnySchema(SchemaNode):
    """A schema without constraints, such as ``Any`` or a bare ``{}``."""

    @classmethod
    def _matches(cls, schema: Mapping) -> bool:
        """Anything: the fallback when no other node type matches."""
        return True

    @classmethod
    def _from_schema(cls, schema: Mapping, parser: _SchemaParser) -> AnySchema:
        return cls(schema=schema)

    def _accepts(self, value: Any) -> bool:
        return True


@dataclass(frozen=True)
class UnsupportedSchema(AnySchema):
    """A schema the parser can't handle, such as a recursive definition.

    It is treated like :class:`AnySchema`, so arrays under it aren't found.
    ``reason`` says what wasn't supported. Find these nodes with
    :func:`iter_nodes` to decide whether to raise, warn or carry on.
    """

    reason: str


# In priority order: an array's schema is also an object's, a Literal's has a
# scalar type too, and AnySchema takes whatever is left.
_NODE_TYPES: tuple[type[SchemaNode], ...] = (
    ArraySchema,
    LiteralSchema,
    UnionSchema,
    ObjectSchema,
    SequenceSchema,
    ScalarSchema,
    AnySchema,
)


def parse_schema(
    schema: Mapping, definitions: Mapping[str, Mapping] | None = None
) -> SchemaNode:
    """Parse a JSON Schema as emitted by pydantic into a :class:`SchemaNode` tree.

    Args:
        schema: The schema to parse, for example
            ``openapi["components"]["schemas"]["Apply_InputSchema"]``.
        definitions: The schemas that ``$ref`` entries point to, by name, for
            example ``openapi["components"]["schemas"]`` or a JSON Schema's
            ``$defs``. A ``$ref`` is resolved by its last path segment, so
            both ``#/components/schemas/Name`` and ``#/$defs/Name`` work.
            Defaults to ``schema["$defs"]``.
    """
    if definitions is None:
        definitions = schema.get("$defs", {})
    return _SchemaParser(definitions).parse(schema)


@dataclass(frozen=True)
class _SchemaParser:
    """Parses schemas against shared ``definitions``.

    ``visiting`` holds the names of the ``$ref`` definitions being expanded
    on the way to the current schema, to catch a definition that contains
    itself.
    """

    definitions: Mapping[str, Mapping]
    visiting: frozenset[str] = frozenset()

    def parse(self, schema: Any) -> SchemaNode:
        """Parse ``schema`` into the first node type in ``_NODE_TYPES`` that matches it."""
        # `True` allows anything. `False` (nothing allowed) is handled by the
        # callers, where pydantic emits it: additionalProperties and items.
        if not isinstance(schema, Mapping):
            return AnySchema()

        if "$ref" in schema:
            name = schema["$ref"].rsplit("/", 1)[-1]
            if name in self.visiting:
                return _unsupported(f"recursive definition {name!r}", schema)
            inner = replace(self, visiting=self.visiting | {name})
            return inner.parse(self.definitions[name])

        all_of = schema.get("allOf", ())
        if len(all_of) > 1:
            return _unsupported(f"'allOf' with {len(all_of)} members", schema)
        if all_of:
            return self.parse(all_of[0])

        node_type = next(t for t in _NODE_TYPES if t._matches(schema))
        return node_type._from_schema(schema, self)


def _unsupported(construct: str, schema: Mapping) -> UnsupportedSchema:
    return UnsupportedSchema(f"cannot parse the {construct}", schema=schema)


def schema_map(
    func: Callable[[SchemaNode | None, Any], Any],
    schema: SchemaNode | None,
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    """Map ``func`` over the leaves of ``tree``, paired with their place in ``schema``.

    Like ``jax.tree.map`` with ``schema`` as the first tree, except that
    ``schema`` decides the structure: ``tree`` is walked alongside it, and each
    leaf is replaced by ``func(node, leaf)``. In a union, the value goes to
    the one member that accepts it.

    Args:
        func: Called as ``func(node, leaf)``, where ``node`` is the schema of
            the leaf's position: for example an :class:`ArraySchema` for an
            array field. It is ``None`` where the schema doesn't say, such as a
            key it doesn't declare or a union with no single member that
            accepts the value.
        schema: The parsed schema of ``tree``, from :func:`parse_schema`.
        tree: A payload such as the inputs to :meth:`Tesseract.apply`: nested
            dicts, lists and tuples (NamedTuples included), with arrays and
            scalars as leaves. A leaf is any value the schema doesn't describe
            as a dict, list or tuple.
        is_leaf: If given, called on each value before it is walked into; a
            value for which it returns ``True`` is passed to ``func`` whole.

    Returns:
        ``tree`` with each leaf replaced, rebuilt with the same container types
        (dicts become plain dicts).
    """
    return schema_map_with_path(partial(_drop_path, func), schema, tree, is_leaf)


def schema_map_with_path(
    func: Callable[[str, SchemaNode | None, Any], Any],
    schema: SchemaNode | None,
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    """Like :func:`schema_map`, but calls ``func(path, node, leaf)``.

    ``path`` locates the leaf in Tesseract's path format, the one
    ``vjp_inputs`` and ``differentiable_arrays`` use: model fields as they
    are, dict keys in braces and indices in brackets, as in
    ``"table.{k}.[0].x"``. The root of ``tree`` is ``""``.
    """
    return _schema_map(func, schema, tree, is_leaf, "")


def _drop_path(func: Callable, path: str, node: SchemaNode | None, leaf: Any) -> Any:
    return func(node, leaf)


def _schema_map(
    func: Callable[[str, SchemaNode | None, Any], Any],
    schema: SchemaNode | None,
    tree: Any,
    is_leaf: Callable[[Any], bool] | None,
    path: str,
) -> Any:
    if isinstance(schema, UnionSchema):
        schema = _pick_union_member(tree, schema)
    if is_leaf is not None and is_leaf(tree):
        return func(path, schema, tree)
    if isinstance(schema, ObjectSchema) and isinstance(tree, Mapping):
        return {
            key: _schema_map(
                func,
                schema.properties.get(key, schema.additional),
                item,
                is_leaf,
                _join(path, _key_segment(schema, key)),
            )
            for key, item in tree.items()
        }
    if isinstance(schema, SequenceSchema) and _is_sequence(tree):
        out = [
            _schema_map(
                func,
                schema.prefix[i] if i < len(schema.prefix) else schema.items,
                item,
                is_leaf,
                _join(path, f"[{i}]"),
            )
            for i, item in enumerate(tree)
        ]
        if hasattr(tree, "_fields"):  # NamedTuple
            return type(tree)(*out)
        return type(tree)(out)
    return func(path, schema, tree)


def _key_segment(schema: ObjectSchema, key: Any) -> str:
    """A model field as it is; a dict key (``additionalProperties``) in braces."""
    if key in schema.properties or schema.additional is None:
        return str(key)
    return "{" + _escape_dict_key(str(key)) + "}"


def _escape_dict_key(key: str) -> str:
    # Same as tesseract_core.runtime.tree_transforms.escape_dict_key. Not shared:
    # the runtime is an optional extra of the SDK, and the SDK isn't installed in
    # a built Tesseract, so neither can import the other.
    return "".join("\\" + c if c in "{}\\" else c for c in key)


def _join(path: str, segment: str) -> str:
    return f"{path}.{segment}" if path else segment


def iter_nodes(schema: SchemaNode) -> Iterator[SchemaNode]:
    """Yield ``schema`` and every node below it, parents before children."""
    yield schema
    for child in schema._children().values():
        yield from iter_nodes(child)


def _fold_schema(func: Callable[[SchemaNode, dict], Any], schema: SchemaNode) -> Any:
    """Call ``func(node, results)`` on every node, children first.

    ``results`` maps each of ``node._children()``'s keys to the value
    ``func`` returned for that child.
    """
    results = {k: _fold_schema(func, c) for k, c in schema._children().items()}
    return func(schema, results)


def _example(schema: SchemaNode, seed: int = 42) -> Any:
    """An example payload that validates against ``schema``, for tests.

    Arrays get random data in [0, 100) of their declared dtype (float64 if
    open), base64-encoded, with every dimension of size 1 so it broadcasts to
    any fixed size.
    """
    return _fold_schema(partial(_example_of, np.random.RandomState(seed)), schema)


def _example_of(rng: np.random.RandomState, node: SchemaNode, results: dict) -> Any:
    raw = node.schema
    if isinstance(node, ArraySchema):
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
    if isinstance(node, LiteralSchema):
        return node.values[0]
    if raw.get("examples"):
        return raw["examples"][0]
    if isinstance(node, UnionSchema):
        not_null = (
            results[i]
            for i, m in enumerate(node.members)
            if not (isinstance(m, ScalarSchema) and m.type == "null")
        )
        return next(not_null, None)
    if isinstance(node, ObjectSchema):
        return {k: results[k] for k in node.properties}
    if isinstance(node, SequenceSchema):
        if node.prefix:
            return [results[i] for i in range(len(node.prefix))]
        return [results[...]] * raw.get("minItems", 1) if ... in results else []
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


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _pick_union_member(value: Any, schema: UnionSchema) -> SchemaNode | None:
    candidates = [m for m in schema.members if m._accepts(value)]
    if len(candidates) > 1:
        # Any accepts everything, so another member that accepts is more specific
        candidates = [m for m in candidates if not isinstance(m, AnySchema)]
    return candidates[0] if len(candidates) == 1 else None


# The Python types each JSON scalar type accepts
_SCALAR_PYTHON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _is_json_scalar(value: Any) -> bool:
    # Numpy scalars subclass Python ones but are left out on purpose: the SDK
    # sends anything with ``__array__`` as an array, not a JSON scalar.
    return isinstance(value, (str, int, float, type(None))) and not hasattr(
        value, "dtype"
    )
