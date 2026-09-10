# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sidecar references: serialize a nested model to its own JSON file.

``Ref[T]`` wraps an arbitrary type ``T`` (typically a Pydantic model holding
arrays) so that, when the enclosing Tesseract output is dumped to a directory,
each ``T`` is written to its own ``.json`` file and the main payload carries
only the relative path to it -- the same idea ``json+binref`` applies to array
buffers, one level up the tree.

``Ref`` is deliberately a thin ``Annotated`` wrapper. That keeps it transparent
to :func:`~tesseract_core.runtime.schema_generation.apply_function_to_model_tree`,
so differentiable-path discovery, ``abstract_eval`` schema derivation and the
gradient endpoints keep seeing the wrapped model's arrays and need no
duplicated shape / dtype declarations.
"""

import re
from collections.abc import Mapping
from typing import Annotated, Any, TypeVar
from uuid import uuid4

import orjson
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, TypeAdapter
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from tesseract_core.runtime.file_interactions import (
    join_paths,
    read_from_path,
    write_to_path,
)

T = TypeVar("T")

#: Discriminator marking an encoded ref, mirroring ``object_type: "array"`` on
#: encoded arrays. Clients that decode responses without access to the
#: Tesseract's schema (e.g. the SDK's HTTPClient) rely on it to tell a ref
#: apart from an ordinary string or object field.
REF_OBJECT_TYPE = "ref"

# Filenames derived from user data are restricted to a conservative charset so
# a stem can never escape the output directory or collide with a .bin file.
_SAFE_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _as_ref_path(value: Any) -> str | None:
    """Return the referenced path if ``value`` encodes a ref, else None.

    Accepts the encoded form ``{"object_type": "ref", "path": ...}`` as well as
    a bare path string, so payloads hand-written against the output directory
    keep working.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and value.get("object_type") == REF_OBJECT_TYPE:
        path = value.get("path")
        if not isinstance(path, str):
            raise ValueError(f"Ref marker must carry a string 'path', got {path!r}.")
        return path
    return None


def _ref_filename(value: Any, context: dict) -> str:
    """Pick the sidecar filename for ``value``, rejecting unsafe or duplicate stems."""
    namer = getattr(value, "__ref_name__", None)
    if namer is None:
        return f"{uuid4()}.json"

    stem = namer()
    if not isinstance(stem, str) or not _SAFE_STEM.match(stem):
        raise ValueError(
            f"{type(value).__name__}.__ref_name__() must return a string matching "
            f"{_SAFE_STEM.pattern} to be used as a Ref filename, got {stem!r}."
        )

    filename = f"{stem}.json"
    # Track names for the duration of one dump, so two refs cannot silently
    # clobber each other's file. Mirrors how '__binref_uuid' is threaded
    # through the serialization context by array_encoding.encode_array.
    written = context.setdefault("__ref_names", set())
    if filename in written:
        raise ValueError(
            f"Duplicate Ref filename {filename!r}: __ref_name__() must be unique "
            "across all refs in a single output payload."
        )
    written.add(filename)
    return filename


class PydanticRefAnnotation:
    """Pydantic annotation implementing the sidecar-file encoding for ``Ref``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"{self.__class__.__name__} cannot be instantiated")

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # Build a standalone, fully resolved schema for the wrapped type. Going
        # through TypeAdapter (rather than the handler) is what LazySequence
        # does too: the handler may hand back unresolved definition references,
        # which cannot be turned into a validator on their own.
        adapter = TypeAdapter(source_type)

        def load(value: Any, info: Any) -> Any:
            context = info.context or {}
            relpath = _as_ref_path(value)
            if relpath is None:
                # Inline object -- validate it directly.
                return adapter.validator.validate_python(value, context=context)

            base_dir = context.get("base_dir")
            if base_dir is None:
                raise ValueError(
                    f"Ref {relpath!r} is a relative path but no base_dir is set. "
                    "Invoke the Tesseract with an input / output path set."
                )
            payload = orjson.loads(read_from_path(join_paths(base_dir, relpath)))
            # Keep base_dir unchanged so binrefs *inside* the sidecar, which are
            # written relative to the same base_dir, still resolve.
            return adapter.validator.validate_python(payload, context=context)

        def dump(value: Any, info: Any) -> Any:
            # Serialize the wrapped model with the caller's settings, so nested
            # arrays honour array_encoding / compression / exclude_unset.
            payload = adapter.serializer.to_python(value, **info.__dict__)

            context = info.context or {}
            base_dir = context.get("base_dir")
            if not info.mode_is_json() or base_dir is None:
                # Python mode, or no directory to write to (plain json /
                # json+base64 responses) -- keep the model inline.
                return payload

            # Sidecars live next to the .bin files so that relative binrefs
            # inside them resolve against the same base_dir.
            subdir = context.get("binref_dir")
            filename = _ref_filename(value, context)
            relpath = join_paths(subdir, filename) if subdir else filename
            write_to_path(orjson.dumps(payload), join_paths(base_dir, relpath))
            return {"object_type": REF_OBJECT_TYPE, "path": relpath}

        schema = core_schema.with_info_plain_validator_function(
            load,
            serialization=core_schema.plain_serializer_function_ser_schema(
                dump, info_arg=True
            ),
        )
        # Stash the inner schema so __get_pydantic_json_schema__ can describe
        # the inline alternative without rebuilding it.
        schema["metadata"] = {"ref_inner_schema": adapter.core_schema}
        return schema

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        inline = handler(_core_schema["metadata"]["ref_inner_schema"])
        return {
            "oneOf": [
                {
                    "type": "object",
                    "description": (
                        "Reference to a JSON file holding this object, relative to the "
                        "Tesseract's output path."
                    ),
                    "properties": {
                        "object_type": {"const": REF_OBJECT_TYPE},
                        "path": {"type": "string", "format": "path"},
                    },
                    "required": ["object_type", "path"],
                    "additionalProperties": False,
                },
                inline,
            ]
        }


class Ref:
    """Type annotation for an object serialized to its own JSON file.

    When the enclosing payload is dumped with a ``base_dir`` in the
    serialization context (which is what the ``json+binref`` output format
    sets, see :func:`tesseract_core.runtime.file_interactions.output_to_bytes`),
    each ``Ref`` writes its value to ``<base_dir>/<name>.json`` and serializes
    as the relative path to that file. Without a ``base_dir`` -- plain ``json``
    and ``json+base64`` responses, or Python-mode dumps -- the object is
    serialized inline, so a Tesseract using ``Ref`` still works over plain HTTP.

    Validation accepts both forms: a path string is read and validated against
    the wrapped type, an inline object is validated directly. Either way the
    wrapped type's own validators run, so array shapes and dtypes inside the
    sidecar are checked exactly as they would be inline.

    Sidecars are named with a UUID by default, matching how ``json+binref``
    names its ``.bin`` files. To get readable filenames, give the wrapped model
    a ``__ref_name__()`` method returning the filename stem; it must be unique
    across the payload.

    Example:
        >>> class Frame(BaseModel):
        ...     name: str
        ...     u: Differentiable[Array[(None, 3), Float32]]
        ...
        ...     def __ref_name__(self) -> str:
        ...         return self.name

        >>> class OutputSchema(BaseModel):
        ...     result: list[Ref[Frame]]

        Dumped with ``json+binref``, this produces
        ``{"result": ["frame_0.json", "frame_1.json"]}`` plus one JSON file per
        frame, each holding binref pointers into the shared ``.bin`` buffer.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        clsname = self.__class__.__name__
        raise RuntimeError(
            f"{clsname} cannot be instantiated directly, "
            f"perhaps you meant to use `{clsname}[MyModel]`?"
        )

    def __class_getitem__(cls, key: Any) -> Any:
        """Wrap the given type so it is serialized to a sidecar JSON file."""
        if isinstance(key, tuple):
            raise ValueError(
                "Ref takes a single parameter: Ref[MyModel]. To control sidecar "
                "filenames, give MyModel a __ref_name__() method."
            )
        return Annotated[key, PydanticRefAnnotation]
