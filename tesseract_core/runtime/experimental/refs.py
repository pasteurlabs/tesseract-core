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
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypeVar
from uuid import uuid4

import orjson
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, TypeAdapter
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from tesseract_core.runtime.file_interactions import (
    PathLike,
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


def _posix_join(subdir: PathLike | None, filename: str) -> str:
    """Join a sidecar path for the wire, always with forward slashes.

    The emitted path travels in the response and is resolved by whoever reads
    it, which may not be on the same OS as the Tesseract. ``join_paths`` uses
    :class:`pathlib.Path`, so it would produce backslashes when the runtime
    happens to run natively on Windows.
    """
    if not subdir:
        return filename
    return "/".join((*Path(subdir).parts, filename))


def _ref_prefix(value: Any, prefix: str | None) -> str | None:
    """Pick the filename prefix for ``value``, or None to fall back to a UUID.

    In order of preference:

    1. the prefix passed to ``Ref[T, "..."]``;
    2. the model's own ``name`` field, when it holds a usable string;
    3. the model's class name.

    A ``name`` that is missing, not a string, or not filename-safe falls
    through to the next option rather than raising: names usually come from
    runtime data, and a bad one should not sink a whole response. A prefix
    passed to ``Ref`` is developer-supplied, so an unusable one is an error.
    """
    if prefix is not None:
        if not _SAFE_STEM.match(prefix):
            raise ValueError(
                f"Invalid Ref prefix {prefix!r}: must match {_SAFE_STEM.pattern}."
            )
        return prefix

    name = getattr(value, "name", None)
    if isinstance(name, str) and _SAFE_STEM.match(name):
        return name

    class_name = type(value).__name__
    if _SAFE_STEM.match(class_name):
        return class_name

    return None


def _ref_filename(value: Any, prefix: str | None, context: dict) -> str:
    """Build the sidecar filename for ``value``.

    Every prefixed name carries a running index (``frame_000``, ``frame_001``,
    ...), so sidecars in one payload can never collide no matter how the prefix
    was derived. Counters live in the serialization context, so they restart on
    each dump -- the same way ``__binref_uuid`` is threaded through by
    ``array_encoding.encode_array``. Two dumps into the *same* directory
    therefore overwrite each other; served Tesseracts avoid this by writing
    into a per-request ``run_<id>/`` directory.
    """
    stem = _ref_prefix(value, prefix)
    if stem is None:
        # Nothing usable to name it after; fall back to how binref names .bin files.
        return f"{uuid4()}.json"

    counters = context.setdefault("__ref_counters", {})
    index = counters.get(stem, 0)
    counters[stem] = index + 1
    return f"{stem}_{index:03d}.json"


@dataclass(frozen=True)
class PydanticRefAnnotation:
    """Pydantic annotation implementing the sidecar-file encoding for ``Ref``."""

    prefix: str | None = None

    def __get_pydantic_core_schema__(
        self,
        source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # Build a standalone, fully resolved schema for the wrapped type. Going
        # through TypeAdapter (rather than the handler) is what LazySequence
        # does too: the handler may hand back unresolved definition references,
        # which cannot be turned into a validator on their own.
        adapter = TypeAdapter(source_type)
        prefix = self.prefix

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
            filename = _ref_filename(value, prefix, context)
            relpath = _posix_join(subdir, filename)
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

    def __get_pydantic_json_schema__(
        self, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
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
    as ``{"object_type": "ref", "path": ...}``. Without a ``base_dir`` -- plain
    ``json`` and ``json+base64`` responses, or Python-mode dumps -- the object
    is serialized inline, so the response stays self-contained.

    Validation accepts an encoded ref, a bare path string, or an inline
    object. In every case the wrapped type's own validators run, so array
    shapes and dtypes inside a sidecar are checked exactly as they would be
    inline.

    Sidecar filenames are ``<prefix>_<index>.json``, where the prefix is taken
    from the first of these that is usable:

    1. the prefix passed as ``Ref[T, "frame"]``;
    2. the model's own ``name`` field;
    3. the model's class name.

    Failing all three (a model with no usable class name), files are named with
    a UUID, matching how ``json+binref`` names its ``.bin`` files. The index
    makes collisions impossible within a payload, so ``name`` need not be
    unique.

    Example:
        >>> class Frame(BaseModel):
        ...     u: Differentiable[Array[(None, 3), Float32]]

        >>> class OutputSchema(BaseModel):
        ...     frames: list[Ref[Frame, "frame"]]

        Dumped with ``json+binref``, this writes ``frame_000.json``,
        ``frame_001.json``, ... each holding binref pointers into the shared
        ``.bin`` buffer. Without the prefix the files would be named after the
        model's ``name`` field, or ``Frame_000.json`` and so on.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        clsname = self.__class__.__name__
        raise RuntimeError(
            f"{clsname} cannot be instantiated directly, "
            f"perhaps you meant to use `{clsname}[MyModel]`?"
        )

    def __class_getitem__(cls, key: Any) -> Any:
        """Wrap a type so it is serialized to a sidecar file, with an optional prefix."""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError(
                    "Ref takes at most two parameters: "
                    'Ref[MyModel] or Ref[MyModel, "filename_prefix"]'
                )
            base_type, prefix = key
            if not isinstance(prefix, str):
                raise ValueError(
                    "Second parameter of Ref[...] must be a filename prefix "
                    f"(a string), got {prefix!r}"
                )
        else:
            base_type, prefix = key, None

        return Annotated[base_type, PydanticRefAnnotation(prefix)]
