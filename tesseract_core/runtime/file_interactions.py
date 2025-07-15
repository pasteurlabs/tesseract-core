# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import urllib.parse
from pathlib import Path
from typing import Any, Literal, Optional, Union, get_args

import fsspec
from pydantic import TypeAdapter

PathLike = Union[str, Path]

supported_format_type = Literal["json", "msgpack", "json+base64", "json+binref"]
SUPPORTED_FORMATS = get_args(supported_format_type)


def guess_format_from_path(path: PathLike) -> supported_format_type:
    """Guess the format from the given path.

    The format is determined by the file extension.
    """
    if path.endswith(".json"):
        return "json"
    elif path.endswith(".msgpack"):
        return "msgpack"

    raise ValueError(
        f"Could not guess format from path {path} (supported formats: {SUPPORTED_FORMATS})"
    )


def output_to_bytes(
    obj: Any, format: supported_format_type, base_dir: Optional[Union[str, Path]] = None
) -> bytes:
    """Encode endpoint output to bytes in the given format.

    obj may contain pydantic.BaseModel / RootModel instances, or regular Python objects.
    """
    ObjSchema = TypeAdapter(type(obj))
    if format == "json":
        return ObjSchema.dump_json(
            obj, context={"array_encoding": "json"}, exclude_unset=True
        )
    elif format == "json+base64":
        return ObjSchema.dump_json(
            obj, context={"array_encoding": "base64"}, exclude_unset=True
        )
    elif format == "json+binref":
        return ObjSchema.dump_json(
            obj,
            context={"array_encoding": "binref", "base_dir": base_dir},
            exclude_unset=True,
        )
    elif format == "msgpack":
        import msgpack

        import tesseract_core.runtime.vendor.msgpack_numpy as msgpack_numpy

        obj_clean = ObjSchema.dump_python(obj, exclude_unset=True)
        return msgpack.packb(obj_clean, default=msgpack_numpy.encode)

    raise ValueError(
        f"Unsupported format {format} (must be one of {SUPPORTED_FORMATS})"
    )


def load_bytes(
    buffer: bytes,
    format: supported_format_type,
) -> Any:
    """Decode the given buffer to a Python object.

    The buffer is expected to be in the given format.
    """
    if format.startswith("json"):
        return json.loads(buffer.decode())
    elif format == "msgpack":
        import msgpack

        import tesseract_core.runtime.vendor.msgpack_numpy as msgpack_numpy

        return msgpack.unpackb(buffer, object_hook=msgpack_numpy.decode)

    raise ValueError(
        f"Unsupported format {format} (must be one of {SUPPORTED_FORMATS})"
    )


def read_from_path(path: PathLike, offset: int = 0, length: int = -1) -> bytes:
    """Read the contents of the given path as bytes.

    Path may be anything supported by fsspec.
    """
    with fsspec.open(path, "rb") as f:
        if offset != 0:
            f.seek(offset)
        return f.read(length)


def write_to_path(buffer: bytes, path: PathLike, append: bool = False) -> None:
    """Write the buffer to the given path.

    Path may be anything supported by fsspec.
    """
    mode = "ab" if append else "wb"
    with fsspec.open(path, mode, auto_mkdir=True) as f:
        f.write(buffer)


def expand_glob(pattern: str) -> list[str]:
    """Expand the given glob pattern.

    Path may be anything supported by fsspec.
    """
    open_files = fsspec.open_files(pattern, "rb", expand=True)
    return sorted(f.path for f in open_files)


def get_filesize(path: PathLike) -> int:
    """Get the size of the given path in bytes.

    Path may be anything supported by fsspec.
    """
    try:
        with fsspec.open(path, "rb") as f:
            f.seek(0, 2)
            return f.tell()
    except FileNotFoundError:
        return 0


def join_paths(base: PathLike, other: PathLike) -> str:
    """Join the base path (URL or local path) with the given other path.

    If the other path is an absolute URL, return it as is.
    """
    if is_absolute_path(other):
        return str(other)
    if is_url(base):
        return urllib.parse.urljoin(base, other)
    return Path(base).joinpath(other).as_posix()


def is_absolute_path(path: PathLike) -> bool:
    """Check if path is an absolute path or a url."""
    return is_url(path) or Path(path).is_absolute()


def is_url(path: PathLike) -> bool:
    """Check if path is a url."""
    return bool(urllib.parse.urlparse(str(path)).scheme)


def parent_path(x: PathLike) -> PathLike:
    """Get parent of given path (which may be a URL)."""
    if is_url(x):
        return urllib.parse.urljoin(x, "..")

    return type(x)(Path(x).parent.as_posix())
