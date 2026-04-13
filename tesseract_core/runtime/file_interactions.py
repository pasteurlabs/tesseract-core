# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import urllib.parse
from pathlib import Path
from typing import Any, Literal, get_args

import fsspec
import orjson
from pydantic import TypeAdapter

PathLike = str | Path

supported_format_type = Literal["json", "json+base64", "json+binref"]
SUPPORTED_FORMATS = get_args(supported_format_type)


def output_to_bytes(
    obj: Any,
    format: supported_format_type,
    base_dir: str | Path | None = None,
    binref_dir: str | Path | None = None,
) -> bytes:
    """Encode endpoint output to bytes in the given format.

    obj may contain pydantic.BaseModel / RootModel instances, or regular Python objects.
    """
    ObjSchema = TypeAdapter(type(obj))
    if format == "json":
        context = {"array_encoding": "json"}
    elif format == "json+base64":
        context = {"array_encoding": "base64"}
    elif format == "json+binref":
        context = {
            "array_encoding": "binref",
            "base_dir": base_dir,
            "binref_dir": binref_dir,
        }
    else:
        raise ValueError(
            f"Unsupported format {format} (must be one of {SUPPORTED_FORMATS})"
        )

    # Two-phase serialization to bypass serde_json's slow UTF-8 scanning
    # on large base64 strings (https://github.com/pydantic/pydantic/issues/12911).
    # Phase 1: run Pydantic serializers (encode_array, etc.) -> plain Python dict.
    # Phase 2: orjson serializes the dict to JSON bytes (~4x faster than serde_json).
    python_dict = ObjSchema.dump_python(
        obj, mode="json", context=context, exclude_unset=True
    )
    return orjson.dumps(python_dict)


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


def resolve_input_path(path: Path) -> Path:
    """Resolve a relative path against RuntimeConfig().input_path.

    Validates that the resolved path stays within the input directory and exists.

    Raises:
        ValueError: If the path escapes the input directory.
        FileNotFoundError: If the resolved path does not exist.
    """
    from tesseract_core.runtime.config import get_config

    input_path = get_config().input_path
    tess_path = (input_path / path).resolve()
    if str(input_path) not in str(tess_path):
        raise ValueError(
            f"Invalid input file reference: {path}. "
            f"Expected path to be relative to {input_path}, but got {tess_path}. "
            "File references have to be relative to --input-path."
        )
    if not tess_path.exists():
        raise FileNotFoundError(f"Input path {tess_path} does not exist.")
    return tess_path


def strip_output_path(path: Path) -> Path:
    """Strip RuntimeConfig().output_path prefix from a path.

    If the path is relative to the output directory, returns the relative portion.

    Raises:
        ValueError: If the path does not exist or is outside the output directory.
    """
    from tesseract_core.runtime.config import get_config

    output_path = get_config().output_path
    if path.is_relative_to(output_path):
        if not path.exists():
            raise ValueError(f"Output path {path} does not exist inside Tesseract")
        return path.relative_to(output_path)
    else:
        full_path = output_path / path
        if not full_path.exists():
            if path.exists():
                raise ValueError(
                    f"Output path {path} is not in {output_path}. "
                    f"All output data must be copied to `--output-path` ({output_path})."
                )
            else:
                raise ValueError(
                    f"Output path {path} is not in {output_path} or Tesseract root"
                )
        return path
