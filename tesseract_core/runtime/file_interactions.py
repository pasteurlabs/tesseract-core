# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import urllib.parse
from pathlib import Path
from typing import Any, Literal, get_args

import fsspec
import orjson
from pydantic import TypeAdapter

PathLike = str | Path

# An output encoding is two orthogonal choices:
#
# - the *output format* -- how host (CPU) arrays are serialized into the JSON
#   response (``json`` inline / ``json+base64`` / ``json+binref``);
# - the *GPU transport* -- how device (GPU) arrays leave the process (``none`` =
#   copied to host and serialized like any CPU array, or a device transport such
#   as ``cuda_ipc`` that exports them by reference without a host round-trip).
#
# They compose freely: a response can inline its CPU arrays as JSON while
# handing its GPU arrays out as ``cuda_ipc`` handles. Clients choose both per
# request via the ``Accept`` header, while the runtime config sets the default
# output format and which GPU transports are offered.
supported_format_type = Literal["json", "json+base64", "json+binref"]

# GPU transports. ``none`` is the always-available default (GPU output is copied
# to the host and encoded via the output format). Any other value passes device
# memory by reference and is an experimental capability that must be enabled on
# the server and requested by the client.
#
# The disabled state is the explicit string ``"none"`` rather than ``None``, so
# it is clear to users that this means "disabled", not "unspecified".
gpu_transport_type = Literal["none", "cuda_ipc"]

# Compressions for array buffers. Disabled compression is ``None`` in the config
# and in array payloads, and ``"none"`` in an Accept header.
compression_type = Literal["lz4"]

# Every output format is always available (none of them are experimental).
SUPPORTED_FORMATS = get_args(supported_format_type)


def available_formats() -> tuple[str, ...]:
    """Output (host-array) formats the runtime accepts.

    These describe how *CPU* arrays are serialized and are always available. How
    *GPU* arrays leave the process is a separate axis; see
    :func:`available_gpu_transports`.
    """
    return SUPPORTED_FORMATS


def available_gpu_transports() -> tuple[str, ...]:
    """GPU transports the runtime accepts for inputs and outputs.

    Always includes ``none``, plus the configured ``gpu_transport`` if it is
    anything else.
    """
    from tesseract_core.runtime.config import get_config

    configured = get_config().gpu_transport
    if configured != "none":
        return ("none", configured)
    return ("none",)


def available_compressions() -> tuple[str, ...]:
    """Output compressions the runtime accepts (``none`` disables compression)."""
    return ("none", *get_args(compression_type))


def output_to_bytes(
    obj: Any,
    format: supported_format_type,
    base_dir: str | Path | None = None,
    binref_dir: str | Path | None = None,
    compression: compression_type | None = None,
    gpu_transport: gpu_transport_type = "none",
) -> bytes:
    """Encode endpoint output to bytes.

    ``format`` chooses how host (CPU) arrays are serialized; ``gpu_transport``
    chooses how device (GPU) arrays leave the process (``none`` copies them to
    the host and serializes them via ``format``; ``cuda_ipc`` exports them by
    reference). The two are independent -- a response may serialize its CPU
    arrays one way and hand out its GPU arrays another -- and ``encode_array``
    routes each array by where it lives.

    obj may contain pydantic.BaseModel / RootModel instances, or regular Python objects.
    """
    allowed = available_formats()
    if format not in allowed:
        raise ValueError(f"Unsupported format {format} (must be one of {allowed})")

    allowed_transports = available_gpu_transports()
    if gpu_transport not in allowed_transports:
        raise ValueError(
            f"Unsupported GPU transport {gpu_transport} "
            f"(must be one of {allowed_transports})"
        )

    if compression not in (None, "lz4"):
        raise ValueError(f"Unsupported compression {compression}")

    ObjSchema = TypeAdapter(type(obj))
    # The host-array encoding (``array_encoding``) and the device transport
    # (``device_transport``) are independent context keys, read per-leaf by
    # encode_array. ``array_encoding`` names only the CPU encoding; a GPU array
    # goes over ``device_transport`` when set, else it is copied to host and
    # encoded like a CPU array.
    if format == "json":
        context: dict[str, Any] = {"array_encoding": "json"}
    elif format == "json+base64":
        context = {"array_encoding": "base64", "compression": compression}
    elif format == "json+binref":
        context = {
            "array_encoding": "binref",
            "base_dir": base_dir,
            "binref_dir": binref_dir,
            "compression": compression,
        }
    else:
        raise ValueError(f"Unsupported format {format} (must be one of {allowed})")

    context["device_transport"] = None if gpu_transport == "none" else gpu_transport

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
    # Coerce to str to avoid "Cannot mix str and non-str arguments"
    # when mixing PurePosixPath / PureWindowsPath on Windows
    base = str(base)
    other = str(other)
    if is_absolute_path(other):
        return other
    if is_url(base):
        return urllib.parse.urljoin(base, other)
    return str(Path(base).joinpath(other))


def is_absolute_path(path: PathLike) -> bool:
    """Check if path is an absolute path or a url."""
    return is_url(path) or Path(path).is_absolute()


def is_url(path: PathLike) -> bool:
    """Check if path is a url."""
    scheme = urllib.parse.urlparse(str(path)).scheme
    # Single-letter schemes are Windows drive letters (e.g., C:), not URLs
    return bool(scheme) and len(scheme) > 1


def parent_path(x: PathLike) -> PathLike:
    """Get parent of given path (which may be a URL)."""
    if is_url(x):
        return urllib.parse.urljoin(x, "..")

    return type(x)(str(Path(x).parent))
