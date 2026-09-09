# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias, TypedDict, get_args
from uuid import uuid4

import lz4.frame
import numpy as np
import pybase64
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PositiveInt,
    StrictStr,
    ValidationInfo,
    create_model,
)
from pydantic_core import PydanticCustomError

from tesseract_core.runtime.file_interactions import (
    get_filesize,
    is_absolute_path,
    is_url,
    join_paths,
    read_from_path,
    write_to_path,
)

AllowedDtypes = Literal[
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "bool",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "complex64",
    "complex128",
]

GPUArray: TypeAlias = Any  # Placeholder for GPU array types (e.g., CuPy, PyTorch, etc.)
EllipsisType: TypeAlias = type(Ellipsis)
ArrayLike: TypeAlias = np.ndarray | np.number | np.bool_
ShapeType: TypeAlias = tuple[int | None, ...] | EllipsisType


class ArrayDict(TypedDict):
    """TypedDict for the JSON representation of an encoded array."""

    object_type: str
    shape: Sequence[int]
    dtype: str
    data: dict[str, Any]


MAX_BINREF_BUFFER_SIZE = 100 * 1024 * 1024  # 100 MB


def _compress(data: bytes, compression: str | None) -> bytes:
    if compression is None:
        return data
    if compression == "lz4":
        return lz4.frame.compress(data)
    raise ValueError(f"Unknown compression: {compression}")


def _decompress(data: bytes, compression: str | None) -> bytes:
    if compression is None:
        return data
    if compression == "lz4":
        return lz4.frame.decompress(data)
    raise ValueError(f"Unknown compression: {compression}")


# Base classes for the different array encodings
# The actual models are created dynamically based on the expected shape and dtype by get_array_model


class Base64ArrayData(BaseModel):
    """Data structure for base64 encoded binary buffers."""

    buffer: Annotated[
        StrictStr,
        Field(
            description="Base64 encoded binary buffer",
            examples=["<base64 encoded string>"],
        ),
    ]
    encoding: Literal["base64"]
    compression: Literal["lz4"] | None = None
    model_config = ConfigDict(extra="forbid")


class BinrefArrayData(BaseModel):
    """Data structure that dumps array data to binary file.

    The buffer field format is ``<path>[:<offset>[:<compressed_size>]]``.
    When compression is set, the buffer must include ``:<compressed_size>``
    so readers know how many compressed bytes to read.
    """

    buffer: StrictStr = Field(pattern=r"^.+?(\:\d+(\:\d+)?)?$")
    encoding: Literal["binref"]
    compression: Literal["lz4"] | None = None

    model_config = ConfigDict(extra="forbid")


class JsonArrayData(BaseModel):
    """Data structure for json buffers (list of decimal numbers)."""

    buffer: JsonValue
    encoding: Literal["json"]
    compression: None = None

    model_config = ConfigDict(extra="forbid")


class CudaIpcArrayData(BaseModel):
    """Data structure for CUDA IPC shared GPU memory handles.

    The buffer field packs all four components as
    ``<device>:<handle>:<storage_offset>:<storage_size>``, where:

    - ``device`` is the CUDA device ordinal the memory lives on,
    - ``handle`` is the base64-encoded 64-byte cudaIpcMemHandle_t (its base64
      alphabet never contains ``:``, so it is safe as a field delimiter),
    - ``storage_offset`` is the byte offset within the cudaMalloc allocation,
    - ``storage_size`` is the total size in bytes of the cudaMalloc allocation.

    This is only the JSON *schema* for the encoding; all the CUDA runtime
    machinery that produces and consumes it lives in
    :mod:`tesseract_core.runtime.cuda.ipc`.
    """

    buffer: StrictStr = Field(
        pattern=r"^\d+:[A-Za-z0-9+/=]+:\d+:\d+$",
        description="Packed CUDA IPC descriptor: <device>:<handle>:<storage_offset>:<storage_size>",
    )
    encoding: Literal["cuda_ipc"]
    compression: None = None

    model_config = ConfigDict(extra="forbid")


class EncodedArrayModel(BaseModel):
    """Base class for general encoded arrays.

    Allowed values for shape and dtype are enforced by subclasses.
    """

    object_type: Literal["array"]
    shape: tuple[PositiveInt, ...]
    dtype: AllowedDtypes
    data: BinrefArrayData | Base64ArrayData | JsonArrayData | CudaIpcArrayData
    model_config = ConfigDict(extra="forbid")


def get_array_model(
    expected_shape: ShapeType, expected_dtype: str | None, flags: Sequence[str]
) -> type[EncodedArrayModel]:
    """Create a Pydantic model for an encoded array that does validation on the given expected shape and dtype."""
    if expected_dtype is None:
        dtype_type = AllowedDtypes
    else:
        # Only allow dtypes that can be cast to the expected dtype
        subdtypes = [
            dtype
            for dtype in get_args(AllowedDtypes)
            if np.can_cast(dtype, expected_dtype, casting="same_kind")
        ]
        dtype_type = Literal[tuple(subdtypes)]

    shape_kwargs = {}

    # Only allow shapes that can be broadcasted to the expected shape
    if expected_shape is Ellipsis:
        # No shape check
        shape_type = tuple[int, ...]
    else:
        # There are 3 cases for each dimension `n`:
        # - n=None: polymorphic dimension, can be any positive int
        # - n=1: fixed dimension, must be 1
        # - n=N: fixed dimension, must be N or 1 (triggers broadcasting to N)
        # Example: expected_shape=(None, 1, 3) -> allowed_vals=tuple[PositiveInt, Literal[1], Literal[1, 3]]
        allowed_vals = []
        for dim in expected_shape:
            if dim is None:
                allowed_vals.append(PositiveInt)
            elif dim == 1:
                allowed_vals.append(Literal[1])
            else:
                allowed_vals.append(Literal[1, dim])

        if not allowed_vals:
            # Scalar -> require empty tuple
            shape_type = tuple[int, ...]

        shape_type = tuple[tuple(allowed_vals)]
        shape_kwargs.update(
            examples=([1 if s is None else s for s in expected_shape],),
            # Dimensionality must match exactly
            min_length=len(expected_shape),
            max_length=len(expected_shape),
        )

    # Add flags to the model config
    config = EncodedArrayModel.model_config
    config["json_schema_extra"] = {"array_flags": flags}

    fields = {
        "object_type": (
            Literal["array"],
            Field(
                description="Indicates that this dict can be parsed to an array.",
                default="array",
            ),
        ),
        "shape": (
            shape_type,
            Field(
                description="Shape of the array",
                **shape_kwargs,
            ),
        ),
        "dtype": (
            dtype_type,
            Field(
                description="Data type of the array",
                examples=[expected_dtype or "float64"],
            ),
        ),
        # Choose the appropriate data structure based on the encoding
        "data": (
            BinrefArrayData | Base64ArrayData | JsonArrayData | CudaIpcArrayData,
            Field(discriminator="encoding"),
        ),
        "model_config": (ConfigDict, config),
    }

    if expected_shape is Ellipsis:
        readable_shape = "anyrank"
    elif not expected_shape:
        readable_shape = "scalar"
    else:
        readable_shape = "_".join(
            str(s) if s is not None else "any" for s in expected_shape
        )

    readable_flags = "_".join(flags) if flags else "noflags"

    out = create_model(
        f"EncodedArrayModel__{readable_shape}__{expected_dtype}__{readable_flags}",
        **fields,
        __base__=EncodedArrayModel,
    )
    return out


def _fast_tobytes(arr: ArrayLike) -> bytes:
    """Convert a NumPy array to bytes without copying if possible."""
    return np.ascontiguousarray(arr).data


def _dump_binref_arraydict(
    arr: ArrayLike,
    base_dir: Path | str,
    subdir: Path | str | None,
    current_binref_uuid: str,
    max_file_size: int = MAX_BINREF_BUFFER_SIZE,
    compression: str | None = None,
) -> tuple[ArrayDict, str]:
    """Dump array to json+binref encoded array dict."""
    target_name = f"{current_binref_uuid}.bin"
    if subdir is not None:
        target_name = join_paths(subdir, target_name)
    target_path = join_paths(base_dir, target_name)

    current_size = get_filesize(target_path)

    # if the current buffer is too large, use a new one
    if current_size > max_file_size:
        current_size = 0
        current_binref_uuid = str(uuid4())
        target_name = f"{current_binref_uuid}.bin"
        if subdir is not None:
            target_name = join_paths(subdir, target_name)
        target_path = join_paths(base_dir, target_name)

    blob = _compress(_fast_tobytes(arr), compression)
    write_to_path(blob, target_path, append=True)
    offset = current_size

    if compression is not None:
        data = {
            "buffer": f"{target_name}:{offset}:{len(blob)}",
            "encoding": "binref",
            "compression": compression,
        }
    else:
        data = {"buffer": f"{target_name}:{offset}", "encoding": "binref"}
    arraydict = {
        "object_type": "array",
        "shape": list(arr.shape),
        "dtype": arr.dtype.name,
        "data": data,
    }
    return arraydict, current_binref_uuid


def _dump_base64_arraydict(arr: ArrayLike, compression: str | None = None) -> ArrayDict:
    """Dump array to json+base64 encoded array dict (plain dict, no Pydantic models)."""
    blob = _compress(_fast_tobytes(arr), compression)
    data: dict[str, Any] = {
        "buffer": pybase64.b64encode_as_string(blob),
        "encoding": "base64",
    }
    if compression is not None:
        data["compression"] = compression
    return {
        "object_type": "array",
        "shape": list(arr.shape),
        "dtype": arr.dtype.name,
        "data": data,
    }


def _dump_json_arraydict(arr: ArrayLike) -> ArrayDict:
    """Dump array to json encoded array dict (plain dict, no Pydantic models)."""
    return {
        "object_type": "array",
        "shape": list(arr.shape),
        "dtype": arr.dtype.name,
        "data": {"buffer": arr.tolist(), "encoding": "json"},
    }


def _load_base64_arraydict(val: ArrayDict) -> np.ndarray:
    """Load array from json+base64 encoded array dict."""
    buffer = pybase64.b64decode(val["data"]["buffer"], validate=True)
    buffer = _decompress(buffer, val["data"].get("compression"))
    return np.frombuffer(buffer, dtype=val["dtype"]).reshape(val["shape"])


def _read_binref_array(
    full_path: str | Path,
    offset: int,
    num_bytes: int,
    dtype: np.dtype,
    count: int,
) -> np.ndarray:
    """Read an uncompressed binref buffer into an owned, writable array.

    The array owns its data and does not depend on the file afterwards, so the
    backing file may be safely recycled (e.g. by a client-side input write pool
    that reuses buffers across requests) once this returns. The result is
    writable, matching what the caller-facing schema expects for array inputs.
    """
    out = np.empty(count, dtype=dtype)
    with open(full_path, "rb") as f:
        if offset:
            f.seek(offset)
        f.readinto(memoryview(out).cast("B"))
    return out


def _load_binref_arraydict(val: ArrayDict, base_dir: str | Path | None) -> np.ndarray:
    """Load array from json+binref encoded array dict."""
    path_match = re.match(
        r"^(?P<path>.+?)(\:(?P<offset>\d+)(\:(?P<compressed_size>\d+))?)?$",
        val["data"]["buffer"],
    )
    if not path_match:
        raise ValueError(
            f"Invalid binref path format: {val['data']['buffer']}. "
            "Expected format is '<path>[:<offset>[:<compressed_size>]]'."
        )
    bufferpath = path_match.group("path")
    if path_match.group("offset") is None:
        offset = 0
    else:
        offset = int(path_match.group("offset"))
    compressed_size_str = path_match.group("compressed_size")

    uses_relative_path = not is_absolute_path(bufferpath) and not is_url(bufferpath)
    if uses_relative_path and base_dir is None:
        raise ValueError(
            "Array data is binref encoded with a relative path but no base_dir is provided. "
            "Invoke the Tesseract with an input / output path set, or make sure that paths are absolute."
        )

    dtype = np.dtype(val["dtype"])
    shape = val["shape"]
    size = 1 if len(shape) == 0 else np.prod(shape)
    num_bytes = int(size * dtype.itemsize)

    compression = val["data"].get("compression")

    if base_dir is not None:
        bufferpath = join_paths(base_dir, bufferpath)

    if compression is None:
        # For uncompressed data on a local filesystem, read directly into an
        # owned array via readinto, avoiding an intermediate bytes copy. The
        # result owns its data, so a client-side input write pool may recycle
        # the backing file after the request. Non-local paths (URLs, object
        # stores) and empty arrays fall back to a plain read.
        if num_bytes > 0 and not is_url(bufferpath) and os.path.isfile(bufferpath):
            count = 1 if len(shape) == 0 else int(size)
            return _read_binref_array(
                bufferpath, offset, num_bytes, dtype, count
            ).reshape(shape)
        buffer = read_from_path(bufferpath, offset=offset, length=num_bytes)
    else:
        if compressed_size_str is None:
            raise ValueError(
                "compressed_size is required in buffer spec when compression is set "
                "(expected format: '<path>:<offset>:<compressed_size>')"
            )
        buffer = _decompress(
            read_from_path(bufferpath, offset=offset, length=int(compressed_size_str)),
            compression,
        )
    return np.frombuffer(buffer, dtype=dtype).reshape(shape)


class BinrefArray:
    """An array whose data lives on disk in binref format, referenced by path.

    Hand one to an ordinary :class:`Array` field in place of a NumPy array and
    the field forwards it verbatim when the client negotiates ``json+binref``
    output -- so a buffer written to disk during ``apply`` reaches the client
    without ever being read back into memory. For any other negotiated format
    (``json``, ``base64``) the buffer is loaded once and encoded like a normal
    array. This mirrors how the ``Array`` type already passes GPU arrays through
    validation untouched and materializes them only when needed (see
    :func:`validate_python_or_gpu_array` and :func:`encode_array`).

    Construct one via a named constructor (the plain constructor raises):

    * :meth:`write` -- write a NumPy array to disk and reference it.
    * :meth:`from_file` -- reference a buffer some other code already wrote (e.g.
      a compiled solver), building the buffer spec from a path + offset.
    * :meth:`from_spec` -- for full control, pass an explicit buffer spec string.

    Example (compiled code wrote ``mesh.bin`` itself)::

        def apply(inputs):
            run_solver(out="mesh.bin")  # writes into the output directory
            arr = BinrefArray.from_file("mesh.bin", shape=(1000, 1000), dtype="float64")
            return OutputSchema(result=arr)

    The buffer path is resolved by the client against the served
    ``output_path``, so it must be relative to that directory (a bare filename is
    resolved as ``output_path / filename``) or an absolute path / URL the
    decoder can reach. The data must be C-contiguous, row-major and match the
    declared ``shape`` and ``dtype``; a mismatch is caught when the field is
    validated.

    Deliberately a plain class (not a dataclass / ``BaseModel``) so Pydantic
    treats it as an opaque leaf and does not flatten it during the Python-mode
    ``model_dump()`` the runtime performs before serialization.
    """

    __slots__ = ("_arraydict",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "BinrefArray cannot be instantiated directly; use one of the named "
            "constructors: BinrefArray.write(arr), "
            "BinrefArray.from_file(path, shape, dtype), or "
            "BinrefArray.from_spec(buffer, shape, dtype)."
        )

    @classmethod
    def from_spec(
        cls,
        buffer: str,
        shape: Sequence[int],
        dtype: str,
        *,
        compression: str | None = None,
    ) -> "BinrefArray":
        """Reference a binref buffer by its raw spec string.

        Args:
            buffer: A ``<path>[:<offset>[:<compressed_size>]]`` spec. The path is
                resolved against the served ``output_path`` (see class docs).
            shape: Shape of the array.
            dtype: NumPy dtype name (e.g. ``"float64"``).
            compression: Compression applied to the buffer, if any (``"lz4"``).
                When set, ``buffer`` must include the compressed size.
        """
        if not isinstance(buffer, str) or not buffer:
            raise ValueError("BinrefArray buffer must be a non-empty path spec")
        allowed_dtypes = [d.lower() for d in get_args(AllowedDtypes)]
        if dtype not in allowed_dtypes:
            raise ValueError(
                f"BinrefArray dtype '{dtype}' is not supported; must be one of: "
                f"{', '.join(allowed_dtypes)}"
            )
        data: dict[str, Any] = {"buffer": buffer, "encoding": "binref"}
        if compression is not None:
            data["compression"] = compression
        return cls._from_arraydict(
            {
                "object_type": "array",
                "shape": [int(s) for s in shape],
                "dtype": dtype,
                "data": data,
            }
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        shape: Sequence[int],
        dtype: str,
        *,
        offset: int = 0,
        compression: str | None = None,
        compressed_size: int | None = None,
    ) -> "BinrefArray":
        """Reference a buffer already on disk (e.g. written by compiled code).

        Args:
            path: Buffer path, relative to the served ``output_path`` (or an
                absolute path / URL).
            shape: Shape of the array.
            dtype: NumPy dtype name (e.g. ``"float64"``).
            offset: Byte offset of the array within the file.
            compression: Compression applied to the buffer, if any (``"lz4"``).
            compressed_size: Number of compressed bytes; required when
                ``compression`` is set so the reader knows how much to read.
        """
        spec = str(path)
        if compression is not None:
            if compressed_size is None:
                raise ValueError("compressed_size is required when compression is set")
            spec = f"{spec}:{int(offset)}:{int(compressed_size)}"
        elif offset:
            spec = f"{spec}:{int(offset)}"
        return cls.from_spec(spec, shape, dtype, compression=compression)

    @classmethod
    def write(
        cls,
        arr: ArrayLike,
        *,
        output_dir: str | Path | None = None,
        compression: str | None = None,
    ) -> "BinrefArray":
        """Write ``arr`` to its own binref buffer on disk and reference it.

        The buffer uses the same on-disk layout as the built-in binref encoder,
        so the result is indistinguishable from a normally-encoded array. Each
        call writes an independent file; to pack many arrays into a few shared,
        rotating buffers use :class:`~tesseract_core.runtime.experimental.BinrefWriter`.

        Args:
            arr: The array to write. Coerced to a contiguous NumPy array.
            output_dir: Directory to write into. Defaults to the configured
                ``output_path`` (see class docs on path resolution).
            compression: Optional compression to apply (currently only ``"lz4"``).
        """
        from tesseract_core.runtime.config import get_config

        if output_dir is None:
            output_dir = get_config().output_path
        arr = np.ascontiguousarray(arr)
        arraydict, _ = _dump_binref_arraydict(
            arr,
            base_dir=output_dir,
            subdir=None,
            current_binref_uuid=str(uuid4()),
            max_file_size=MAX_BINREF_BUFFER_SIZE,
            compression=compression,
        )
        return cls._from_arraydict(arraydict)

    @classmethod
    def _from_arraydict(cls, arraydict: "ArrayDict") -> "BinrefArray":
        """Wrap a pre-built binref ``ArrayDict`` (internal / writer use)."""
        obj = cls.__new__(cls)
        obj._arraydict = arraydict
        return obj

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the referenced array."""
        return tuple(self._arraydict["shape"])

    @property
    def dtype(self) -> str:
        """The dtype name of the referenced array."""
        return self._arraydict["dtype"]

    @property
    def buffer(self) -> str:
        """The ``<path>[:<offset>[:<compressed_size>]]`` buffer spec."""
        return self._arraydict["data"]["buffer"]

    def to_arraydict(self) -> "ArrayDict":
        """The binref ``ArrayDict`` this reference serializes to."""
        return self._arraydict

    def load(self, context: dict[str, Any] | None = None) -> np.ndarray:
        """Read the referenced buffer into a NumPy array.

        This is the one operation that actually touches the buffer bytes; the
        rest of the reference stays lazy. Paths are resolved against the
        ``base_dir`` in ``context`` if given, else the configured
        ``output_path``.
        """
        return _load_ref(self, context or {})

    def __array__(self, dtype: Any = None) -> np.ndarray:
        # Lets NumPy (and any array-like consumer) materialize the reference on
        # demand via ``np.asarray(ref)``.
        arr = self.load()
        return arr.astype(dtype) if dtype is not None else arr

    def __repr__(self) -> str:
        return (
            f"BinrefArray(buffer={self.buffer!r}, shape={self.shape!r}, "
            f"dtype={self.dtype!r})"
        )


def _load_ref(val: BinrefArray, context: dict[str, Any]) -> np.ndarray:
    """Load a :class:`BinrefArray`'s buffer into a NumPy array.

    The buffer path is relative to the served ``output_path``, which is also
    where binref serialization resolves relative paths from, so we reuse the
    binref loader with it as ``base_dir``.
    """
    from tesseract_core.runtime.config import get_config

    base_dir = context.get("base_dir", get_config().output_path)
    return _load_binref_arraydict(val.to_arraydict(), base_dir)


def validate_binref_array(
    val: BinrefArray, expected_shape: ShapeType, expected_dtype: str | None
) -> BinrefArray:
    """Validate a :class:`BinrefArray`'s shape/dtype without reading the buffer.

    Returns the reference unchanged so it can later be forwarded verbatim (see
    :func:`encode_array`). Only the reference's declared ``shape``/``dtype`` are
    inspected -- no file is opened. Mirrors the shape/dtype checks in
    :func:`_coerce_shape_dtype` but never casts (a cast would require reading and
    rewriting the buffer, defeating the point of a passthrough).
    """
    shape = val.shape
    dtype_name = val.dtype

    if expected_shape is not Ellipsis:
        if len(shape) != len(expected_shape) or any(
            exp is not None and got != exp
            for got, exp in zip(shape, expected_shape, strict=False)
        ):
            raise PydanticCustomError(
                "array_shape_mismatch",
                "Binref array shape {actual_shape} is incompatible with expected "
                "shape {expected_shape}",
                {"actual_shape": shape, "expected_shape": tuple(expected_shape)},
            )

    allowed_dtypes = [dtype.lower() for dtype in get_args(AllowedDtypes)]
    if dtype_name not in allowed_dtypes:
        raise PydanticCustomError(
            "array_invalid_dtype",
            "Binref array has unsupported dtype '{actual_dtype}'; must be one "
            "of: {allowed_dtypes}",
            {"actual_dtype": dtype_name, "allowed_dtypes": ", ".join(allowed_dtypes)},
        )

    if expected_dtype is not None and dtype_name != expected_dtype:
        raise PydanticCustomError(
            "array_dtype_mismatch",
            "Binref array dtype '{actual_dtype}' does not match expected dtype "
            "'{expected_dtype}' (a passthrough binref is not cast)",
            {"actual_dtype": dtype_name, "expected_dtype": expected_dtype},
        )

    return val


def _coerce_shape_dtype(
    arr: ArrayLike,
    expected_shape: ShapeType,
    expected_dtype: str | None,
    context: dict[str, Any] | None = None,
) -> ArrayLike:
    """Coerce the shape and dtype of the passed array to the expected values.

    The behavior can be controlled via the ``context`` dict:

    - ``strict_shapes`` (bool): When True, reject arrays whose shape doesn't
      match the expected shape exactly (no broadcasting of size-1 dims).
    - ``strict_types`` (bool): When True, reject arrays whose dtype doesn't
      match the expected dtype exactly (no same-kind casting).
    """
    # NOTE: When making changes here, be mindful that this function is called on
    # every array validation, and inefficient code here can cause significant performance
    # issues, especially for large arrays. In particular, avoid any operations that copy the
    # array data (like astype or tolist) unless necessary.

    if context is None:
        context = {}

    strict_shapes = context.get("strict_shapes", False)
    strict_types = context.get("strict_types", False)

    if expected_shape is Ellipsis:
        # No shape check
        out_shape = arr.shape
    else:
        if len(arr.shape) != len(expected_shape):
            raise PydanticCustomError(
                "array_dimensionality_mismatch",
                "Array has wrong number of dimensions: got {actual_dims}D, expected {expected_dims}D",
                {
                    "actual_dims": len(arr.shape),
                    "expected_dims": len(expected_shape),
                },
            )

        out_shape = tuple(
            # Polymorphic dims -> keep the passed shape
            arr.shape[i] if expected_shape[i] is None else expected_shape[i]
            for i in range(len(expected_shape))
        )

    if strict_shapes and arr.shape != out_shape:
        raise PydanticCustomError(
            "array_shape_mismatch",
            "Array shape {actual_shape} does not match expected shape {expected_shape} "
            "(strict_shapes=True, no broadcasting)",
            {
                "actual_shape": arr.shape,
                "expected_shape": out_shape,
            },
        )

    # Broadcast the arr to the expected shape and dtype
    try:
        arr = np.broadcast_to(arr, out_shape)
    except ValueError:
        raise PydanticCustomError(
            "array_shape_mismatch",
            "Array shape {actual_shape} is incompatible with expected shape {expected_shape}",
            {
                "actual_shape": arr.shape,
                "expected_shape": out_shape,
            },
        ) from None

    if expected_dtype is not None:
        if strict_types:
            if str(arr.dtype) != expected_dtype:
                raise PydanticCustomError(
                    "array_dtype_mismatch",
                    "Array dtype '{actual_dtype}' does not match expected dtype '{expected_dtype}' "
                    "(strict_types=True, no casting)",
                    {
                        "actual_dtype": str(arr.dtype),
                        "expected_dtype": expected_dtype,
                    },
                )
        elif not np.can_cast(arr.dtype, expected_dtype, casting="same_kind"):
            raise PydanticCustomError(
                "array_dtype_mismatch",
                "Array dtype '{actual_dtype}' cannot be safely cast to '{expected_dtype}'",
                {
                    "actual_dtype": str(arr.dtype),
                    "expected_dtype": expected_dtype,
                },
            )
        arr = arr.astype(expected_dtype, copy=False)

    allowed_dtypes = [dtype.lower() for dtype in get_args(AllowedDtypes)]
    if arr.dtype.name not in allowed_dtypes:
        raise PydanticCustomError(
            "array_invalid_dtype",
            "Array has unsupported dtype '{actual_dtype}'; must be one of: {allowed_dtypes}",
            {
                "actual_dtype": arr.dtype.name,
                "allowed_dtypes": ", ".join(allowed_dtypes),
            },
        )

    if not out_shape:
        # Cast to a scalar type
        return arr.dtype.type(arr)

    return arr


def python_to_array(
    val: Any,
    expected_shape: ShapeType,
    expected_dtype: str | None,
    context: dict[str, Any] | None = None,
) -> ArrayLike:
    """Coerce a Python object to a NumPy array of the given shape and dtype.

    Always materialises a host NumPy array. ``context`` carries the validation
    flags consumed by :func:`_coerce_shape_dtype` (e.g. ``strict_shapes`` /
    ``strict_types``). Callers that want to keep a GPU array on-device (e.g. to
    encode it via CUDA IPC) must special-case it *before* calling this -- see
    :func:`validate_python_or_gpu_array` for the input-validation path and
    :func:`encode_array` for serialization.
    """
    val = np.asarray(val, order="C")
    if not np.issubdtype(val.dtype, np.number) and not np.issubdtype(
        val.dtype, np.bool_
    ):
        raise PydanticCustomError(
            "array_non_numeric",
            "Could not parse value as a numeric array (contains non-numeric data)",
            {},
        )
    return _coerce_shape_dtype(val, expected_shape, expected_dtype, context)


def validate_python_or_gpu_array(
    val: Any,
    info: ValidationInfo,
    expected_shape: ShapeType,
    expected_dtype: str | None,
) -> ArrayLike | GPUArray:
    """Validate a Python array-like input, keeping GPU arrays on-device.

    Used as the "load from a Python object" validator. Objects that live in GPU
    memory (exposing ``__cuda_array_interface__``) are validated but returned
    unchanged, so they can later be encoded via CUDA IPC without a host copy;
    coercing them to NumPy here would force a device-to-host transfer (or fail,
    since CuPy refuses implicit conversion). Everything else is coerced to a
    NumPy array via :func:`python_to_array`.
    """
    from tesseract_core.runtime.cuda import ipc as cuda_ipc

    # A binref reference: validate its declared shape/dtype but leave the buffer
    # on disk, so it can later be forwarded verbatim (see encode_array). Mirrors
    # the GPU-array passthrough below.
    if isinstance(val, BinrefArray):
        return validate_binref_array(val, expected_shape, expected_dtype)

    if cuda_ipc.has_cuda_array_interface(val):
        return cuda_ipc.validate_cuda_array(val, expected_shape, expected_dtype)

    context = info.context if info.context else {}
    return python_to_array(val, expected_shape, expected_dtype, context)


def decode_array(
    val: EncodedArrayModel,
    info: ValidationInfo,
    expected_shape: ShapeType,
    expected_dtype: str | None,
) -> ArrayLike:
    """Decode an EncodedArrayModel to a NumPy array."""
    from tesseract_core.runtime.config import get_config

    context = info.context if info.context else {}

    try:
        if val.data.encoding == "base64":
            data = _load_base64_arraydict(val.model_dump())

        elif val.data.encoding == "binref":
            base_dir = context.get("base_dir", get_config().input_path)
            subdir = context.get("binref_dir", None)
            if subdir is not None:
                base_dir = join_paths(base_dir, subdir)
            data = _load_binref_arraydict(val.model_dump(), base_dir)

        elif val.data.encoding == "cuda_ipc":
            from tesseract_core.runtime.device_transport import get_transport

            # Returns a framework-agnostic on-GPU wrapper — skip numpy coercion
            transport = get_transport(val.data.encoding)
            return transport.receive(val.model_dump())

        # keep checking for "raw" for backwards compat
        elif val.data.encoding in {"json", "raw"}:
            data = np.asarray(val.data.buffer).reshape(val.shape)
            if np.issubdtype(data.dtype, np.floating) and np.issubdtype(
                val.dtype, np.integer
            ):
                if np.any(data % 1):
                    raise PydanticCustomError(
                        "array_expected_integer",
                        "Expected integer data, but array contains floating point values",
                        {},
                    )
            data = data.astype(val.dtype, casting="unsafe", copy=False)

        else:
            # Unreachable
            raise AssertionError(f"Unsupported encoding: {val.data.encoding}")

    except PydanticCustomError:
        # Re-raise PydanticCustomError directly without wrapping
        raise
    except Exception as e:
        raise PydanticCustomError(
            "array_decode_error",
            "Failed to decode array buffer ({encoding} encoding): {error}",
            {"encoding": val.data.encoding, "error": str(e)},
        ) from e

    data = _coerce_shape_dtype(data, expected_shape, expected_dtype, context)
    return data


def encode_array(
    arr: ArrayLike | BinrefArray | GPUArray,
    info: Any,
    expected_shape: ShapeType,
    expected_dtype: str | None,
) -> ArrayDict | ArrayLike | BinrefArray | GPUArray:
    """Encode a NumPy or GPU array for serialization.

    An output encoding is two orthogonal choices carried in the context (see
    :func:`tesseract_core.runtime.file_interactions.output_to_bytes`):

    - ``array_encoding`` -- how a host (CPU) array is serialized (``json`` /
      ``base64`` / ``binref``);
    - ``device_transport`` -- how a device (GPU) array is exported without a
      host copy (a transport name such as ``cuda_ipc``, or ``None`` for none).

    Each array is routed per-leaf by *where it lives*, not by a single
    whole-response choice: a GPU array is exported over the device transport when
    one is set, and every host array (plus any GPU array when no transport is
    set) is serialized via the host encoding. In Python mode there is nothing to
    serialize, so arrays pass through as-is.

    An :class:`Array` field may also hold a :class:`BinrefArray` -- an on-disk
    buffer forwarded verbatim for binref output, or loaded and re-encoded for any
    other host encoding.
    """
    from tesseract_core.runtime.config import get_config
    from tesseract_core.runtime.cuda import ipc as cuda_ipc

    context = info.context if info.context else {}
    array_encoding = context.get("array_encoding", "json")
    device_transport = context.get("device_transport")

    # A binref reference. In Python mode pass it through untouched so it survives
    # the runtime's model_dump()/validate round-trip (see runtime.core.apply). In
    # JSON mode with binref output, forward the on-disk buffer verbatim -- the
    # whole point, no read-back. For any other host encoding, load the buffer once
    # and fall through to the normal numpy path so the field behaves like a normal
    # Array of the same shape and dtype.
    if isinstance(arr, BinrefArray):
        if not info.mode_is_json():
            return arr
        if array_encoding == "binref":
            return arr.to_arraydict()
        # Any other encoding must inline the data, so the on-disk buffer has to
        # be read into memory -- the exact cost a BinrefArray exists to avoid.
        # Warn loudly (the values are still correct) so this is not a silent
        # memory blow-up; request json+binref output to forward it from disk.
        nbytes = int(np.prod(arr.shape)) * np.dtype(arr.dtype).itemsize
        warnings.warn(
            f"A BinrefArray ({nbytes / 1024**2:.1f} MiB) is being read into "
            f"memory to satisfy a '{array_encoding}' response; the on-disk buffer "
            "cannot be forwarded for this encoding. Request 'json+binref' output "
            "to stream it from disk without loading it.",
            RuntimeWarning,
            stacklevel=2,
        )
        arr = _load_ref(arr, context)

    is_gpu_array = cuda_ipc.has_cuda_array_interface(arr)

    # Python mode -> return the array as-is, without any host copy. GPU arrays
    # are preserved on-device so that the intermediate model_dump()/validate
    # round-trip in the runtime (see runtime.core.apply) is lossless.
    if not info.mode_is_json():
        if is_gpu_array:
            return arr
        return python_to_array(arr, expected_shape, expected_dtype, context)

    # A GPU array with a device transport set is exported by reference, staying
    # on-device. A GPU array without a transport, or any host array, falls
    # through to the host encoding below -- so a mixed payload (some GPU, some
    # CPU arrays) serializes each leaf by where it lives instead of failing.
    if device_transport is not None and is_gpu_array:
        from tesseract_core.runtime.device_transport import get_transport

        transport = get_transport(device_transport)
        return transport.descriptor(transport.register(arr))

    # Host encoding: the data must reach the host. A GPU array survived
    # validation untouched (see validate_python_or_gpu_array), so materialise it
    # here with an explicit device-to-host copy before the numpy-based coercion.
    if is_gpu_array and not isinstance(arr, np.ndarray):
        arr = cuda_ipc.cuda_array_to_host(arr)

    # Convert to a NumPy array if necessary
    arr = python_to_array(arr, expected_shape, expected_dtype, context)
    if array_encoding == "base64":
        return _dump_base64_arraydict(arr, compression=context.get("compression"))
    elif array_encoding == "binref":
        base_dir = context.get("base_dir", get_config().output_path)
        subdir = context.get("binref_dir", None)
        data, new_binref_uuid = _dump_binref_arraydict(
            arr,
            base_dir=base_dir,
            subdir=subdir,
            current_binref_uuid=context.get("__binref_uuid", str(uuid4())),
            max_file_size=context.get("max_file_size", MAX_BINREF_BUFFER_SIZE),
            compression=context.get("compression"),
        )
        context["__binref_uuid"] = new_binref_uuid
        return data
    elif array_encoding == "json":
        return _dump_json_arraydict(arr)
    else:
        # Unreachable
        raise AssertionError(f"Unsupported encoding: {array_encoding}")
