# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import re
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
    model_config = ConfigDict(extra="forbid")


class CudaIpcArrayData(BaseModel):
    """Data structure for CUDA IPC shared GPU memory handles.

    The handle is the base64-encoded 64-byte cudaIpcMemHandle_t.
    The device is the CUDA device ordinal the memory lives on.
    storage_offset is the byte offset within the cudaMalloc allocation.
    storage_size is the total size in bytes of the cudaMalloc allocation.
    """

    handle: StrictStr = Field(
        description="Base64-encoded cudaIpcMemHandle_t (64 bytes)"
    )
    device: int = Field(description="CUDA device ordinal")
    storage_offset: int = Field(default=0, description="Byte offset within allocation")
    storage_size: int = Field(description="Total allocation size in bytes")
    encoding: Literal["cuda_ipc"]
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


def _has_cuda_array_interface(obj: Any) -> bool:
    """Check if an object exposes the __cuda_array_interface__ protocol.

    This protocol is supported by PyTorch, CuPy, JAX, Numba, and any
    CUDA-aware Python library. It indicates the object holds data in
    GPU device memory.
    """
    return hasattr(obj, "__cuda_array_interface__")


def _cuda_array_to_host(arr: Any) -> np.ndarray:
    """Copy a GPU array to a host NumPy array (explicit device-to-host copy).

    Handles CuPy (``.get()``) and PyTorch (``.cpu().numpy()``). Used for non-IPC
    encodings, where the bytes must reach the host.
    """
    get = getattr(arr, "get", None)
    if callable(get):  # CuPy
        return np.asarray(get())
    cpu = getattr(arr, "cpu", None)
    if callable(cpu):  # PyTorch
        return np.asarray(cpu().numpy())
    raise TypeError(
        f"Cannot copy GPU array of type {type(arr).__name__} to host; "
        "expected a CuPy or PyTorch array."
    )


def _get_cuda_array_info(arr: Any) -> tuple[int, int, tuple[int, ...], str]:
    """Extract (device_ptr, nbytes, shape, numpy_dtype_str) from a CUDA array.

    Works with any object that implements __cuda_array_interface__ (v2+):
    PyTorch tensors, CuPy arrays, JAX DeviceArrays, Numba device arrays, etc.
    """
    iface = arr.__cuda_array_interface__
    data_ptr = iface["data"][0]
    shape = tuple(iface["shape"])
    typestr = iface["typestr"]  # e.g. "<f4", "|b1"
    dtype = np.dtype(typestr)
    nbytes = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
    return data_ptr, nbytes, shape, dtype.name


def _is_c_contiguous(arr: Any) -> bool:
    """Whether a CUDA array's memory is C-contiguous per __cuda_array_interface__.

    cuda_ipc transfers a flat, contiguous byte range: encode copies (or hands
    off) ``prod(shape) * itemsize`` consecutive bytes and decode rebuilds a
    contiguous array from shape/dtype alone (the payload carries no strides). A
    non-contiguous source would therefore be silently misread, so callers must
    reject it.

    Per the protocol, ``strides = None`` means row-major contiguous. A non-None
    ``strides`` is still contiguous iff it equals the row-major strides implied
    by shape and itemsize.
    """
    iface = arr.__cuda_array_interface__
    strides = iface.get("strides")
    if strides is None:
        return True
    shape = tuple(iface["shape"])
    itemsize = np.dtype(iface["typestr"]).itemsize
    expected = []
    acc = itemsize
    for dim in reversed(shape):
        expected.append(acc)
        acc *= dim
    expected.reverse()
    return tuple(strides) == tuple(expected)


# ---------------------------------------------------------------------------
# CUDA IPC via ctypes (framework-agnostic, no torch/cupy dependency)
# ---------------------------------------------------------------------------

_CUDART_HANDLE = None
_CUDA_IPC_HANDLE_SIZE = 64  # cudaIpcMemHandle_t is always 64 bytes

# cudaIpcMemLazyEnablePeerAccess
_CUDA_IPC_LAZY_ENABLE_PEER_ACCESS = 0x01


class _CudaIpcMemHandle(ctypes.Structure):
    """ctypes mirror of ``cudaIpcMemHandle_t`` (an opaque 64-byte blob).

    This *must* be a Structure (not a bare ``c_byte`` array) so that ctypes
    passes it **by value** to the CUDA runtime, matching the C ABI where
    ``cudaIpcMemHandle_t`` is a struct argument. Passing a ``c_byte`` array
    would be marshalled as a pointer, which makes ``cudaIpcOpenMemHandle``
    fail with ``cudaErrorInvalidValue`` (error code 1).
    """

    _fields_ = [("reserved", ctypes.c_byte * _CUDA_IPC_HANDLE_SIZE)]


def _get_cudart():
    """Lazily load the CUDA runtime shared library and declare signatures."""
    global _CUDART_HANDLE
    if _CUDART_HANDLE is not None:
        return _CUDART_HANDLE

    import ctypes.util

    cudart = None
    # Try common names for the CUDA runtime library
    for name in ("cudart", "cudart64_13", "cudart64_12", "cudart64_11"):
        path = ctypes.util.find_library(name)
        if path:
            cudart = ctypes.CDLL(path)
            break

    if cudart is None:
        # Fallback: try loading directly (covers systems where find_library
        # misses the versioned soname).
        for path in (
            "libcudart.so",
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so.11",
            "libcudart.dylib",
            "cudart64_13.dll",
            "cudart64_12.dll",
        ):
            try:
                cudart = ctypes.CDLL(path)
                break
            except OSError:
                continue

    if cudart is None:
        raise RuntimeError(
            "Could not find CUDA runtime library (libcudart). "
            "Make sure CUDA is installed and LD_LIBRARY_PATH is set."
        )

    # Declare argument/return types so ctypes marshals 64-bit pointers and the
    # 64-byte handle struct correctly (defaults assume C int, which truncates
    # pointers and passes structs by reference).
    cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    cudart.cudaSetDevice.restype = ctypes.c_int
    cudart.cudaIpcGetMemHandle.argtypes = [
        ctypes.POINTER(_CudaIpcMemHandle),
        ctypes.c_void_p,
    ]
    cudart.cudaIpcGetMemHandle.restype = ctypes.c_int
    # NOTE: the handle is the second argument *by value* (a struct), not a
    # pointer -- this is the crux of getting IPC to work through ctypes.
    cudart.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _CudaIpcMemHandle,
        ctypes.c_uint,
    ]
    cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int
    cudart.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    cudart.cudaIpcCloseMemHandle.restype = ctypes.c_int
    cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
    cudart.cudaGetErrorString.restype = ctypes.c_char_p
    # Used by the VMM staging-buffer fallback (see _stage_for_legacy_ipc).
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int
    cudart.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    cudart.cudaMemcpy.restype = ctypes.c_int

    _CUDART_HANDLE = cudart
    return _CUDART_HANDLE


_CUDA_DRIVER_HANDLE = None


def _get_cuda_driver():
    """Lazily load the CUDA driver library (libcuda) and declare signatures.

    The driver API is only needed for ``cuMemGetAddressRange``, which recovers
    the base pointer and size of the allocation backing a device pointer. This
    is required because IPC handles reference the *whole* allocation, while a
    given array may point partway into it (common with pooled allocators like
    CuPy and PyTorch).
    """
    global _CUDA_DRIVER_HANDLE
    if _CUDA_DRIVER_HANDLE is not None:
        return _CUDA_DRIVER_HANDLE

    import ctypes.util

    driver = None
    for name in ("cuda",):
        path = ctypes.util.find_library(name)
        if path:
            driver = ctypes.CDLL(path)
            break
    if driver is None:
        for path in ("libcuda.so", "libcuda.so.1", "nvcuda.dll"):
            try:
                driver = ctypes.CDLL(path)
                break
            except OSError:
                continue
    if driver is None:
        raise RuntimeError(
            "Could not find CUDA driver library (libcuda). "
            "Make sure an NVIDIA driver is installed."
        )

    # CUdeviceptr is an unsigned integer the width of a pointer.
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuMemGetAddressRange_v2.argtypes = [
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_ulonglong,
    ]
    driver.cuMemGetAddressRange_v2.restype = ctypes.c_int
    driver.cuInit(0)

    _CUDA_DRIVER_HANDLE = driver
    return _CUDA_DRIVER_HANDLE


def _cuda_get_allocation_base(device_ptr: int) -> tuple[int, int]:
    """Return ``(base_ptr, size)`` of the allocation containing ``device_ptr``.

    Uses the driver API ``cuMemGetAddressRange`` so that pointers into the
    middle of a (possibly pooled) allocation are resolved to the base pointer
    an IPC handle actually references, plus the byte offset can be derived as
    ``device_ptr - base_ptr``.
    """
    driver = _get_cuda_driver()
    base = ctypes.c_ulonglong()
    size = ctypes.c_size_t()
    ret = driver.cuMemGetAddressRange_v2(
        ctypes.byref(base), ctypes.byref(size), ctypes.c_ulonglong(device_ptr)
    )
    if ret != 0:
        raise RuntimeError(f"cuMemGetAddressRange failed with error code {ret}")
    return base.value, size.value


def _cuda_error_string(cudart: Any, code: int) -> str:
    """Best-effort human-readable CUDA error string for an error code."""
    try:
        msg = cudart.cudaGetErrorString(code)
        if msg:
            return msg.decode()
    except Exception:
        pass
    return f"error code {code}"


def _cuda_ipc_get_mem_handle(device_ptr: int) -> bytes:
    """Call cudaIpcGetMemHandle for a device pointer. Returns 64 raw bytes.

    ``device_ptr`` should be the *base* of the allocation (see
    :func:`_cuda_get_allocation_base`); IPC handles always reference the whole
    underlying allocation.

    Raises ``RuntimeError`` if the pointer is rejected by the legacy IPC API
    (e.g. VMM/pool-backed memory -- see :func:`_stage_for_legacy_ipc`, which
    callers should fall back to on failure).
    """
    cudart = _get_cudart()
    handle = _CudaIpcMemHandle()
    ret = cudart.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(device_ptr))
    if ret != 0:
        raise RuntimeError(
            f"cudaIpcGetMemHandle failed: {_cuda_error_string(cudart, ret)}"
        )
    return bytes(handle.reserved)


def _cuda_ipc_open_mem_handle(handle_bytes: bytes, device: int) -> int:
    """Call cudaIpcOpenMemHandle. Returns the base device pointer (int).

    The returned pointer is the base of the producer's allocation as mapped
    into this process; callers must add any per-array byte offset themselves.
    """
    cudart = _get_cudart()

    # Set the target device (IPC memory must be opened on the device it lives
    # on).
    ret = cudart.cudaSetDevice(device)
    if ret != 0:
        raise RuntimeError(
            f"cudaSetDevice({device}) failed: {_cuda_error_string(cudart, ret)}"
        )

    handle = _CudaIpcMemHandle()
    ctypes.memmove(handle.reserved, handle_bytes, _CUDA_IPC_HANDLE_SIZE)
    dev_ptr = ctypes.c_void_p()
    ret = cudart.cudaIpcOpenMemHandle(
        ctypes.byref(dev_ptr), handle, ctypes.c_uint(_CUDA_IPC_LAZY_ENABLE_PEER_ACCESS)
    )
    if ret != 0:
        raise RuntimeError(
            f"cudaIpcOpenMemHandle failed: {_cuda_error_string(cudart, ret)}"
        )
    if dev_ptr.value is None:
        raise RuntimeError("cudaIpcOpenMemHandle returned a null pointer")
    return dev_ptr.value


def _cuda_ipc_close_mem_handle(device_ptr: int) -> None:
    """Call cudaIpcCloseMemHandle to release an IPC-opened base device pointer.

    ``device_ptr`` must be the base pointer returned by
    :func:`_cuda_ipc_open_mem_handle` (not an offset pointer into it).
    """
    cudart = _get_cudart()
    ret = cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(device_ptr))
    if ret != 0:
        raise RuntimeError(
            f"cudaIpcCloseMemHandle failed: {_cuda_error_string(cudart, ret)}"
        )


_cudaMemcpyDeviceToDevice = 3


def _stage_for_legacy_ipc(base_ptr: int, storage_size: int) -> int:
    """Copy a VMM/pool-backed allocation into a fresh ``cudaMalloc`` buffer.

    The legacy ``cudaIpcGetMemHandle`` API rejects memory that CUDA's Virtual
    Memory Management API (``cuMemCreate``/``cuMemAddressReserve``) allocated
    -- which is what modern pool allocators use, including JAX/XLA's default
    GPU allocator (confirmed: ``cudaIpcGetMemHandle`` returns
    ``cudaErrorInvalidValue`` for such pointers; CuPy's and PyTorch's default
    caching allocators happen to use plain ``cudaMalloc`` pools, so they don't
    hit this).

    Rather than replicate CUDA's VMM export path (which requires transferring
    a POSIX file descriptor between processes via ``SCM_RIGHTS`` over a Unix
    domain socket -- a real fd, not just its integer value, is meaningless in
    another process's fd table), we take the simpler route of copying the data
    device-to-device into a plain ``cudaMalloc`` allocation, which *is*
    IPC-exportable via the legacy API. This costs one on-GPU copy but avoids a
    new cross-process handshake; it is still far cheaper than a host round-trip.

    Returns the device pointer of the new (caller-owned, offset-zero) buffer.
    The caller is responsible for freeing it via ``cudaFree`` once the export
    is no longer needed (see :func:`_release_cuda_ipc_exports`).
    """
    cudart = _get_cudart()
    staging_ptr = ctypes.c_void_p()
    ret = cudart.cudaMalloc(ctypes.byref(staging_ptr), ctypes.c_size_t(storage_size))
    if ret != 0:
        raise RuntimeError(f"cudaMalloc failed: {_cuda_error_string(cudart, ret)}")

    ret = cudart.cudaMemcpy(
        staging_ptr,
        ctypes.c_void_p(base_ptr),
        ctypes.c_size_t(storage_size),
        ctypes.c_int(_cudaMemcpyDeviceToDevice),
    )
    if ret != 0:
        cudart.cudaFree(staging_ptr)
        raise RuntimeError(f"cudaMemcpy failed: {_cuda_error_string(cudart, ret)}")

    return staging_ptr.value


# ---------------------------------------------------------------------------
# CUDA IPC array encode / decode
# ---------------------------------------------------------------------------

# Producer-side keepalive registry (a "ring" of depth 1).
#
# A CUDA IPC handle is only valid while the *producer* keeps the source
# allocation alive. In a request/response server the output array would
# otherwise be freed the instant the handler returns -- possibly before the
# client has copied it out -- and a pooled allocator (CuPy/PyTorch) could then
# recycle the block, so the client would silently read *wrong* data.
#
# We therefore retain the arrays exported during the *current* request here, and
# release them at the START of the next request (see
# :func:`_release_cuda_ipc_exports`). This bounds pinned GPU memory to a single
# request's worth of outputs.
#
# This is correct under two documented assumptions:
#   1. Clients issue cuda_ipc requests *serially* (never concurrently). A serial
#      client's call to request N+1 cannot begin until its handling of the
#      response to request N has fully returned -- and the client copies each
#      output into its own buffer before returning (see
#      :func:`_load_cuda_ipc_arraydict`). So by the time request N+1 releases
#      request N's exports, the client is provably done borrowing them.
#   2. The client copies decoded outputs into client-owned memory (which the
#      decode path does unconditionally).
_CUDA_IPC_EXPORT_REGISTRY: list[Any] = []

# Device pointers of VMM-fallback staging buffers (see _stage_for_legacy_ipc)
# awaiting cudaFree. Kept separate from _CUDA_IPC_EXPORT_REGISTRY (which holds
# plain pinned array references) since these need an explicit free call
# instead of just dropping a reference, but are released at the same point and
# for the same reasons.
_CUDA_IPC_STAGING_BUFFERS: list[int] = []


def _release_cuda_ipc_exports() -> None:
    """Release producer-side arrays pinned for CUDA IPC export by the last request.

    Called at the start of each runtime request so the previous request's
    exported GPU buffers are freed just before the next request runs, while
    still having survived the (serial) client's copy-out of that request's
    outputs.
    """
    _CUDA_IPC_EXPORT_REGISTRY.clear()

    staging_ptrs, _CUDA_IPC_STAGING_BUFFERS[:] = list(_CUDA_IPC_STAGING_BUFFERS), []
    if staging_ptrs:
        cudart = _get_cudart()
        for ptr in staging_ptrs:
            cudart.cudaFree(ctypes.c_void_p(ptr))


def _pin_cuda_ipc_export(arr: Any) -> None:
    """Retain a reference to a source array so its GPU memory stays valid.

    The reference is held until the next request calls
    :func:`_release_cuda_ipc_exports`.
    """
    _CUDA_IPC_EXPORT_REGISTRY.append(arr)


def _pin_cuda_ipc_staging_buffer(device_ptr: int) -> None:
    """Register a VMM-fallback staging buffer (see :func:`_stage_for_legacy_ipc`) for cudaFree.

    Freed at the same point ordinary pinned arrays are released (start of the
    next request), for the same reasons (see the module comment above).
    """
    _CUDA_IPC_STAGING_BUFFERS.append(device_ptr)


def _dump_cuda_ipc_arraydict(arr: Any) -> ArrayDict:
    """Dump a CUDA array to a JSON dict with a CUDA IPC handle.

    Works with any object that implements __cuda_array_interface__:
    PyTorch tensors, CuPy arrays, JAX DeviceArrays, etc.

    The IPC handle allows another process on the same host (with --ipc=host)
    to access the GPU memory directly without any CPU round-trip.

    The source array is pinned in a process-global registry (see
    :func:`_pin_cuda_ipc_export`) so its GPU memory is not freed or recycled
    before the consumer copies it out; the pin is released at the start of the
    next request (:func:`_release_cuda_ipc_exports`).

    Frameworks with VMM/pool-backed GPU allocators (e.g. JAX/XLA) hand out
    pointers that the legacy ``cudaIpcGetMemHandle`` API rejects; in that case
    this transparently falls back to staging the array's bytes into a fresh
    ``cudaMalloc`` buffer via one on-GPU copy (see :func:`_stage_for_legacy_ipc`)
    and exports a handle to that instead. Still far cheaper than a host round-trip.
    """
    if not _has_cuda_array_interface(arr):
        raise ValueError(
            "cuda_ipc encoding requires a CUDA array "
            f"(object with __cuda_array_interface__), got {type(arr).__name__}"
        )

    if not _is_c_contiguous(arr):
        raise ValueError(
            "cuda_ipc encoding requires a C-contiguous array; got one with "
            f"strides {arr.__cuda_array_interface__.get('strides')}. Make a "
            "contiguous copy first (e.g. cupy.ascontiguousarray / "
            "torch.Tensor.contiguous)."
        )

    data_ptr, nbytes, shape, dtype_name = _get_cuda_array_info(arr)

    # Keep the source allocation alive until exports are explicitly released.
    # (Still needed even on the VMM fallback path below: _stage_for_legacy_ipc
    # reads from `arr`'s memory synchronously before returning, but keeping the
    # pin simplifies the two paths to an identical cleanup story.)
    _pin_cuda_ipc_export(arr)

    # IPC handles reference the *whole* backing allocation, not the array's
    # (possibly offset) data pointer. Pooled allocators (CuPy, PyTorch) hand
    # out many arrays from a single cudaMalloc block, so we must resolve the
    # allocation base, take the handle on that base, and record the byte offset
    # of this array within the allocation.
    base_ptr, storage_size = _cuda_get_allocation_base(data_ptr)
    storage_offset = data_ptr - base_ptr

    # Get the IPC handle for the base of the allocation.
    try:
        handle_bytes = _cuda_ipc_get_mem_handle(base_ptr)
    except RuntimeError:
        # Legacy IPC rejected this pointer -- almost certainly because it's
        # VMM/pool-backed (see _stage_for_legacy_ipc). Copy just this array's
        # own bytes (not the whole, possibly huge, backing allocation) into a
        # fresh cudaMalloc buffer and export a handle to *that* instead.
        staging_ptr = _stage_for_legacy_ipc(data_ptr, nbytes)
        _pin_cuda_ipc_staging_buffer(staging_ptr)
        storage_offset = 0
        storage_size = nbytes
        handle_bytes = _cuda_ipc_get_mem_handle(staging_ptr)

    # Determine device ordinal
    device = 0
    # CuPy arrays expose .device.id, torch tensors expose .device.index
    if hasattr(arr, "device"):
        dev = arr.device
        if hasattr(dev, "id"):
            device = dev.id  # CuPy
        elif hasattr(dev, "index") and dev.index is not None:
            device = dev.index  # PyTorch

    return {
        "object_type": "array",
        "shape": list(shape),
        "dtype": dtype_name,
        "data": {
            "handle": pybase64.b64encode_as_string(handle_bytes),
            "device": device,
            "storage_offset": storage_offset,
            "storage_size": storage_size,
            "encoding": "cuda_ipc",
        },
    }


def _load_cuda_ipc_arraydict(val: ArrayDict) -> Any:
    """Load a CUDA array from a JSON dict with a CUDA IPC handle.

    The calling process must share the IPC namespace with the producer
    (e.g. both run with --ipc=host on Docker) and see the same GPU.

    Returns a freshly-allocated, caller-owned ``cupy.ndarray``: the IPC handle
    is opened, the data is copied device-to-device into memory owned by this
    process, and the IPC mapping is closed before returning. The borrow of the
    producer's memory therefore lasts only for a single on-GPU copy, so the
    producer is free to reuse or release the exported buffer as soon as this
    call returns.

    CuPy is the only hard requirement for decoding; downstream code can convert
    the returned array to PyTorch/JAX via DLPack or __cuda_array_interface__.
    """
    try:
        import cupy
    except ImportError:
        raise RuntimeError(
            "cuda_ipc decoding requires CuPy to be installed "
            "(pip install cupy-cuda12x or similar)."
        ) from None

    data = val["data"]
    handle_bytes = pybase64.b64decode(data["handle"], validate=True)
    device = data["device"]
    storage_size = data["storage_size"]
    storage_offset = data.get("storage_offset", 0)

    dtype = np.dtype(val["dtype"])
    shape = tuple(val["shape"])

    base_ptr = _cuda_ipc_open_mem_handle(handle_bytes, device)
    try:
        # Wrap the producer's (foreign) memory as an unowned view -- owner=None
        # so cupy never tries to free memory it does not own.
        mem = cupy.cuda.UnownedMemory(base_ptr, storage_size, owner=None)
        memptr = cupy.cuda.MemoryPointer(mem, storage_offset)
        view = cupy.ndarray(shape, dtype=dtype, memptr=memptr)
        # Copy into caller-owned memory and make sure the copy has completed
        # before we close the mapping (otherwise we could unmap mid-copy).
        with cupy.cuda.Device(device):
            owned = view.copy()
        cupy.cuda.runtime.deviceSynchronize()
        return owned
    finally:
        _cuda_ipc_close_mem_handle(base_ptr)


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


def _validate_cuda_array(
    val: Any, expected_shape: ShapeType, expected_dtype: str | None
) -> Any:
    """Validate a GPU array's shape/dtype without pulling it off the device.

    Returns the object unchanged so it can later be encoded via CUDA IPC (see
    :func:`encode_array`). Only the ``__cuda_array_interface__`` metadata is
    inspected -- no device-to-host copy or kernel launch occurs. Mirrors the
    shape/dtype checks in :func:`_coerce_shape_dtype`, but never casts (a cast
    would need a device copy the caller did not ask for).
    """
    _, _, shape, dtype_name = _get_cuda_array_info(val)

    # Shape: Ellipsis means "no check"; otherwise each dim must match unless the
    # expected dim is None (a polymorphic wildcard).
    if expected_shape is not Ellipsis:
        if len(shape) != len(expected_shape) or any(
            exp is not None and got != exp
            for got, exp in zip(shape, expected_shape, strict=False)
        ):
            raise PydanticCustomError(
                "array_shape_mismatch",
                "Array shape {actual_shape} is incompatible with expected "
                "shape {expected_shape}",
                {"actual_shape": shape, "expected_shape": tuple(expected_shape)},
            )

    allowed_dtypes = [dtype.lower() for dtype in get_args(AllowedDtypes)]
    if dtype_name not in allowed_dtypes:
        raise PydanticCustomError(
            "array_invalid_dtype",
            "Array has unsupported dtype '{actual_dtype}'; must be one of: "
            "{allowed_dtypes}",
            {"actual_dtype": dtype_name, "allowed_dtypes": ", ".join(allowed_dtypes)},
        )

    if expected_dtype is not None and dtype_name != expected_dtype:
        raise PydanticCustomError(
            "array_dtype_mismatch",
            "GPU array dtype '{actual_dtype}' does not match expected dtype "
            "'{expected_dtype}' (cuda_ipc does not cast on device)",
            {"actual_dtype": dtype_name, "expected_dtype": expected_dtype},
        )

    return val


def python_to_array(
    val: Any,
    info: ValidationInfo,
    expected_shape: ShapeType,
    expected_dtype: str | None,
) -> ArrayLike:
    """Convert a Python object to a NumPy array.

    Objects that live in GPU memory (exposing ``__cuda_array_interface__``) are
    passed through unchanged -- validated but not copied to the host -- so they
    can be encoded via CUDA IPC during serialization. Coercing them to NumPy
    here would force a device-to-host transfer (or fail, as CuPy refuses
    implicit conversion), defeating the purpose of ``json+cuda_ipc``.
    """
    if _has_cuda_array_interface(val):
        return _validate_cuda_array(val, expected_shape, expected_dtype)

    val = np.asarray(val, order="C")
    if not np.issubdtype(val.dtype, np.number) and not np.issubdtype(
        val.dtype, np.bool_
    ):
        raise PydanticCustomError(
            "array_non_numeric",
            "Could not parse value as a numeric array (contains non-numeric data)",
            {},
        )
    context = info.context if info.context else {}
    return _coerce_shape_dtype(val, expected_shape, expected_dtype, context)


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
            # Returns a cupy.ndarray on GPU — skip numpy coercion
            return _load_cuda_ipc_arraydict(val.model_dump())

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
    arr: ArrayLike, info: Any, expected_shape: ShapeType, expected_dtype: str | None
) -> ArrayDict | ArrayLike:
    """Encode a NumPy array for serialization.

    In Python mode, returns the raw array as-is.
    """
    from tesseract_core.runtime.config import get_config

    context = info.context if info.context else {}
    array_encoding = context.get("array_encoding", "json")

    # For cuda_ipc, skip numpy conversion — array stays on GPU
    if array_encoding == "cuda_ipc":
        if not info.mode_is_json():
            return arr
        if not _has_cuda_array_interface(arr):
            raise ValueError(
                "cuda_ipc encoding requires a CUDA array "
                f"(object with __cuda_array_interface__), got {type(arr).__name__}"
            )
        return _dump_cuda_ipc_arraydict(arr)

    # Python mode -> return the array as-is, without any host copy. GPU arrays
    # are preserved on-device so that the intermediate model_dump()/validate
    # round-trip in the runtime (see runtime.core.apply) is lossless.
    if not info.mode_is_json():
        if _has_cuda_array_interface(arr):
            return arr
        return python_to_array(arr, info, expected_shape, expected_dtype)

    # JSON, non-IPC encoding: the data must reach the host. A GPU array survived
    # validation untouched (see python_to_array), so materialise it here with an
    # explicit device-to-host copy before the numpy-based coercion.
    if _has_cuda_array_interface(arr) and not isinstance(arr, np.ndarray):
        arr = _cuda_array_to_host(arr)

    # Convert to a NumPy array if necessary
    arr = python_to_array(arr, info, expected_shape, expected_dtype)
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
