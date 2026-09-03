# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA IPC array encoding: zero-copy GPU array exchange between processes.

This module holds everything specific to the ``json+cuda_ipc`` encoding, kept
separate from the framework-agnostic host encodings in
:mod:`tesseract_core.runtime.array_encoding`. Nothing here is imported unless a
Tesseract actually encodes or decodes a CUDA IPC array, so the CUDA runtime and
driver libraries are only touched on that path.

The JSON schema for this encoding (``CudaIpcArrayData``) lives alongside the
other array-data models in :mod:`array_encoding`; this module holds the CUDA
runtime machinery. The public entry points used by :mod:`array_encoding` are:

* :func:`has_cuda_array_interface` / :func:`cuda_array_to_host` -- host-copy
  helpers for non-IPC encodings of GPU arrays,
* :func:`validate_cuda_array` -- shape/dtype validation without a device copy,
* :func:`dump_cuda_ipc_arraydict` / :func:`load_cuda_ipc_arraydict` -- the
  encode/decode pair,
* :func:`release_pinned_ipc_exports` -- keepalive cleanup, called once per
  request by both the server (for its outputs) and the client (for its inputs).

:func:`iter_cudart_candidates` is also public, but for a different audience:
out-of-process consumers that ``dlopen`` libcudart themselves (e.g. the
``tesseract_jax`` C++ FFI shim) can reuse this module's discovery -- including
the pip-wheel fallback and forward-compatible version range -- instead of
maintaining their own soname list.
"""

import ctypes
import ctypes.util
import importlib.util
import weakref
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pybase64
from pydantic_core import PydanticCustomError

from tesseract_core.runtime.array_encoding import AllowedDtypes, ArrayDict, ShapeType


def has_cuda_array_interface(obj: Any) -> bool:
    """Check if an object exposes the __cuda_array_interface__ protocol.

    This protocol is supported by PyTorch, CuPy, JAX, Numba, and any
    CUDA-aware Python library. It indicates the object holds data in
    GPU device memory.
    """
    return hasattr(obj, "__cuda_array_interface__")


def cuda_array_to_host(arr: Any) -> np.ndarray:
    """Copy a GPU array to a host NumPy array (explicit device-to-host copy).

    Used for non-IPC encodings, where the bytes must reach the host. Handles
    CuPy (``.get()``) and PyTorch (``.cpu().numpy()``) explicitly, then falls
    back to ``np.asarray`` for any other framework whose arrays support
    ``__array__`` (e.g. JAX, which fetches to host on conversion).
    """
    get = getattr(arr, "get", None)
    if callable(get):  # CuPy
        return np.asarray(get())
    cpu = getattr(arr, "cpu", None)
    if callable(cpu):  # PyTorch
        return np.asarray(cpu().numpy())
    # JAX arrays (and any other framework exposing __array__) fetch to host here;
    # CuPy deliberately raises rather than copy implicitly, which is why it is
    # handled explicitly above. Require __array__ so a bare object does not slip
    # through as a useless object-dtype array.
    if hasattr(arr, "__array__"):
        host = np.asarray(arr)
        if host.dtype != object:
            return host
    raise TypeError(
        f"Cannot copy GPU array of type {type(arr).__name__} to host; "
        "expected a CuPy, PyTorch, or __array__-convertible numeric array."
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


# CUDA runtime major versions we probe for by number, newest first. cuda_ipc
# only calls a handful of long-stable runtime symbols (cudaIpc*, cudaMemcpy,
# cudaMalloc, cudaSetDevice, cudaDeviceSynchronize), so a newer major than any
# listed here is very likely to work -- the range is generous and open-ended at
# the top precisely so a freshly released CUDA does not need a code change. The
# floor is the oldest major whose ABI we still expect to encounter in the wild.
_CUDART_MAJOR_NEWEST = 20
_CUDART_MAJOR_OLDEST = 11
_CUDART_MAJORS = tuple(range(_CUDART_MAJOR_NEWEST, _CUDART_MAJOR_OLDEST - 1, -1))


def _cudart_sonames() -> tuple[str, ...]:
    """Candidate CUDA runtime library filenames, most-preferred first.

    Unversioned names come first: on a system with a CUDA toolkit the loader
    resolves ``libcudart.so`` / ``libcudart.dylib`` via the dev symlink to
    whatever major is installed, so we never have to know the number. The
    versioned names that follow are generated over :data:`_CUDART_MAJORS`
    (newest first) rather than hand-enumerated, so a new CUDA major is picked up
    without editing this file.
    """
    names = ["libcudart.so", "libcudart.dylib"]
    names += [f"libcudart.so.{major}" for major in _CUDART_MAJORS]
    names += [f"cudart64_{major}.dll" for major in _CUDART_MAJORS]
    return tuple(names)


# Glob patterns for locating a wheel-shipped runtime by filename in a known lib
# directory. Unlike the loader-name list above, here we have a concrete
# directory to scan, so we can match *any* version present rather than probe a
# fixed set -- fully forward-compatible for the wheel case.
_CUDART_GLOBS = ("libcudart.so.*", "libcudart.so", "libcudart.dylib", "cudart64_*.dll")


def _cudart_soname_sort_key(path: Path) -> tuple[int, int]:
    """Sort key placing higher CUDA majors first among wheel candidates.

    Returns ``(0 if versioned else 1, -major)`` so ``sorted`` yields versioned
    names first and, among them, newest-major-first. The major is parsed from
    ``libcudart.so.<major>`` / ``cudart64_<major>.dll``; names without a
    parseable version (e.g. an unversioned ``libcudart.so`` symlink) sort after
    all versioned ones so a concrete version wins.
    """
    name = path.name
    major = -1
    if name.startswith("libcudart.so."):
        tail = name[len("libcudart.so.") :].split(".", 1)[0]
        if tail.isdigit():
            major = int(tail)
    elif name.startswith("cudart64_") and name.endswith(".dll"):
        tail = name[len("cudart64_") : -len(".dll")]
        if tail.isdigit():
            major = int(tail)
    # Versioned first (-major ascending == major descending); unversioned last.
    return (0 if major >= 0 else 1, -major)


def _iter_wheel_cudart_paths() -> Iterator[str]:
    """Yield candidate absolute paths to libcudart shipped inside pip wheels.

    The pip CUDA wheels (``nvidia-cuda-runtime-cuXX``, pulled in transitively by
    ``jax[cudaXX]`` / ``cupy-cudaXXx``) install the runtime under a
    ``site-packages/nvidia/<pkg>/lib/libcudart.so.NN`` directory (``<pkg>`` is
    typically ``cuda_runtime``). That directory is on neither
    ``LD_LIBRARY_PATH`` nor the ``ldconfig`` cache, so both
    :func:`ctypes.util.find_library` and a bare ``ctypes.CDLL(soname)`` miss it
    -- observed on GPU CI runners with no system CUDA toolkit installed. We
    locate the wheel directory ourselves so this venv-local runtime is found
    (and, per :func:`iter_cudart_candidates`, preferred over a system one).
    """

    def _spec_locations(name: str) -> Iterable[str]:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            return ()
        return () if spec is None else (spec.submodule_search_locations or ())

    lib_dirs: list[Path] = []
    seen: set[str] = set()

    def _add(lib_dir: Path) -> None:
        key = str(lib_dir)
        if key not in seen:
            seen.add(key)
            lib_dirs.append(lib_dir)

    # Preferred: ask importlib where the runtime wheel's package lives, so we do
    # not hardcode the site-packages layout.
    for location in _spec_locations("nvidia.cuda_runtime"):
        _add(Path(location) / "lib")

    # Fallback: glob every nvidia namespace package on the import path, in case
    # the runtime is bundled under a differently named package. Anchoring on the
    # nvidia namespace itself (rather than a fixed depth off __file__) keeps this
    # correct for editable installs and non-standard layouts.
    for location in _spec_locations("nvidia"):
        for lib in Path(location).glob("*/lib"):
            _add(lib)

    for lib_dir in lib_dirs:
        candidates: set[Path] = set()
        for pattern in _CUDART_GLOBS:
            candidates.update(p for p in lib_dir.glob(pattern) if p.is_file())
        # Newest major first, so a wheel dir that somehow holds several runtimes
        # (or a dev symlink alongside a versioned .so) prefers the highest one.
        for candidate in sorted(candidates, key=_cudart_soname_sort_key):
            yield str(candidate)


def iter_cudart_candidates() -> Iterator[str]:
    """Yield libcudart names/paths to try loading, most-preferred first.

    Each item is an argument suitable for ``ctypes.CDLL`` **and** for a raw
    ``dlopen``/``LoadLibrary``: either a bare soname the system loader resolves
    (e.g. ``"libcudart.so.12"``) or an absolute path to a wheel-shipped runtime.
    The order encodes the search strategy:

    1. absolute paths to pip-wheel CUDA installs (see
       :func:`_iter_wheel_cudart_paths`), so a venv's runtime wins over a system
       one -- this matches how JAX and PyTorch load libcudart (their loaders
       ``dlopen`` the wheel copy by absolute path first, falling back to the
       system library only if no wheel is present). Agreeing with them on which
       runtime is loaded matters for a codec that hands device memory to them;
    2. ``ctypes.util.find_library`` results (search the ``ldconfig`` cache, and
       on non-glibc platforms other loader paths), unversioned name first then
       generated per-major Windows stems;
    3. bare sonames, for systems where ``find_library`` misses the versioned name
       but the loader can still resolve it (e.g. an already-loaded copy, or via
       ``LD_LIBRARY_PATH``, which ``dlopen`` honours but ``find_library`` does
       not).

    This is the public discovery surface: non-Python consumers (e.g. the
    ``tesseract_jax`` C++ FFI shim, which ``dlopen``s libcudart itself) can use
    it to locate the same runtime this module loads, so both agree on the wheel
    preference and stay forward-compatible with new CUDA majors without their own
    hardcoded soname list. Discovery only -- the caller does the actual load.

    Items are de-duplicated preserving order; existence is not guaranteed (a
    candidate may still fail to load), so callers should try each in turn.
    """
    # find_library wants a stem (not a full soname): the unversioned name first,
    # then generated per-major Windows stems.
    find_library_stems = ("cudart", *(f"cudart64_{major}" for major in _CUDART_MAJORS))
    from_find_library = (ctypes.util.find_library(stem) for stem in find_library_stems)
    candidates = chain(
        _iter_wheel_cudart_paths(),
        filter(None, from_find_library),
        _cudart_sonames(),
    )

    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def _find_cudart() -> Any:
    """Locate and load libcudart (a ``ctypes.CDLL``), or ``None`` if not found.

    Tries each candidate from :func:`iter_cudart_candidates` in order and
    returns the first that loads.
    """
    for candidate in iter_cudart_candidates():
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    return None


def _get_cudart():
    """Lazily load the CUDA runtime shared library and declare signatures."""
    global _CUDART_HANDLE
    if _CUDART_HANDLE is not None:
        return _CUDART_HANDLE

    cudart = _find_cudart()

    if cudart is None:
        raise RuntimeError(
            "Could not find CUDA runtime library (libcudart). Make sure CUDA is "
            "installed and on the loader path (set LD_LIBRARY_PATH), or install a "
            "CUDA runtime wheel (e.g. nvidia-cuda-runtime-cu13)."
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
    # pointer. This is the crux of getting IPC to work through ctypes.
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
    # Used by decode to block until a device-to-device copy completes before the
    # IPC mapping is closed (see _load_cuda_ipc_arraydict).
    cudart.cudaDeviceSynchronize.argtypes = []
    cudart.cudaDeviceSynchronize.restype = ctypes.c_int

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
    (e.g. VMM/pool-backed memory; see :func:`_stage_for_legacy_ipc`, which
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
_cudaMemcpyDeviceToHost = 2


def _stage_for_legacy_ipc(base_ptr: int, storage_size: int) -> int:
    """Copy a VMM/pool-backed allocation into a fresh ``cudaMalloc`` buffer.

    The legacy ``cudaIpcGetMemHandle`` API rejects memory that CUDA's Virtual
    Memory Management API (``cuMemCreate``/``cuMemAddressReserve``) allocated,
    which is what modern pool allocators use, including JAX/XLA's default GPU
    allocator (confirmed: ``cudaIpcGetMemHandle`` returns
    ``cudaErrorInvalidValue`` for such pointers; CuPy's and PyTorch's default
    caching allocators happen to use plain ``cudaMalloc`` pools, so they don't
    hit this).

    Rather than replicate CUDA's VMM export path (which requires transferring
    a POSIX file descriptor between processes via ``SCM_RIGHTS`` over a Unix
    domain socket, since a real fd, not just its integer value, is meaningless
    in another process's fd table), we take the simpler route of copying the data
    device-to-device into a plain ``cudaMalloc`` allocation, which *is*
    IPC-exportable via the legacy API. This costs one on-GPU copy but avoids a
    new cross-process handshake; it is still far cheaper than a host round-trip.

    Returns the device pointer of the new (caller-owned, offset-zero) buffer.
    The caller is responsible for freeing it via ``cudaFree`` once the export
    is no longer needed (see :func:`release_pinned_ipc_exports`).
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

# Keepalive registry for arrays exported via CUDA IPC by the current request.
#
# A CUDA IPC handle is only valid while the *exporting* process keeps the source
# allocation alive. If the exported array were freed the instant it is handed
# off (before the consumer opens and copies it out) a pooled allocator
# (CuPy/PyTorch) could recycle the block, so the consumer would silently read
# *wrong* data.
#
# Both sides of a request/response exchange export arrays and share this global
# registry: the server exports its outputs, and the client exports its inputs.
# Each side retains the arrays it exported until it has positive evidence the
# consumer is done borrowing them, then calls :func:`release_pinned_ipc_exports`.
# This bounds pinned GPU memory to a single request's worth of exports per side.
#
# The two sides release at different moments because the evidence arrives at
# different moments:
#
#   * Server: releases at the START of the next request. The server's outputs
#     must outlive the handler's return -- the response (carrying the IPC
#     handles) is only serialized and sent afterwards, so the client has not yet
#     copied them out. A serial client cannot issue request N+1 until it has
#     fully handled response N, so start-of-request N+1 is the first moment the
#     server knows request N's outputs are safe to reclaim.
#
#   * Client: releases at the END of the request. The server decodes the
#     client's inputs *during* request handling -- :func:`load_cuda_ipc_arraydict`
#     copies each input into server-owned memory and closes the mapping before
#     the response is sent -- so by the time the HTTP call returns (with the body
#     buffered) the inputs are provably dead and can be released immediately.
#
# Both rely on the same two assumptions:
#   1. Requests are issued *serially* (never concurrently).
#   2. The consumer copies decoded arrays into consumer-owned memory before it
#      releases the exporter's buffer (which the decode path does
#      unconditionally; see :func:`load_cuda_ipc_arraydict`).
_CUDA_IPC_EXPORT_REGISTRY: list[Any] = []

# Device pointers of VMM-fallback staging buffers (see _stage_for_legacy_ipc)
# awaiting cudaFree. Kept separate from _CUDA_IPC_EXPORT_REGISTRY (which holds
# plain pinned array references) since these need an explicit free call
# instead of just dropping a reference, but are released at the same point and
# for the same reasons.
_CUDA_IPC_STAGING_BUFFERS: list[int] = []


def release_pinned_ipc_exports() -> None:
    """Release arrays pinned for CUDA IPC export and free their staging buffers.

    Drops the keepalive references held since the last release and frees any
    VMM-fallback staging buffers. Driven by both sides of a cuda_ipc exchange
    but at different points in the request lifecycle (server: start of the next
    request; client: end of the current request); see the registry comment above
    for why each timing is safe.
    """
    _CUDA_IPC_EXPORT_REGISTRY.clear()

    staging_ptrs, _CUDA_IPC_STAGING_BUFFERS[:] = list(_CUDA_IPC_STAGING_BUFFERS), []
    if staging_ptrs:
        cudart = _get_cudart()
        for ptr in staging_ptrs:
            cudart.cudaFree(ctypes.c_void_p(ptr))


def _pin_cuda_ipc_export(arr: Any) -> None:
    """Retain a reference to a source array so its GPU memory stays valid.

    The reference is held until the exporting side calls
    :func:`release_pinned_ipc_exports`.
    """
    _CUDA_IPC_EXPORT_REGISTRY.append(arr)


def _pin_cuda_ipc_staging_buffer(device_ptr: int) -> None:
    """Register a VMM-fallback staging buffer (see :func:`_stage_for_legacy_ipc`) for cudaFree.

    Freed at the same point ordinary pinned arrays are released (when the
    exporting side calls :func:`release_pinned_ipc_exports`), for the same
    reasons (see the module comment above).
    """
    _CUDA_IPC_STAGING_BUFFERS.append(device_ptr)


def dump_cuda_ipc_arraydict(arr: Any) -> ArrayDict:
    """Dump a CUDA array to a JSON dict with a CUDA IPC handle.

    Works with any object that implements __cuda_array_interface__:
    PyTorch tensors, CuPy arrays, JAX DeviceArrays, etc.

    The IPC handle allows another process on the same host (with --ipc=host)
    to access the GPU memory directly without any CPU round-trip.

    The source array is pinned in a process-global registry (see
    :func:`_pin_cuda_ipc_export`) so its GPU memory is not freed or recycled
    before the consumer copies it out; the pin is released once the exporting
    side calls :func:`release_pinned_ipc_exports`.

    Frameworks with VMM/pool-backed GPU allocators (e.g. JAX/XLA) hand out
    pointers that the legacy ``cudaIpcGetMemHandle`` API rejects; in that case
    this transparently falls back to staging the array's bytes into a fresh
    ``cudaMalloc`` buffer via one on-GPU copy (see :func:`_stage_for_legacy_ipc`)
    and exports a handle to that instead. Still far cheaper than a host round-trip.
    """
    if not has_cuda_array_interface(arr):
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
        # Legacy IPC rejected this pointer, almost certainly because it's
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

    handle_b64 = pybase64.b64encode_as_string(handle_bytes)
    return {
        "object_type": "array",
        "shape": list(shape),
        "dtype": dtype_name,
        "data": {
            "buffer": f"{device}:{handle_b64}:{storage_offset}:{storage_size}",
            "encoding": "cuda_ipc",
        },
    }


# ---------------------------------------------------------------------------
# DLPack ABI (framework-agnostic zero-copy exchange)
# ---------------------------------------------------------------------------
#
# The decode path returns an object (:class:`IpcDeviceArray`) that owns a
# device buffer and exposes it via both ``__cuda_array_interface__`` and
# DLPack, so Torch/JAX/CuPy can all adopt it zero-copy without CuPy being a
# decode-time dependency. The structs below mirror the DLPack C ABI closely
# enough for those consumers.

_kDLCUDA = 2  # DLDeviceType for CUDA global memory


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


# void (*)(struct DLManagedTensor *self)
_DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DLManagedTensorDeleter),
    ]


# DLDataTypeCode values (kDLInt, kDLUInt, kDLFloat, ..., kDLBool, kDLComplex).
_DLPACK_TYPE_CODES = {
    "i": 0,  # kDLInt
    "u": 1,  # kDLUInt
    "f": 2,  # kDLFloat
    "b": 6,  # kDLBool
    "c": 5,  # kDLComplex
}

# Keep PyCapsule_* usable from ctypes for the DLPack capsule handshake.
_ctypes_pythonapi = ctypes.pythonapi
_ctypes_pythonapi.PyCapsule_New.restype = ctypes.py_object
_ctypes_pythonapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
]
_ctypes_pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
_ctypes_pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_ctypes_pythonapi.PyCapsule_SetName.restype = ctypes.c_int
_ctypes_pythonapi.PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]
_ctypes_pythonapi.PyCapsule_IsValid.restype = ctypes.c_int
_ctypes_pythonapi.PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]


def _dlpack_dtype(dtype: np.dtype) -> _DLDataType:
    """Map a NumPy dtype to a DLPack ``DLDataType`` (code/bits/lanes)."""
    code = _DLPACK_TYPE_CODES.get(dtype.kind)
    if code is None:
        raise TypeError(f"dtype {dtype!r} has no DLPack type code")
    return _DLDataType(code=code, bits=dtype.itemsize * 8, lanes=1)


def _finalize_ipc_device_array(state: dict) -> None:
    """Release an :class:`IpcDeviceArray`'s device buffer exactly once.

    Registered via :func:`weakref.finalize`, so it runs when the array is
    garbage-collected *and* at interpreter shutdown, and can fire at most once.
    ``state`` is the array's mutable ownership record, shared by reference with
    the live object so ``__dlpack__`` can hand ownership off before this runs:

    * ``dlpack_token is None`` and not ``freed``: we still own the buffer, so
      free it.
    * ``dlpack_token`` set: ``__dlpack__`` moved the buffer into a DLPack bundle;
      drop the bundle iff its capsule was never consumed (a consumer that took
      the capsule already owns the free).
    """
    try:
        if state["dlpack_token"] is None:
            if not state["freed"]:
                _cuda_free(state["ptr"])
                state["freed"] = True
        else:
            _drop_unconsumed_dlpack_bundle(state["dlpack_token"])
    except Exception:
        # Finalizers must never raise.
        pass


class IpcDeviceArray:
    """Owns a device buffer decoded from a CUDA IPC handle.

    The buffer is a fresh, process-owned ``cudaMalloc`` allocation holding the
    array's own bytes (the producer's IPC mapping is copied into it and then
    closed by the decoder). The object is framework-agnostic:

    * ``__cuda_array_interface__`` (v3) lets CuPy / Numba / PyTorch adopt it,
    * ``__dlpack__`` / ``__dlpack_device__`` let Torch and JAX adopt it,

    both zero-copy. ``.copy_to_host()`` / ``np.asarray(...)`` materialise a host
    NumPy copy so it can be inspected without any GPU framework installed.

    Ownership of the device buffer is released exactly once: either a
    :func:`weakref.finalize` callback frees it (see
    :func:`_finalize_ipc_device_array`), or a DLPack consumer takes it (the
    capsule is renamed to ``"used_dltensor"`` on consumption, transferring the
    free to the consumer's deleter). ``_state["freed"]`` guards against a double
    free.
    """

    def __init__(
        self, ptr: int, device: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> None:
        self._ptr = ptr
        self.device = device
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self._nbytes = (
            int(np.prod(self.shape)) * self.dtype.itemsize
            if self.shape
            else self.dtype.itemsize
        )
        # Ownership record shared by reference with the finalizer below.
        #   freed:        True once the buffer is freed or ownership was handed
        #                 to a DLPack capsule; prevents a double free.
        #   dlpack_token: token of the DLPack bundle produced by __dlpack__ (see
        #                 _DLPACK_BUNDLES), or None if __dlpack__ was never
        #                 called. Ownership of the buffer moves into that bundle
        #                 when it is created.
        self._state: dict = {"ptr": ptr, "freed": False, "dlpack_token": None}
        self._finalizer = weakref.finalize(
            self, _finalize_ipc_device_array, self._state
        )

    # -- inspection ------------------------------------------------------

    @property
    def nbytes(self) -> int:
        """Size of the owned device buffer in bytes."""
        return self._nbytes

    @property
    def __cuda_array_interface__(self) -> dict:
        return {
            "shape": self.shape,
            "typestr": self.dtype.str,
            "data": (self._ptr, False),  # read-write
            "strides": None,  # C-contiguous
            "version": 3,
        }

    def copy_to_host(self) -> np.ndarray:
        """Copy the owned device buffer into a fresh host NumPy array."""
        if self._state["freed"]:
            raise RuntimeError("device buffer has been released")
        cudart = _get_cudart()
        host = np.empty(self.shape, dtype=self.dtype)
        ret = cudart.cudaMemcpy(
            host.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(self._ptr),
            ctypes.c_size_t(self._nbytes),
            ctypes.c_int(_cudaMemcpyDeviceToHost),
        )
        if ret != 0:
            raise RuntimeError(
                f"cudaMemcpy (device->host) failed: {_cuda_error_string(cudart, ret)}"
            )
        ret = cudart.cudaDeviceSynchronize()
        if ret != 0:
            raise RuntimeError(
                f"cudaDeviceSynchronize failed: {_cuda_error_string(cudart, ret)}"
            )
        return host

    def __array__(self, dtype: Any = None) -> np.ndarray:
        host = self.copy_to_host()
        return host if dtype is None else host.astype(dtype)

    # -- DLPack ----------------------------------------------------------

    def __dlpack_device__(self) -> tuple[int, int]:
        return (_kDLCUDA, self.device)

    def __dlpack__(self, stream: Any = None, **kwargs: Any) -> Any:
        """Return a ``"dltensor"`` PyCapsule wrapping the owned buffer.

        Ownership of the device buffer moves into a self-contained DLPack bundle
        (see :func:`_make_dlpack_bundle`) whose deleter frees it. The bundle's
        lifetime is deliberately *not* tied to this object's, because a consumer
        (Torch/JAX) may keep the tensor long after this ``IpcDeviceArray`` is
        gone and will call the deleter then. Whoever ends up owning the capsule
        (the consumer, or the finalizer for an un-consumed capsule) frees the
        buffer exactly once.
        """
        if self._state["freed"]:
            raise RuntimeError("device buffer has been released")

        capsule, token = _make_dlpack_bundle(
            self._ptr, self.device, self.shape, self.dtype
        )
        # The buffer now belongs to the bundle; this object must not free it.
        # The finalizer reads this shared state to drop the bundle iff its
        # capsule is never consumed.
        self._state["dlpack_token"] = token
        self._state["freed"] = True
        return capsule


def _cuda_free(ptr: int) -> None:
    """Free a device buffer allocated with ``cudaMalloc`` (best effort)."""
    if not ptr:
        return
    cudart = _get_cudart()
    cudart.cudaFree(ctypes.c_void_p(ptr))


# ---------------------------------------------------------------------------
# DLPack bundle registry
# ---------------------------------------------------------------------------
#
# A DLPack capsule must outlive the object that produced it: the consumer may
# hold the borrowed tensor arbitrarily long and only calls the deleter when it
# is done. We therefore keep each capsule's backing ctypes state (the
# DLManagedTensor, the shape array, the CFUNCTYPE deleter trampoline) alive in a
# process-global registry keyed by an integer token, rather than on the
# producing IpcDeviceArray. The deleter removes its own entry when invoked, so
# the state is reclaimed exactly when the consumer releases the tensor.

_DLPACK_BUNDLES: dict[int, Any] = {}
_DLPACK_NEXT_TOKEN = 0


def _make_dlpack_bundle(
    ptr: int, device: int, shape: tuple[int, ...], dtype: np.dtype
) -> tuple[Any, int]:
    """Build a ``"dltensor"`` capsule that owns ``ptr`` and register its state.

    Returns ``(capsule, token)``. The buffer is freed exactly once, by the
    deleter, whether the capsule is consumed by a framework or dropped
    un-consumed via :func:`_drop_unconsumed_dlpack_bundle`.
    """
    global _DLPACK_NEXT_TOKEN
    token = _DLPACK_NEXT_TOKEN
    _DLPACK_NEXT_TOKEN += 1

    shape_arr = (ctypes.c_int64 * len(shape))(*shape)

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.dl_tensor.device = _DLDevice(device_type=_kDLCUDA, device_id=device)
    managed.dl_tensor.ndim = len(shape)
    managed.dl_tensor.dtype = _dlpack_dtype(dtype)
    managed.dl_tensor.shape = shape_arr
    managed.dl_tensor.strides = ctypes.cast(None, ctypes.POINTER(ctypes.c_int64))
    managed.dl_tensor.byte_offset = 0

    def _deleter(_managed_ptr: int) -> None:
        # Runs when the consumer releases the tensor. Free the buffer and drop
        # our registry entry so the ctypes state can be reclaimed. Guard against
        # a second invocation (bundle already gone).
        bundle = _DLPACK_BUNDLES.pop(token, None)
        if bundle is not None:
            _cuda_free(ptr)

    c_deleter = _DLManagedTensorDeleter(_deleter)
    managed.deleter = c_deleter
    managed.manager_ctx = None

    capsule = _ctypes_pythonapi.PyCapsule_New(ctypes.byref(managed), b"dltensor", None)

    # Keep every object the capsule/consumer may still touch alive until the
    # deleter drops the entry.
    _DLPACK_BUNDLES[token] = (managed, shape_arr, c_deleter, capsule)
    return capsule, token


def _drop_unconsumed_dlpack_bundle(token: int) -> None:
    """Free a bundle's buffer iff its capsule was never consumed.

    Called from the :class:`IpcDeviceArray` finalizer. If the capsule is still
    named ``"dltensor"`` no framework adopted it, so we invoke the deleter to
    free the buffer. If it was renamed to ``"used_dltensor"`` a consumer owns it
    and will (or already did) free it via the deleter, so we leave it alone.
    """
    bundle = _DLPACK_BUNDLES.get(token)
    if bundle is None:
        return
    _managed, _shape_arr, c_deleter, capsule = bundle
    still_dltensor = bool(_ctypes_pythonapi.PyCapsule_IsValid(capsule, b"dltensor"))
    if still_dltensor:
        # Nobody adopted it -> free now (the deleter pops the registry entry).
        c_deleter(0)


def load_cuda_ipc_arraydict(val: ArrayDict) -> "IpcDeviceArray":
    """Load a CUDA array from a JSON dict with a CUDA IPC handle.

    The calling process must share the IPC namespace with the producer
    (e.g. both run with --ipc=host on Docker) and see the same GPU.

    Returns a freshly-allocated, caller-owned :class:`IpcDeviceArray`: the IPC
    handle is opened, the array's own bytes are copied device-to-device into a
    ``cudaMalloc`` buffer owned by this process, the copy is synchronised, and
    the IPC mapping is closed before returning. The borrow of the producer's
    memory therefore lasts only for a single on-GPU copy, so the producer is
    free to reuse or release the exported buffer as soon as this call returns.

    The result carries no framework dependency: it exposes both
    ``__cuda_array_interface__`` and ``__dlpack__`` so Torch/JAX/CuPy can adopt
    it zero-copy, plus ``.copy_to_host()`` / ``np.asarray(...)`` for inspection.
    """
    cudart = _get_cudart()

    device_str, handle_b64, storage_offset_str, _storage_size_str = val["data"][
        "buffer"
    ].split(":")
    handle_bytes = pybase64.b64decode(handle_b64, validate=True)
    device = int(device_str)
    storage_offset = int(storage_offset_str)

    dtype = np.dtype(val["dtype"])
    shape = tuple(val["shape"])
    nbytes = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize

    # Allocate the owned buffer up front (on the target device) so that if the
    # copy fails we still close the IPC mapping and free the buffer cleanly.
    ret = cudart.cudaSetDevice(device)
    if ret != 0:
        raise RuntimeError(
            f"cudaSetDevice({device}) failed: {_cuda_error_string(cudart, ret)}"
        )
    owned_ptr = ctypes.c_void_p()
    ret = cudart.cudaMalloc(ctypes.byref(owned_ptr), ctypes.c_size_t(nbytes))
    if ret != 0:
        raise RuntimeError(f"cudaMalloc failed: {_cuda_error_string(cudart, ret)}")

    base_ptr = _cuda_ipc_open_mem_handle(handle_bytes, device)
    try:
        # Copy only this array's own bytes out of the producer's (offset)
        # mapping into our fresh buffer, then block until the copy is done so we
        # never unmap mid-copy.
        ret = cudart.cudaMemcpy(
            owned_ptr,
            ctypes.c_void_p(base_ptr + storage_offset),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(_cudaMemcpyDeviceToDevice),
        )
        if ret != 0:
            raise RuntimeError(
                f"cudaMemcpy (device->device) failed: {_cuda_error_string(cudart, ret)}"
            )
        ret = cudart.cudaDeviceSynchronize()
        if ret != 0:
            raise RuntimeError(
                f"cudaDeviceSynchronize failed: {_cuda_error_string(cudart, ret)}"
            )
    except Exception:
        cudart.cudaFree(owned_ptr)
        raise
    finally:
        _cuda_ipc_close_mem_handle(base_ptr)

    return IpcDeviceArray(owned_ptr.value, device, shape, dtype)


def validate_cuda_array(
    val: Any, expected_shape: ShapeType, expected_dtype: str | None
) -> Any:
    """Validate a GPU array's shape/dtype without pulling it off the device.

    Returns the object unchanged so it can later be encoded via CUDA IPC (see
    :func:`tesseract_core.runtime.array_encoding.encode_array`). Only the
    ``__cuda_array_interface__`` metadata is inspected -- no device-to-host copy
    or kernel launch occurs. Mirrors the shape/dtype checks in
    :func:`tesseract_core.runtime.array_encoding._coerce_shape_dtype`, but never
    casts (a cast would need a device copy the caller did not ask for).
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


# ---------------------------------------------------------------------------
# DeviceTransport backend
# ---------------------------------------------------------------------------
#
# The functions above are the cuda_ipc encode/decode/release machinery. The
# thin adapter below exposes them through the shared DeviceTransport interface
# so cuda_ipc is one registered transport among (eventually) several, driven by
# the same lifecycle rather than a bespoke branch in the encode/decode dispatch.
# It adds no behavior: each method delegates to the corresponding function, and
# cuda_ipc's inert handle needs no bootstrap and no flush (the consumer pulls).


class CudaIpcTransport:
    """DeviceTransport backend for the legacy same-host ``json+cuda_ipc`` mode."""

    name = "cuda_ipc"
    reach = "same_host"

    def bootstrap(self, role: Any, peer_offer: Any) -> None:
        """No-op: the IPC handle is self-contained, so no shared state to set up."""

    def register(self, arr: Any, session: Any = None) -> ArrayDict:
        """Pin ``arr`` and build its IPC descriptor (the finished array dict).

        cuda_ipc mints the handle and packs the wire string in one call, so the
        per-array handle *is* the array dict and :meth:`descriptor` is a
        passthrough. Splitting them would take the IPC handle twice for nothing.
        """
        return dump_cuda_ipc_arraydict(arr)

    def descriptor(self, handle: ArrayDict) -> ArrayDict:
        """Return the array dict :meth:`register` already produced."""
        return handle

    def flush(self, session: Any = None) -> None:
        """No-op: cuda_ipc is receiver-driven, so there is nothing to post."""

    def receive(self, val: ArrayDict, session: Any = None) -> "IpcDeviceArray":
        """Copy the exported bytes into a fresh consumer-owned ``IpcDeviceArray``."""
        return load_cuda_ipc_arraydict(val)

    def release(self, session: Any = None) -> None:
        """Drop the producer-side pins from this request's exports."""
        release_pinned_ipc_exports()


def _register_cuda_ipc_transport() -> None:
    """Register the cuda_ipc backend once this module is imported."""
    from tesseract_core.runtime.device_transport import register_transport

    register_transport(CudaIpcTransport())


_register_cuda_ipc_transport()
