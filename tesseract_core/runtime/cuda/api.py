# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain-Python wrappers around the CUDA runtime and driver APIs.

Every function here takes and returns ordinary Python values -- device pointers
are plain ``int``s, IPC handles are ``bytes``, sizes are ``int``s. No ctypes
object crosses the boundary, so callers (chiefly
:mod:`tesseract_core.runtime.cuda.ipc`) never touch ctypes. The loaded
libraries are held as module globals and initialised lazily on first use.

Errors are raised as ``RuntimeError`` with a decoded CUDA error string. To run
this without a real GPU, replace the functions in this module wholesale (see the
``mocked_cuda`` test fixture), rather than reaching into ctypes internals.
"""

import ctypes
from typing import Any

from tesseract_core.runtime.cuda import loader

# cudaMemcpyKind values.
MEMCPY_DEVICE_TO_HOST = 2
MEMCPY_DEVICE_TO_DEVICE = 3

# cudaIpcMemLazyEnablePeerAccess
_IPC_LAZY_ENABLE_PEER_ACCESS = 0x01

# Re-exported so callers that assemble/validate IPC payloads can reason about the
# handle width without importing the loader's ABI details.
IPC_HANDLE_SIZE = loader.CUDA_IPC_HANDLE_SIZE

_cudart: Any = None
_driver: Any = None


def _get_cudart() -> Any:
    """Return the loaded CUDA runtime library, loading it on first use."""
    global _cudart
    if _cudart is None:
        _cudart = loader.load_cudart()
    return _cudart


def _get_driver() -> Any:
    """Return the loaded CUDA driver library, loading it on first use."""
    global _driver
    if _driver is None:
        _driver = loader.load_cuda_driver()
    return _driver


def _error_string(code: int) -> str:
    """Best-effort human-readable CUDA error string for an error code."""
    try:
        msg = _get_cudart().cudaGetErrorString(code)
        if msg:
            return msg.decode()
    except Exception:
        pass
    return f"error code {code}"


def _check(code: int, what: str) -> None:
    """Raise ``RuntimeError`` if a CUDA call returned a non-zero error code.

    A failed runtime call also sets the CUDA runtime API's *sticky* last-error.
    Left uncleared, the next CUDA consumer in the process reads it as its own
    failure -- e.g. after ``cudaIpcGetMemHandle`` rejects VMM/pool-backed memory
    on the staging fallback path, JAX/XLA's next kernel launch aborts with
    ``cudaErrorInvalidValue`` "before calling cuModuleGetFunction". Draining the
    sticky error here keeps every expected failure contained to its own call.
    """
    if code != 0:
        _get_cudart().cudaGetLastError()
        raise RuntimeError(f"{what} failed: {_error_string(code)}")


# -- device / memory management --------------------------------------------


def set_device(device: int) -> None:
    """Select the active CUDA device."""
    _check(_get_cudart().cudaSetDevice(device), f"cudaSetDevice({device})")


def malloc(nbytes: int) -> int:
    """Allocate ``nbytes`` of device memory; return its device pointer."""
    ptr = ctypes.c_void_p()
    _check(
        _get_cudart().cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)),
        "cudaMalloc",
    )
    return ptr.value


def free(device_ptr: int) -> None:
    """Free a device buffer allocated with :func:`malloc` (best effort).

    A null/zero pointer is ignored so callers can free unconditionally.
    """
    if not device_ptr:
        return
    _get_cudart().cudaFree(ctypes.c_void_p(device_ptr))


def memcpy_device_to_device(dst: int, src: int, nbytes: int) -> None:
    """Copy ``nbytes`` between two device pointers."""
    _check(
        _get_cudart().cudaMemcpy(
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(MEMCPY_DEVICE_TO_DEVICE),
        ),
        "cudaMemcpy (device->device)",
    )


def memcpy_device_to_host(host_ptr: int, src: int, nbytes: int) -> None:
    """Copy ``nbytes`` from a device pointer into host memory at ``host_ptr``."""
    _check(
        _get_cudart().cudaMemcpy(
            ctypes.c_void_p(host_ptr),
            ctypes.c_void_p(src),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(MEMCPY_DEVICE_TO_HOST),
        ),
        "cudaMemcpy (device->host)",
    )


def device_synchronize() -> None:
    """Block until all previously issued device work has completed."""
    _check(_get_cudart().cudaDeviceSynchronize(), "cudaDeviceSynchronize")


# -- IPC -------------------------------------------------------------------


def ipc_get_mem_handle(device_ptr: int) -> bytes:
    """Return the 64-byte IPC handle for a device allocation.

    ``device_ptr`` should be the *base* of the allocation (see
    :func:`get_allocation_base`); IPC handles always reference the whole
    underlying allocation.

    Raises ``RuntimeError`` if the pointer is rejected by the legacy IPC API
    (e.g. VMM/pool-backed memory; see :func:`stage_for_legacy_ipc`, which
    callers should fall back to on failure).
    """
    cudart = _get_cudart()
    handle = loader.CudaIpcMemHandle()
    _check(
        cudart.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(device_ptr)),
        "cudaIpcGetMemHandle",
    )
    return bytes(handle.reserved)


def ipc_open_mem_handle(handle_bytes: bytes, device: int) -> int:
    """Open an IPC handle on ``device``; return the mapped base device pointer.

    The returned pointer is the base of the producer's allocation as mapped
    into this process; callers must add any per-array byte offset themselves.
    """
    cudart = _get_cudart()
    # IPC memory must be opened on the device it lives on.
    set_device(device)

    handle = loader.CudaIpcMemHandle()
    ctypes.memmove(handle.reserved, handle_bytes, IPC_HANDLE_SIZE)
    dev_ptr = ctypes.c_void_p()
    _check(
        cudart.cudaIpcOpenMemHandle(
            ctypes.byref(dev_ptr), handle, ctypes.c_uint(_IPC_LAZY_ENABLE_PEER_ACCESS)
        ),
        "cudaIpcOpenMemHandle",
    )
    if dev_ptr.value is None:
        raise RuntimeError("cudaIpcOpenMemHandle returned a null pointer")
    return dev_ptr.value


def ipc_close_mem_handle(device_ptr: int) -> None:
    """Release an IPC-opened base device pointer.

    ``device_ptr`` must be the base pointer returned by
    :func:`ipc_open_mem_handle` (not an offset pointer into it).
    """
    _check(
        _get_cudart().cudaIpcCloseMemHandle(ctypes.c_void_p(device_ptr)),
        "cudaIpcCloseMemHandle",
    )


def stage_for_legacy_ipc(src_ptr: int, nbytes: int) -> int:
    """Copy ``nbytes`` into a fresh ``cudaMalloc`` buffer IPC-exportable via the legacy API.

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
    The caller is responsible for freeing it via :func:`free` once the export
    is no longer needed.
    """
    staging_ptr = malloc(nbytes)
    try:
        memcpy_device_to_device(staging_ptr, src_ptr, nbytes)
    except Exception:
        free(staging_ptr)
        raise
    return staging_ptr


# -- driver API ------------------------------------------------------------


def get_allocation_base(device_ptr: int) -> tuple[int, int]:
    """Return ``(base_ptr, size)`` of the allocation containing ``device_ptr``.

    Uses the driver API ``cuMemGetAddressRange`` so that pointers into the
    middle of a (possibly pooled) allocation are resolved to the base pointer
    an IPC handle actually references; the byte offset can then be derived as
    ``device_ptr - base_ptr``.
    """
    driver = _get_driver()
    base = ctypes.c_ulonglong()
    size = ctypes.c_size_t()
    ret = driver.cuMemGetAddressRange_v2(
        ctypes.byref(base), ctypes.byref(size), ctypes.c_ulonglong(device_ptr)
    )
    if ret != 0:
        raise RuntimeError(f"cuMemGetAddressRange failed with error code {ret}")
    return base.value, size.value
