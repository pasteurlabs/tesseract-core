# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA VMM POSIX-fd sharing: the copy-free path behind ``json+cuda_ipc``.

Legacy CUDA IPC (``cudaIpcGetMemHandle``) rejects memory allocated through the
CUDA Virtual Memory Management API (``cuMemCreate``) -- which is what modern
pooled allocators use, notably JAX/XLA's default GPU allocator and PyTorch's
``expandable_segments``. For those, :mod:`tesseract_core.runtime.cuda_ipc` falls
back to :func:`~tesseract_core.runtime.cuda_ipc._stage_for_legacy_ipc`: an extra
device-to-device copy into a fresh ``cudaMalloc`` buffer that legacy IPC *can*
export. This module removes that copy for VMM-backed memory by exporting the VMM
allocation *by reference* instead.

The mechanics differ from legacy IPC in two ways that shape the code:

* **The handle is a POSIX file descriptor**, not a serializable blob, so it
  cannot ride in the JSON response. It is passed out-of-band over a Unix domain
  socket via ``SCM_RIGHTS``. The producer runs a tiny fd-passing server; the
  JSON descriptor carries the socket path and an export id, and the consumer
  connects to fetch the fd. (Verified to work across a container->host boundary
  with ``--ipc=host`` plus a shared mount, i.e. the served-Tesseract deployment.)

* **The export carries no cross-process ordering guarantee**, so the producer
  must ``cuCtxSynchronize`` after any pending writes before handing off the fd.

This path is reached *through* ``json+cuda_ipc`` -- :mod:`cuda_ipc` selects it
automatically when the source memory is VMM-exportable and falls back to the
legacy/staging path otherwise -- so there is no separate user-facing format.
"""

from __future__ import annotations

import ctypes
import os
import socket
import threading
from typing import Any

import numpy as np

from tesseract_core.runtime.array_encoding import ArrayDict
from tesseract_core.runtime.cuda_ipc import (
    IpcDeviceArray,
    _cuda_error_string,
    _get_cudart,
    _is_c_contiguous,
    has_cuda_array_interface,
)

# ---------------------------------------------------------------------------
# CUDA driver bindings for the VMM API (via libcuda)
# ---------------------------------------------------------------------------

_CU: Any = None

# CUmemAllocationHandleType
_CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
# CUmemLocationType
_CU_MEM_LOCATION_TYPE_DEVICE = 1
# CUmemAccess_flags
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


def _get_cuda_driver() -> Any:
    """Lazily load libcuda and declare the VMM signatures used here."""
    global _CU
    if _CU is not None:
        return _CU
    import ctypes.util

    lib = None
    path = ctypes.util.find_library("cuda")
    if path:
        lib = ctypes.CDLL(path)
    else:
        for name in ("libcuda.so", "libcuda.so.1"):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
    if lib is None:
        raise RuntimeError("Could not find the CUDA driver library (libcuda).")

    lib.cuInit(0)
    P = ctypes.POINTER
    # Recover the VMM allocation handle backing a device pointer.
    lib.cuMemRetainAllocationHandle.argtypes = [P(ctypes.c_ulonglong), ctypes.c_void_p]
    lib.cuMemRetainAllocationHandle.restype = ctypes.c_int
    # Export it to a shareable POSIX fd.
    lib.cuMemExportToShareableHandle.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulonglong,
        ctypes.c_int,
        ctypes.c_ulonglong,
    ]
    lib.cuMemExportToShareableHandle.restype = ctypes.c_int
    lib.cuMemRelease.argtypes = [ctypes.c_ulonglong]
    lib.cuMemRelease.restype = ctypes.c_int
    # Import + map on the consumer.
    lib.cuMemImportFromShareableHandle.argtypes = [
        P(ctypes.c_ulonglong),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.cuMemImportFromShareableHandle.restype = ctypes.c_int
    lib.cuMemAddressReserve.argtypes = [
        P(ctypes.c_ulonglong),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_ulonglong,
        ctypes.c_ulonglong,
    ]
    lib.cuMemAddressReserve.restype = ctypes.c_int
    lib.cuMemMap.argtypes = [
        ctypes.c_ulonglong,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_ulonglong,
        ctypes.c_ulonglong,
    ]
    lib.cuMemMap.restype = ctypes.c_int
    lib.cuMemUnmap.argtypes = [ctypes.c_ulonglong, ctypes.c_size_t]
    lib.cuMemUnmap.restype = ctypes.c_int
    lib.cuMemSetAccess.argtypes = [
        ctypes.c_ulonglong,
        ctypes.c_size_t,
        P(_CUmemAccessDesc),
        ctypes.c_size_t,
    ]
    lib.cuMemSetAccess.restype = ctypes.c_int
    lib.cuMemAddressFree.argtypes = [ctypes.c_ulonglong, ctypes.c_size_t]
    lib.cuMemAddressFree.restype = ctypes.c_int
    lib.cuMemGetAddressRange_v2.argtypes = [
        P(ctypes.c_ulonglong),
        P(ctypes.c_size_t),
        ctypes.c_ulonglong,
    ]
    lib.cuMemGetAddressRange_v2.restype = ctypes.c_int
    lib.cuCtxSynchronize.argtypes = []
    lib.cuCtxSynchronize.restype = ctypes.c_int
    lib.cuDeviceGet.argtypes = [P(ctypes.c_int), ctypes.c_int]
    lib.cuDeviceGet.restype = ctypes.c_int

    _CU = lib
    return _CU


def _cu_check(ret: int, what: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{what} failed: CUresult={ret}")


def is_vmm_exportable(data_ptr: int) -> bool:
    """Whether ``data_ptr`` is backed by a VMM allocation we can export by fd.

    ``cuMemRetainAllocationHandle`` succeeds only for memory allocated via the
    VMM API (``cuMemCreate``) -- JAX/XLA's allocator, PyTorch
    ``expandable_segments``, or our own :func:`cuMemCreate` buffers. Default
    CuPy/PyTorch pools and legacy ``cudaMalloc`` return an error, so the caller
    keeps the legacy IPC / staging path for those.
    Returns ``False`` (rather than raising) if the CUDA driver cannot even be
    loaded -- e.g. a GPU-less host running the mocked encode path -- so the caller
    transparently keeps the legacy path there too.
    """
    try:
        driver = _get_cuda_driver()
    except RuntimeError:
        return False
    handle = ctypes.c_ulonglong()
    ret = driver.cuMemRetainAllocationHandle(
        ctypes.byref(handle), ctypes.c_void_p(data_ptr)
    )
    if ret != 0:
        return False
    driver.cuMemRelease(handle)
    return True


# ---------------------------------------------------------------------------
# Producer side: fd-passing server + export registry
# ---------------------------------------------------------------------------
#
# A POSIX fd is only meaningful once passed to another process via SCM_RIGHTS, so
# the producer runs a small Unix-socket server that hands out the fd for an
# export id on request. The socket path travels in the JSON descriptor. The
# server is process-global and started lazily on the first VMM export.

_FD_SERVER: _FdPassServer | None = None
# Export id -> (retained VMM handle, keepalive array). Cleared on release.
_VMM_EXPORT_REGISTRY: dict[int, tuple[int, Any]] = {}
_NEXT_EXPORT_ID = 0
_EXPORT_LOCK = threading.Lock()


class _FdPassServer:
    """A Unix-socket server that hands out an exported VMM fd per export id.

    One connection per fetch: the consumer sends the 8-byte export id, the
    server exports the retained handle to a fresh fd and passes it back via
    ``SCM_RIGHTS``. Exporting per fetch (rather than caching the fd) keeps the
    server stateless beyond the retained handle and avoids fd leaks.
    """

    def __init__(self) -> None:
        # The socket must live on a path the *consumer* can reach. When the
        # Tesseract is served in a container, the consumer runs on the host, so
        # the socket has to sit on the shared bind-mount -- the runtime's
        # output_path, the same directory json+binref uses to hand files across
        # the boundary. Fall back to a private tempdir for the bare same-host
        # (non-container) case where any path is reachable.
        import tempfile
        import uuid

        base_dir = _fd_socket_base_dir()
        os.makedirs(base_dir, exist_ok=True)
        self._dir = tempfile.mkdtemp(prefix="tsr-vmm-", dir=base_dir)
        # AF_UNIX paths are capped at ~108 bytes; keep the socket name short.
        self.path = os.path.join(self._dir, f"{uuid.uuid4().hex[:8]}.sock")
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.path)
        # The consumer may run under a different uid than this server -- notably
        # a host client connecting to a containerized server (often root). Make
        # the socket (and its dir) connectable regardless of uid; the data it
        # gates (a GPU fd) is already reachable to anyone sharing --ipc=host.
        try:
            os.chmod(self._dir, 0o777)
            os.chmod(self.path, 0o777)
        except OSError:
            pass
        self._sock.listen(64)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._running = True
        self._thread.start()

    def _serve(self) -> None:
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            try:
                self._handle(conn)
            except Exception:
                pass
            finally:
                conn.close()

    def _handle(self, conn: socket.socket) -> None:
        raw = conn.recv(8)
        if len(raw) != 8:
            return
        export_id = int.from_bytes(raw, "little")
        with _EXPORT_LOCK:
            entry = _VMM_EXPORT_REGISTRY.get(export_id)
        if entry is None:
            conn.sendmsg([b"\x00"])  # miss: no ancillary fd
            return
        handle, _arr = entry
        driver = _get_cuda_driver()
        fd = ctypes.c_int()
        ret = driver.cuMemExportToShareableHandle(
            ctypes.byref(fd),
            ctypes.c_ulonglong(handle),
            _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0,
        )
        if ret != 0:
            conn.sendmsg([b"\x00"])
            return
        try:
            conn.sendmsg(
                [b"\x01"],
                [
                    (
                        socket.SOL_SOCKET,
                        socket.SCM_RIGHTS,
                        int(fd.value).to_bytes(4, "little"),
                    )
                ],
            )
        finally:
            os.close(fd.value)

    def close(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass


def _fd_socket_base_dir() -> str:
    """Directory to bind the fd-passing socket under.

    Prefers the runtime's ``output_path`` (the host<->container shared mount, so
    a host consumer can reach the socket a containerized server binds), then an
    explicit ``TESSERACT_VMM_SOCKET_DIR`` override, else the system temp dir for
    the bare same-host case. Never raises: falls back to temp on any error.
    """
    override = os.environ.get("TESSERACT_VMM_SOCKET_DIR")
    if override:
        return override
    try:
        from tesseract_core.runtime.config import get_config

        output_path = get_config().output_path
        if output_path and output_path != ".":
            return output_path
    except Exception:
        pass
    import tempfile

    return tempfile.gettempdir()


def _get_fd_server() -> _FdPassServer:
    global _FD_SERVER
    if _FD_SERVER is None:
        _FD_SERVER = _FdPassServer()
    return _FD_SERVER


def release_vmm_exports() -> None:
    """Release VMM allocation handles retained for this request's exports.

    Mirrors :func:`tesseract_core.runtime.cuda_ipc.release_pinned_ipc_exports`.
    The fd-passing server itself is process-global and left running.
    """
    driver = _get_cuda_driver()
    with _EXPORT_LOCK:
        entries = list(_VMM_EXPORT_REGISTRY.items())
        _VMM_EXPORT_REGISTRY.clear()
    for _export_id, (handle, _arr) in entries:
        driver.cuMemRelease(ctypes.c_ulonglong(handle))


def dump_vmm_arraydict(arr: Any) -> ArrayDict:
    """Export a VMM-backed CUDA array by fd and return a ``cuda_ipc`` descriptor.

    The descriptor's ``buffer`` uses the VMM variant form
    ``vmm:{sockpath_b64}:{export_id}:{storage_offset}:{storage_size}:{device}``.
    Requires :func:`is_vmm_exportable` to be true for the array's pointer; the
    caller checks that before routing here.
    """
    if not has_cuda_array_interface(arr):
        raise ValueError("vmm encoding requires a CUDA array")
    if not _is_c_contiguous(arr):
        raise ValueError("vmm encoding requires a C-contiguous array")

    driver = _get_cuda_driver()
    iface = arr.__cuda_array_interface__
    data_ptr = iface["data"][0]
    shape = tuple(iface["shape"])
    dtype = np.dtype(iface["typestr"])
    device = _device_ordinal(arr)

    # Retain the VMM handle backing this pointer, and record the byte offset of
    # the array within the whole mapped allocation (pooled VMM allocators hand
    # out many arrays from one reservation).
    base = ctypes.c_ulonglong()
    size = ctypes.c_size_t()
    _cu_check(
        driver.cuMemGetAddressRange_v2(
            ctypes.byref(base), ctypes.byref(size), ctypes.c_ulonglong(data_ptr)
        ),
        "cuMemGetAddressRange",
    )
    storage_offset = data_ptr - base.value
    storage_size = size.value

    handle = ctypes.c_ulonglong()
    _cu_check(
        driver.cuMemRetainAllocationHandle(
            ctypes.byref(handle), ctypes.c_void_p(base.value)
        ),
        "cuMemRetainAllocationHandle",
    )

    # The export carries no ordering guarantee: make sure the producer's writes
    # to this memory are complete before a consumer can map and read it.
    _cu_check(driver.cuCtxSynchronize(), "cuCtxSynchronize")

    global _NEXT_EXPORT_ID
    with _EXPORT_LOCK:
        export_id = _NEXT_EXPORT_ID
        _NEXT_EXPORT_ID += 1
        _VMM_EXPORT_REGISTRY[export_id] = (handle.value, arr)

    server = _get_fd_server()
    import pybase64

    sock_b64 = pybase64.b64encode_as_string(server.path.encode())
    return {
        "object_type": "array",
        "shape": list(shape),
        "dtype": dtype.name,
        "data": {
            "buffer": (
                f"vmm:{sock_b64}:{export_id}:{storage_offset}:{storage_size}:{device}"
            ),
            "encoding": "cuda_ipc",
        },
    }


def _device_ordinal(arr: Any) -> int:
    device = 0
    if hasattr(arr, "device"):
        dev = arr.device
        if hasattr(dev, "id"):
            device = dev.id
        elif hasattr(dev, "index") and dev.index is not None:
            device = dev.index
    return device


# ---------------------------------------------------------------------------
# Consumer side: fetch the fd, import + map, copy into an owned buffer
# ---------------------------------------------------------------------------


def _fetch_fd(sock_path: str, export_id: int) -> int:
    """Fetch the exported fd for ``export_id`` from the producer's fd server."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(sock_path)
        s.sendall(export_id.to_bytes(8, "little"))
        msg, ancdata, _flags, _addr = s.recvmsg(1, socket.CMSG_LEN(4))
        if not msg or msg[0] != 1:
            raise RuntimeError(f"vmm fd server returned no fd for export {export_id}")
        for level, ctype, data in ancdata:
            if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                return int.from_bytes(data[:4], "little")
        raise RuntimeError("vmm fd server sent no SCM_RIGHTS fd")
    finally:
        s.close()


def load_vmm_arraydict(val: ArrayDict) -> IpcDeviceArray:
    """Decode a VMM ``cuda_ipc`` descriptor: map the producer's memory, copy out.

    Imports the producer's VMM allocation (via the fd fetched over the socket),
    maps it, copies just this array's own bytes into a fresh ``cudaMalloc``
    buffer owned by this process, unmaps, and returns an :class:`IpcDeviceArray`
    -- the same consumer-facing wrapper as legacy ``cuda_ipc``. The borrow of the
    producer's memory lasts only for the copy.
    """
    import pybase64

    _tag, sock_b64, export_id_s, offset_s, size_s, device_s = val["data"][
        "buffer"
    ].split(":")
    sock_path = pybase64.b64decode(sock_b64, validate=True).decode()
    export_id = int(export_id_s)
    storage_offset = int(offset_s)
    storage_size = int(size_s)
    device = int(device_s)

    dtype = np.dtype(val["dtype"])
    shape = tuple(val["shape"])
    nbytes = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize

    driver = _get_cuda_driver()
    cudart = _get_cudart()

    ret = cudart.cudaSetDevice(device)
    if ret != 0:
        raise RuntimeError(
            f"cudaSetDevice({device}) failed: {_cuda_error_string(cudart, ret)}"
        )
    owned_ptr = ctypes.c_void_p()
    ret = cudart.cudaMalloc(ctypes.byref(owned_ptr), ctypes.c_size_t(nbytes))
    if ret != 0:
        raise RuntimeError(f"cudaMalloc failed: {_cuda_error_string(cudart, ret)}")

    fd = _fetch_fd(sock_path, export_id)
    handle = ctypes.c_ulonglong()
    mapped_ptr = ctypes.c_ulonglong()
    mapped = False
    try:
        _cu_check(
            driver.cuMemImportFromShareableHandle(
                ctypes.byref(handle),
                ctypes.cast(fd, ctypes.c_void_p),
                _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            ),
            "cuMemImportFromShareableHandle",
        )
        _cu_check(
            driver.cuMemAddressReserve(ctypes.byref(mapped_ptr), storage_size, 0, 0, 0),
            "cuMemAddressReserve",
        )
        _cu_check(driver.cuMemMap(mapped_ptr, storage_size, 0, handle, 0), "cuMemMap")
        mapped = True
        acc = _CUmemAccessDesc()
        acc.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
        acc.location.id = device
        acc.flags = _CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _cu_check(
            driver.cuMemSetAccess(mapped_ptr, storage_size, ctypes.byref(acc), 1),
            "cuMemSetAccess",
        )

        # Copy just this array's bytes (at its offset) into our owned buffer.
        ret = cudart.cudaMemcpy(
            owned_ptr,
            ctypes.c_void_p(mapped_ptr.value + storage_offset),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(3),  # cudaMemcpyDeviceToDevice
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
        if mapped:
            driver.cuMemUnmap(mapped_ptr, storage_size)
            driver.cuMemAddressFree(mapped_ptr, storage_size)
        if handle.value:
            driver.cuMemRelease(handle)
        # The imported fd is dup'd into our process; close our copy.
        try:
            os.close(fd)
        except OSError:
            pass

    return IpcDeviceArray(owned_ptr.value, device, shape, dtype)
