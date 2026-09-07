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
The VMM machinery is packaged as a :class:`~tesseract_core.runtime.device_transport.DeviceTransport`
(:class:`VmmTransport`): the fd-passing server is its session, created in
:meth:`VmmTransport.bootstrap` (owned by the served app's lifespan) and reused
across a request's exports.
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
# Producer side: fd-passing server
# ---------------------------------------------------------------------------
#
# A POSIX fd is only meaningful once passed to another process via SCM_RIGHTS, so
# the producer runs a small Unix-socket server that hands out the fd for an
# export id on request. The socket path travels in the JSON descriptor. The
# server is the transport's *session*: :meth:`VmmTransport.bootstrap` creates it
# (owned by the served app's lifespan; see ``serve.create_rest_api``), and the
# same instance serves every export until release.

# AF_UNIX message framing: an 8-byte little-endian count N, followed by N 8-byte
# little-endian export ids. The reply is N status bytes plus one SCM_RIGHTS
# control message carrying a fd for every id that hit (misses contribute a 0
# status byte and no fd). Batching lets a consumer fetch a whole apply's exports
# in one round-trip; a single-id request is just N == 1.
_REQ_COUNT_BYTES = 8
_ID_BYTES = 8


class _FdPassServer:
    """A Unix-socket server that hands out exported VMM fds by export id.

    The consumer sends a batch of export ids; the server exports each retained
    handle to a fresh fd and passes them all back in one ``SCM_RIGHTS`` message.
    Exporting per fetch (rather than caching the fd) keeps the server stateless
    beyond the retained handles and avoids fd leaks. The connection is kept open
    for reuse across requests; the server loops on it until the peer hangs up.
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

        # Export id -> (retained VMM handle, keepalive array). Cleared on release.
        self._registry: dict[int, tuple[int, Any]] = {}
        self._next_export_id = 0
        self._lock = threading.Lock()

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

    # -- export bookkeeping ---------------------------------------------------

    def register(self, handle: int, keepalive: Any) -> int:
        """Retain ``handle`` under a fresh export id and return the id."""
        with self._lock:
            export_id = self._next_export_id
            self._next_export_id += 1
            self._registry[export_id] = (handle, keepalive)
        return export_id

    def release(self) -> None:
        """Release every retained handle from this request's exports.

        The server (and its socket) stay up for the next request; only the
        retained VMM handles are dropped, bounding pinned memory to one request's
        worth of exports (mirrors ``cuda_ipc.release_pinned_ipc_exports``).
        """
        with self._lock:
            entries = list(self._registry.values())
            self._registry.clear()
        if not entries:
            # Nothing was exported via the VMM path (the usual case): don't touch
            # the driver at all, so this stays a no-op rather than failing to
            # load libcuda on a host without it.
            return
        driver = _get_cuda_driver()
        for handle, _keepalive in entries:
            driver.cuMemRelease(ctypes.c_ulonglong(handle))

    # -- socket server --------------------------------------------------------

    def _serve(self) -> None:
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            # One thread per connection so a slow/persistent consumer does not
            # block others; the connection loops until the peer hangs up.
            worker = threading.Thread(
                target=self._serve_conn, args=(conn,), daemon=True
            )
            worker.start()

    def _serve_conn(self, conn: socket.socket) -> None:
        # A stalled consumer must not tie up this worker forever.
        conn.settimeout(30.0)
        try:
            while self._running:
                if not self._handle_one(conn):
                    return
        except (TimeoutError, OSError):
            return
        finally:
            conn.close()

    def _handle_one(self, conn: socket.socket) -> bool:
        """Serve one batch request on ``conn``. Returns False when the peer is done."""
        raw = _recv_exactly(conn, _REQ_COUNT_BYTES)
        if raw is None:
            return False  # clean hang-up between requests
        count = int.from_bytes(raw, "little")
        if count <= 0 or count > 4096:
            return False  # framing error / absurd count: drop the connection
        ids_raw = _recv_exactly(conn, count * _ID_BYTES)
        if ids_raw is None:
            return False
        export_ids = [
            int.from_bytes(ids_raw[i * _ID_BYTES : (i + 1) * _ID_BYTES], "little")
            for i in range(count)
        ]

        statuses = bytearray(count)
        fds: list[int] = []
        driver = _get_cuda_driver()
        try:
            for pos, export_id in enumerate(export_ids):
                with self._lock:
                    entry = self._registry.get(export_id)
                if entry is None:
                    continue  # miss: status stays 0, no fd
                handle, _keepalive = entry
                fd = ctypes.c_int()
                ret = driver.cuMemExportToShareableHandle(
                    ctypes.byref(fd),
                    ctypes.c_ulonglong(handle),
                    _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                    0,
                )
                if ret != 0:
                    continue
                statuses[pos] = 1
                fds.append(fd.value)
            ancdata = (
                [
                    (
                        socket.SOL_SOCKET,
                        socket.SCM_RIGHTS,
                        b"".join(fd.to_bytes(4, "little") for fd in fds),
                    )
                ]
                if fds
                else []
            )
            conn.sendmsg([bytes(statuses)], ancdata)
        finally:
            # The fds are dup'd into the peer by sendmsg; close our copies.
            for fd in fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
        return True

    def close(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass


def _recv_exactly(conn: socket.socket, n: int) -> bytes | None:
    """Read exactly ``n`` bytes, or ``None`` on a clean EOF before any byte.

    Raises on a partial read (a truncated frame is a protocol error, not a
    graceful hang-up).
    """
    chunks = []
    remaining = n
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            if remaining == n:
                return None  # clean EOF at a message boundary
            raise OSError("peer closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


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


# ---------------------------------------------------------------------------
# Consumer side: fetch the fd(s), import + map, copy into an owned buffer
# ---------------------------------------------------------------------------
#
# The consumer keeps one connection per socket path for the life of the process
# and reuses it across arrays. The connect/accept round-trip is a fixed cost
# independent of array size, so reusing the connection is what keeps the VMM
# path competitive with legacy IPC on small/medium arrays.

# _FETCH_CONNS_LOCK guards the cache dict only; _FETCH_IO_LOCK serializes the
# send/recv on the shared connection (a single request/reply must not interleave
# with another). Kept separate so a slow fetch does not block cache lookups.
_FETCH_CONNS: dict[str, socket.socket] = {}
_FETCH_CONNS_LOCK = threading.Lock()
_FETCH_IO_LOCK = threading.Lock()


def _fetch_conn(sock_path: str) -> socket.socket:
    """Return a cached connection to ``sock_path``, opening one if needed."""
    with _FETCH_CONNS_LOCK:
        conn = _FETCH_CONNS.get(sock_path)
        if conn is not None:
            return conn
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(sock_path)
        _FETCH_CONNS[sock_path] = conn
        return conn


def _drop_conn(sock_path: str) -> None:
    """Discard a cached (broken) connection so the next fetch reconnects."""
    with _FETCH_CONNS_LOCK:
        conn = _FETCH_CONNS.pop(sock_path, None)
    if conn is not None:
        try:
            conn.close()
        except OSError:
            pass


def _fetch_fds(sock_path: str, export_ids: list[int]) -> list[int]:
    """Fetch fds for ``export_ids`` from the producer's fd server, in order.

    Returns one fd per requested id. Raises if the server reports a miss for any
    id (each hit contributes a status byte of 1 and one fd in the SCM_RIGHTS
    message, in request order). Reuses a persistent connection per socket path;
    a broken connection is dropped and retried once.
    """
    count = len(export_ids)
    payload = count.to_bytes(_REQ_COUNT_BYTES, "little") + b"".join(
        eid.to_bytes(_ID_BYTES, "little") for eid in export_ids
    )

    def _attempt() -> list[int]:
        conn = _fetch_conn(sock_path)
        with _FETCH_IO_LOCK:  # one request/reply at a time on the shared conn
            conn.sendall(payload)
            statuses, ancdata = _recv_reply(conn, count)
        recv_fds: list[int] = []
        for level, ctype, data in ancdata:
            if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                for i in range(len(data) // 4):
                    recv_fds.append(int.from_bytes(data[i * 4 : i * 4 + 4], "little"))
        if any(s != 1 for s in statuses) or len(recv_fds) != count:
            # Close any fds we did receive so a partial hit does not leak.
            for fd in recv_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise RuntimeError(
                f"vmm fd server returned no fd for one of exports {export_ids}"
            )
        return recv_fds

    try:
        return _attempt()
    except (TimeoutError, OSError):
        # Stale connection (server restarted, etc.): reconnect once.
        _drop_conn(sock_path)
        return _attempt()


def _recv_reply(conn: socket.socket, count: int) -> tuple[bytes, list]:
    """Receive the ``count``-byte status prefix and its SCM_RIGHTS ancillary data."""
    msg, ancdata, _flags, _addr = conn.recvmsg(count, socket.CMSG_SPACE(count * 4))
    if len(msg) != count:
        raise OSError("short reply from vmm fd server")
    return msg, ancdata


def _fetch_fd(sock_path: str, export_id: int) -> int:
    """Fetch a single exported fd (convenience wrapper over :func:`_fetch_fds`)."""
    return _fetch_fds(sock_path, [export_id])[0]


def _device_ordinal(arr: Any) -> int:
    device = 0
    if hasattr(arr, "device"):
        dev = arr.device
        if hasattr(dev, "id"):
            device = dev.id
        elif hasattr(dev, "index") and dev.index is not None:
            device = dev.index
    return device


def _build_vmm_descriptor(arr: Any, server: _FdPassServer) -> ArrayDict:
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

    # cuMemRetainAllocationHandle succeeds only for VMM-backed memory. Because
    # json+cuda_vmm is an explicit opt-in, a non-VMM allocation here is a user
    # error, not something to silently paper over -- fail loudly and actionably
    # rather than degrading to a copy behind the user's back (that is what
    # json+cuda_ipc is for).
    handle = ctypes.c_ulonglong()
    ret = driver.cuMemRetainAllocationHandle(
        ctypes.byref(handle), ctypes.c_void_p(base.value)
    )
    if ret != 0:
        raise RuntimeError(
            "json+cuda_vmm requires VMM-backed device memory, but this array's "
            "allocation is not VMM-exportable (cuMemRetainAllocationHandle failed, "
            f"CUresult={ret}). This is expected for the default CuPy/PyTorch "
            "caching allocators. Either use json+cuda_ipc (always works; stages a "
            "copy for such memory), or allocate through a VMM-backed allocator "
            "(JAX/XLA, or PyTorch with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)."
        )

    # The export carries no ordering guarantee: make sure the producer's writes
    # to this memory are complete before a consumer can map and read it.
    _cu_check(driver.cuCtxSynchronize(), "cuCtxSynchronize")

    export_id = server.register(handle.value, arr)

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


def _import_map_and_copy(val: ArrayDict) -> IpcDeviceArray:
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


# ---------------------------------------------------------------------------
# DeviceTransport backend
# ---------------------------------------------------------------------------
#
# The VMM path shares the ``cuda_ipc`` wire format (``encoding: "cuda_ipc"``,
# with a ``vmm:``-prefixed buffer) and is selected transparently by
# :mod:`cuda_ipc` when the source memory is VMM-exportable. It is packaged here
# as a DeviceTransport so its fd-passing server lives in ``bootstrap`` (owned by
# the served app's lifespan) and its export bookkeeping/release ride the shared
# transport lifecycle rather than a bespoke set of module globals.


class VmmTransport:
    """DeviceTransport backend for the copy-free VMM fd path.

    The transport's *session* is the :class:`_FdPassServer`. ``bootstrap`` on the
    producer creates it; the served app owns that call (and the matching
    ``release``/``close``) via its lifespan. When no session has been
    bootstrapped -- the bare SDK path that encodes without a running server -- a
    process-global fallback server is started lazily, so a direct
    ``dump``/``load`` still works.
    """

    name = "vmm"
    reach = "same_host"

    def bootstrap(self, role: Any, peer_offer: Any = None) -> Any:
        """Producer: create and return the fd-passing server (the session).

        The server is also installed as the process fallback, so the encode path
        (which reaches VMM by pointer, deep inside serialization, without a
        session in hand) and this session are the *same* instance. The served
        app calls this once from its lifespan, giving the socket a deterministic
        startup and teardown instead of a leaked lazy daemon.

        The consumer needs no producer-side state (it pulls fds over the socket
        named in each descriptor), so its bootstrap is a no-op returning None.
        """
        if role == "producer":
            server = _FdPassServer()
            _set_fallback_server(server)
            return server
        return None

    def register(self, arr: Any, session: Any = None) -> ArrayDict:
        """Export ``arr`` by fd and build its ``vmm:`` descriptor.

        As with cuda_ipc, the per-array handle *is* the finished array dict, so
        :meth:`descriptor` is a passthrough.
        """
        server = session if session is not None else _get_fallback_server()
        return _build_vmm_descriptor(arr, server)

    def descriptor(self, handle: ArrayDict) -> ArrayDict:
        """Return the array dict :meth:`register` already produced."""
        return handle

    def flush(self, session: Any = None) -> None:
        """No-op: the consumer pulls fds, so there is nothing to post."""

    def receive(self, val: ArrayDict, session: Any = None) -> IpcDeviceArray:
        """Import the exported memory and copy it into an owned buffer."""
        return _import_map_and_copy(val)

    def release(self, session: Any = None) -> None:
        """Drop this request's retained VMM handles.

        A bootstrapped session installs itself as the fallback, so releasing the
        fallback covers both the served path and the sessionless SDK path; the
        explicit ``session`` release is a harmless no-op in that case. A no-op
        overall when nothing was ever exported.
        """
        if session is not None:
            session.release()
        with _FALLBACK_LOCK:
            server = _FALLBACK_SERVER
        if server is not None and server is not session:
            server.release()

    def shutdown(self) -> None:
        """Close the fallback/session server. Called by the app's lifespan."""
        global _FALLBACK_SERVER
        with _FALLBACK_LOCK:
            server = _FALLBACK_SERVER
            _FALLBACK_SERVER = None
        if server is not None:
            server.release()
            server.close()


# The fallback server backs both the served path (installed by ``bootstrap``)
# and the sessionless SDK path (started on first use). Kept process-global since
# the encode path reaches VMM by pointer without a session in hand.
_FALLBACK_SERVER: _FdPassServer | None = None
_FALLBACK_LOCK = threading.Lock()


def _set_fallback_server(server: _FdPassServer) -> None:
    global _FALLBACK_SERVER
    with _FALLBACK_LOCK:
        _FALLBACK_SERVER = server


def _get_fallback_server() -> _FdPassServer:
    global _FALLBACK_SERVER
    with _FALLBACK_LOCK:
        if _FALLBACK_SERVER is None:
            _FALLBACK_SERVER = _FdPassServer()
        return _FALLBACK_SERVER


def _register_vmm_transport() -> None:
    """Register the VMM backend once this module is imported."""
    from tesseract_core.runtime.device_transport import register_transport

    register_transport(VmmTransport())


_register_vmm_transport()


# ---------------------------------------------------------------------------
# Back-compat entry points used by cuda_ipc and the test suite
# ---------------------------------------------------------------------------
#
# cuda_ipc reaches the VMM path through these thin module-level functions (it
# selects VMM by pointer, sharing the wire format), and the tests monkeypatch
# them. They delegate to the registered transport so there is one implementation.


def dump_vmm_arraydict(arr: Any) -> ArrayDict:
    """Export ``arr`` via the VMM transport's fallback server. See :class:`VmmTransport`."""
    from tesseract_core.runtime.device_transport import get_transport

    transport = get_transport("vmm")
    return transport.register(arr)


def load_vmm_arraydict(val: ArrayDict) -> IpcDeviceArray:
    """Decode a VMM descriptor via the VMM transport. See :class:`VmmTransport`."""
    from tesseract_core.runtime.device_transport import get_transport

    return get_transport("vmm").receive(val)


def release_vmm_exports() -> None:
    """Release VMM handles retained by the fallback (sessionless) path.

    The served path releases through its lifespan-owned session; this covers the
    bare SDK path that dumps/loads without a session. A no-op when nothing was
    exported.
    """
    from tesseract_core.runtime.device_transport import get_transport

    get_transport("vmm").release()
