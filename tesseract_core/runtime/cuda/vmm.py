# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA VMM POSIX-fd sharing: the copy-free ``cuda_vmm`` GPU transport.

Legacy CUDA IPC (``cudaIpcGetMemHandle``) rejects memory allocated through the
CUDA Virtual Memory Management API (``cuMemCreate``) -- which is what modern
pooled allocators use, notably JAX/XLA's default GPU allocator and PyTorch's
``expandable_segments``. For those, the ``cuda_ipc`` transport falls back to
:func:`tesseract_core.runtime.cuda.api.stage_for_legacy_ipc`: an extra
device-to-device copy into a fresh ``cudaMalloc`` buffer that legacy IPC *can*
export. This transport removes that copy for VMM-backed memory by exporting the
VMM allocation *by reference* instead.

The mechanics differ from legacy IPC in two ways that shape the code:

* **The handle is a POSIX file descriptor**, not a serializable blob, so it
  cannot ride in the JSON response. It is passed out-of-band over a Unix domain
  socket via ``SCM_RIGHTS``. The producer runs a tiny fd-passing server; the
  JSON descriptor carries the socket path and an export id, and the consumer
  connects to fetch the fd. (Verified to work across a container->host boundary
  with ``--ipc=host`` plus a shared mount, i.e. the served-Tesseract deployment.)

* **The export carries no cross-process ordering guarantee**, so the producer
  must ``cuCtxSynchronize`` after any pending writes before handing off the fd.

``cuda_vmm`` is a sibling of ``cuda_ipc``, selected by ``gpu_transport``. It has
its own wire encoding (``encoding: "cuda_vmm"``, ``vmm:``-prefixed buffer) but
returns the same consumer-facing :class:`~tesseract_core.runtime.cuda.ipc.IpcDeviceArray`
wrapper, so the decode side never forks. The VMM machinery is packaged as a
:class:`~tesseract_core.runtime.device_transport.DeviceTransport`
(:class:`CudaVmmTransport`): the fd-passing server is its session, opened for the
producer via :meth:`CudaVmmTransport.session` (a context manager the served
app's lifespan, the CLI, and tests each wrap around their export work) and reused
across a request's exports. The open server is held in a process-global slot
because the encode path reaches VMM by pointer deep inside serialization, with no
session object in scope to thread down. Exporting without an open session raises.
"""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np

from tesseract_core.runtime.array_encoding import ArrayDict
from tesseract_core.runtime.cuda import api as cuda_api
from tesseract_core.runtime.cuda.ipc import (
    IpcDeviceArray,
    _is_c_contiguous,
    has_cuda_array_interface,
)
from tesseract_core.runtime.device_transport import DeviceTransport


def is_vmm_exportable(data_ptr: int) -> bool:
    """Whether ``data_ptr`` is backed by a VMM allocation we can export by fd.

    ``cuMemRetainAllocationHandle`` succeeds only for memory allocated via the
    VMM API (``cuMemCreate``) -- JAX/XLA's allocator, PyTorch
    ``expandable_segments``, or our own ``cuMemCreate`` buffers. Default
    CuPy/PyTorch pools and legacy ``cudaMalloc`` return an error, so the caller
    keeps the legacy IPC / staging path for those.
    Returns ``False`` (rather than raising) if the CUDA driver cannot even be
    loaded -- e.g. a GPU-less host running the mocked encode path -- so the caller
    transparently keeps the legacy path there too.
    """
    try:
        handle = cuda_api.retain_allocation_handle(data_ptr)
    except RuntimeError:
        return False
    if handle is None:
        return False
    cuda_api.mem_release(handle)
    return True


# ---------------------------------------------------------------------------
# Producer side: fd-passing server
# ---------------------------------------------------------------------------
#
# A POSIX fd is only meaningful once passed to another process via SCM_RIGHTS, so
# the producer runs a small Unix-socket server that hands out the fd for an
# export id on request. The socket path travels in the JSON descriptor. The
# server is the transport's *session*: :meth:`CudaVmmTransport.session` opens it
# (owned by the served app's lifespan; see ``serve.create_rest_api``), and the
# same instance serves every export until the session closes.

# AF_UNIX message framing: an 8-byte little-endian count N, followed by N 8-byte
# little-endian export ids. The reply is N status bytes plus one SCM_RIGHTS
# control message carrying a fd for every id that hit (misses contribute a 0
# status byte and no fd). Batching lets a consumer fetch a whole apply's exports
# in one round-trip; a single-id request is just N == 1.
_REQ_COUNT_BYTES = 8
_ID_BYTES = 8

# AF_UNIX socket paths are capped by the OS (108 bytes on Linux, 104 on macOS).
# A shared bind-mount (container output_path) or a long system tempdir can push
# an absolute socket path past that. The portable escape hatch is to operate on
# the path relative to its directory -- the kernel only ever sees the short leaf
# name -- by temporarily changing into that directory around the bind/connect.
# chdir is process-global, so it is serialized and restored under this lock.
_CWD_LOCK = threading.Lock()
# Comfortably under the smaller (macOS) limit, leaving room for the NUL and any
# platform slack; absolute paths at or below this bind directly.
_AF_UNIX_MAX = 100


def _bind_or_connect_short(sock: socket.socket, path: str, *, bind: bool) -> None:
    """Bind or connect ``sock`` to ``path``, tolerating over-long AF_UNIX paths.

    If the absolute ``path`` fits the AF_UNIX limit it is used directly. Otherwise
    the operation is retried from inside the socket's directory using the leaf
    name only, so the kernel sees a short relative path. The directory switch is
    process-global, hence serialized under ``_CWD_LOCK`` and always restored.
    """
    op = sock.bind if bind else sock.connect
    if len(os.fsencode(path)) <= _AF_UNIX_MAX:
        op(path)
        return
    directory, name = os.path.split(path)
    if len(os.fsencode(name)) > _AF_UNIX_MAX:
        raise OSError(
            f"AF_UNIX socket name {name!r} is itself too long ({len(name)} bytes); "
            "set a shorter TESSERACT_VMM_SOCKET_DIR."
        )
    with _CWD_LOCK:
        prev = os.getcwd()
        os.chdir(directory)
        try:
            op(name)
        finally:
            os.chdir(prev)


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
        # One private dir per server (isolates the socket so it can be made
        # world-connectable without exposing siblings); the socket name inside it
        # is kept short so the full path stays within the AF_UNIX limit where it
        # can (and _bind_or_connect_short covers the case where base_dir alone is
        # already long, e.g. a deep macOS tempdir or a nested output_path mount).
        self._dir = tempfile.mkdtemp(prefix="tsr-vmm-", dir=base_dir)
        self.path = os.path.join(self._dir, f"{uuid.uuid4().hex[:8]}.sock")
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _bind_or_connect_short(self._sock, self.path, bind=True)
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
        worth of exports (mirrors ``cuda.ipc.release_pinned_ipc_exports``).
        """
        with self._lock:
            entries = list(self._registry.values())
            self._registry.clear()
        if not entries:
            # Nothing was exported via the VMM path (the usual case): don't touch
            # the driver at all, so this stays a no-op rather than failing to
            # load libcuda on a host without it.
            return
        for handle, _keepalive in entries:
            cuda_api.mem_release(handle)

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
        try:
            for pos, export_id in enumerate(export_ids):
                with self._lock:
                    entry = self._registry.get(export_id)
                if entry is None:
                    continue  # miss: status stays 0, no fd
                handle, _keepalive = entry
                fd = cuda_api.export_to_shareable_fd(handle)
                if fd is None:
                    continue
                statuses[pos] = 1
                fds.append(fd)
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

    An explicit ``vmm_socket_dir`` config value (``TESSERACT_VMM_SOCKET_DIR``)
    wins; otherwise the runtime's ``output_path`` -- the host<->container shared
    mount, so a host consumer can reach the socket a containerized server binds --
    is used, falling back to the system temp dir for the bare same-host case.
    Never raises: falls back to temp on any error.
    """
    try:
        from tesseract_core.runtime.config import get_config

        config = get_config()
        if config.vmm_socket_dir:
            return config.vmm_socket_dir
        if config.output_path and config.output_path != ".":
            return config.output_path
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
        _bind_or_connect_short(conn, sock_path, bind=False)
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
    """Export a VMM-backed CUDA array by fd and return a ``cuda_vmm`` descriptor.

    The descriptor's ``buffer`` uses the VMM form
    ``vmm:{sockpath_b64}:{export_id}:{storage_offset}:{storage_size}:{device}``.
    Requires :func:`is_vmm_exportable` to be true for the array's pointer; the
    caller checks that before routing here.
    """
    if not has_cuda_array_interface(arr):
        raise ValueError("cuda_vmm encoding requires a CUDA array")
    if not _is_c_contiguous(arr):
        raise ValueError("cuda_vmm encoding requires a C-contiguous array")

    iface = arr.__cuda_array_interface__
    data_ptr = iface["data"][0]
    shape = tuple(iface["shape"])
    dtype = np.dtype(iface["typestr"])
    device = _device_ordinal(arr)

    # Retain the VMM handle backing this pointer, and record the byte offset of
    # the array within the whole mapped allocation (pooled VMM allocators hand
    # out many arrays from one reservation).
    base_ptr, storage_size = cuda_api.get_allocation_base(data_ptr)
    storage_offset = data_ptr - base_ptr

    # cuMemRetainAllocationHandle succeeds only for VMM-backed memory. Because
    # cuda_vmm is an explicit opt-in, a non-VMM allocation here is a user error,
    # not something to silently paper over -- fail loudly and actionably rather
    # than degrading to a copy behind the user's back (that is what cuda_ipc is
    # for).
    handle = cuda_api.retain_allocation_handle(base_ptr)
    if handle is None:
        raise RuntimeError(
            "gpu_transport='cuda_vmm' requires VMM-backed device memory, but this "
            "array's allocation is not VMM-exportable (cuMemRetainAllocationHandle "
            "failed). This is expected for the default CuPy/PyTorch "
            "caching allocators. Either use gpu_transport='cuda_ipc' (always works; "
            "stages a copy for such memory), or allocate through a VMM-backed "
            "allocator (JAX/XLA, or PyTorch with "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)."
        )

    # The export carries no ordering guarantee: make sure the producer's writes
    # to this memory are complete before a consumer can map and read it.
    cuda_api.ctx_synchronize()

    export_id = server.register(handle, arr)

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
            "encoding": "cuda_vmm",
        },
    }


def _import_map_and_copy(val: ArrayDict) -> IpcDeviceArray:
    """Decode a VMM ``cuda_vmm`` descriptor: map the producer's memory, copy out.

    Imports the producer's VMM allocation (via the fd fetched over the socket),
    maps it, copies just this array's own bytes into a fresh ``cudaMalloc``
    buffer owned by this process, unmaps, and returns an :class:`IpcDeviceArray`
    -- the same consumer-facing wrapper as ``cuda_ipc``. The borrow of the
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

    # Allocate the owned buffer up front (on the target device) so that if any
    # later step fails we still unmap and free cleanly.
    cuda_api.set_device(device)
    owned_ptr = cuda_api.malloc(nbytes)

    fd = _fetch_fd(sock_path, export_id)
    handle = 0
    mapped_ptr = 0
    mapped = False
    try:
        handle = cuda_api.import_from_shareable_fd(fd)
        mapped_ptr = cuda_api.address_reserve(storage_size)
        cuda_api.mem_map(mapped_ptr, storage_size, handle)
        mapped = True
        cuda_api.mem_set_access_rw(mapped_ptr, storage_size, device)

        # Copy just this array's bytes (at its offset) into our owned buffer,
        # then block until the copy is done so we never unmap mid-copy.
        cuda_api.memcpy_device_to_device(owned_ptr, mapped_ptr + storage_offset, nbytes)
        cuda_api.device_synchronize()
    except Exception:
        cuda_api.free(owned_ptr)
        raise
    finally:
        if mapped:
            cuda_api.mem_unmap(mapped_ptr, storage_size)
            cuda_api.address_free(mapped_ptr, storage_size)
        if handle:
            cuda_api.mem_release(handle)
        # The imported fd is dup'd into our process; close our copy.
        try:
            os.close(fd)
        except OSError:
            pass

    return IpcDeviceArray(owned_ptr, device, shape, dtype)


# ---------------------------------------------------------------------------
# DeviceTransport backend
# ---------------------------------------------------------------------------


class CudaVmmTransport(DeviceTransport):
    """DeviceTransport backend for the copy-free VMM fd path.

    The transport's *session* is the :class:`_FdPassServer`. A producer opens one
    with :meth:`session` (a context manager) for the span of its export work, and
    the open server installs itself as the process-active server so the encode
    path can find it.
    """

    name = "cuda_vmm"
    reach = "same_host"

    @contextmanager
    def session(
        self, role: Literal["producer", "consumer"] = "producer"
    ) -> Iterator[Any]:
        """Open a producer fd-passing server for the duration of the ``with``.

        This is the explicit lifecycle every producer uses -- the served app's
        lifespan, the CLI ``run`` path, and tests each wrap their export work in
        it. On enter it creates the :class:`_FdPassServer` and installs it as the
        process-active server, so the encode path (which reaches VMM by pointer
        deep inside serialization, without a session object in hand) finds it. On
        exit it releases the request's retained handles and closes the socket.

        Consumers need no producer-side state (they pull fds over the socket named
        in each descriptor), so a consumer session is a no-op yielding ``None``.
        """
        if role != "producer":
            yield None
            return
        server = _FdPassServer()
        _set_active_server(server)
        try:
            yield server
        finally:
            _clear_active_server(server)
            server.release()
            server.close()

    def bootstrap(self, role: Any, peer_offer: Any = None) -> Any:
        """Satisfy the ``DeviceTransport`` contract; prefer :meth:`session`.

        The consumer needs no producer-side state (it pulls fds over the socket
        named in each descriptor), so it returns ``None``. The producer lifecycle
        lives in :meth:`session`, whose ``with`` block guarantees teardown.
        """
        return None

    def register(self, arr: Any, session: Any = None) -> ArrayDict:
        """Export ``arr`` by fd and build its ``vmm:`` descriptor.

        As with cuda_ipc, the per-array handle *is* the finished array dict, so
        :meth:`descriptor` is a passthrough. ``session`` is used when supplied;
        otherwise the process-active server (from an open :meth:`session`) is
        required -- exporting with no session open is a programming error.
        """
        server = session if session is not None else _require_active_server()
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

        Releases the given ``session`` when supplied, else the process-active
        server. A no-op when no server is active (nothing was ever exported).
        """
        server = session if session is not None else _active_server()
        if server is not None:
            server.release()


# The producer's fd-passing server, held process-global for the life of an open
# session so the encode path can look it up (it reaches VMM by pointer, with no
# session object to thread down). Populated only by an open ``session``, so
# exporting with none open raises.
_ACTIVE_SERVER: _FdPassServer | None = None
_ACTIVE_SERVER_LOCK = threading.Lock()


def _set_active_server(server: _FdPassServer) -> None:
    global _ACTIVE_SERVER
    with _ACTIVE_SERVER_LOCK:
        _ACTIVE_SERVER = server


def _clear_active_server(expected: _FdPassServer) -> None:
    """Clear the active slot iff it still holds ``expected``.

    Guards against clobbering a newer session that opened concurrently.
    """
    global _ACTIVE_SERVER
    with _ACTIVE_SERVER_LOCK:
        if _ACTIVE_SERVER is expected:
            _ACTIVE_SERVER = None


def _active_server() -> _FdPassServer | None:
    with _ACTIVE_SERVER_LOCK:
        return _ACTIVE_SERVER


def _require_active_server() -> _FdPassServer:
    server = _active_server()
    if server is None:
        raise RuntimeError(
            "cuda_vmm export attempted with no active fd-passing session. Open one "
            "with CudaVmmTransport.session() (the served app does this in its "
            "lifespan) before encoding GPU arrays via gpu_transport='cuda_vmm'."
        )
    return server
