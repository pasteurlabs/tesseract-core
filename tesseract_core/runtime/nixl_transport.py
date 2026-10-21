# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL device transport: point-to-point GPU array exchange via NIXL.

Like ``cuda_ipc`` (:mod:`tesseract_core.runtime.cuda.ipc`), this transport passes
a GPU array between two processes without a host round-trip. NIXL picks its
backend by itself: CUDA IPC or shared memory on one host, and UCX across hosts
(RDMA where the hardware supports it, TCP otherwise). NIXL is only loaded once a
Tesseract encodes or decodes a NIXL array.

The consumer initiates the transfer, a READ. The producer registers the array's
device memory with its NIXL agent and sends its agent's name and metadata, the
array's address and its device in the JSON message. The consumer adds the
producer as a remote agent and reads the bytes into a buffer of its own, which it
returns as the same :class:`~tesseract_core.runtime.cuda.ipc.IpcDeviceArray` that
``cuda_ipc`` returns. The producer's registration does nothing until then, so
this fits the request lifecycle of ``cuda_ipc``.

The message holds plain numbers instead of NIXL's descriptor serialization,
which is a pickle, so neither process unpickles what the other sends.
"""

from __future__ import annotations

import collections
import contextlib
import ctypes
import importlib
import os
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

import numpy as np
import pybase64

from tesseract_core.runtime.array_encoding import ArrayDict
from tesseract_core.runtime.config import get_config
from tesseract_core.runtime.cuda import api as cuda_api
from tesseract_core.runtime.cuda.ipc import (
    IpcDeviceArray,
    contiguous_on_device,
    has_cuda_array_interface,
    on_device,
    without_autograd,
)
from tesseract_core.runtime.device_transport import DeviceTransport

# ---------------------------------------------------------------------------
# Lazy NIXL agent (one per process)
# ---------------------------------------------------------------------------
#
# Each process has a NIXL agent, which owns the UCX backend and the memory
# registrations. Concurrent requests share it, so it is created with strict
# thread synchronization (by default, NIXL synchronizes nothing).
#
# Arrays read in parallel (see _read_spans) go through a second agent, with one
# UCX worker per read. Every read connects to its producer afresh (see
# _remote_agent), which takes longer the more workers the reading agent has, so
# all other reads use the single-worker agent.

_NIXL_AGENT: Any = None
_PARALLEL_READ_AGENT: Any = None
_NIXL_AGENT_LOCK = threading.Lock()

# Threads that run the parallel reads of an array. NIXL runs each thread's
# transfers on one of the agent's UCX workers, so reads only use several
# workers, and connections, when they come from several threads.
_READ_POOL: ThreadPoolExecutor | None = None

# A lock per producer agent, so that reads from one producer take turns (see
# _remote_agent), and the number of reads using each lock.
_PRODUCER_LOCKS: dict[str, threading.Lock] = {}
_PRODUCER_READS: collections.Counter[str] = collections.Counter()
_PRODUCER_LOCKS_LOCK = threading.Lock()

# How long a consumer waits for a posted READ to complete. It only runs out when
# a transfer stalls, e.g. because the producer died, not on a slow large one.
_XFER_TIMEOUT = 120.0


def _nixl_agent_name() -> str:
    """A NIXL agent name unique to this process, even across hosts.

    NIXL refuses a remote agent with the local agent's name, and pids repeat
    across hosts, so the name adds a random part.
    """
    return f"tesseract-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _import_nixl_api() -> Any:
    """Import the NIXL build that matches the CUDA runtime this process uses.

    NIXL ships one build per CUDA major, and two builds cannot share a process.
    Its ``nixl`` package picks one from the CUDA framework it finds (torch, if
    installed), which can differ from the runtime that
    :mod:`tesseract_core.runtime.cuda.api` loads. Each build's UCX also loads
    libcudart of its major by bare soname, which fails where the runtime comes
    from a pip wheel, and UCX then treats GPU memory as host memory.

    So this makes the runtime's libcudart global, where UCX finds it, and
    imports the build already imported, else the matching one, else NIXL's own
    choice.
    """
    ctypes.CDLL(cuda_api.runtime_library_path(), mode=ctypes.RTLD_GLOBAL)

    loaded = sys.modules.get("nixl._api")
    if loaded is not None:
        return loaded

    build = f"nixl_cu{cuda_api.runtime_version() // 1000}"
    try:
        return importlib.import_module(f"{build}._api")
    except ModuleNotFoundError as exc:
        if exc.name != build:
            raise
    return importlib.import_module("nixl._api")


def _create_nixl_agent(num_workers: int) -> Any:
    """Create a NIXL agent with a UCX backend of ``num_workers`` workers."""
    try:
        nixl_api = _import_nixl_api()
    except ImportError as exc:  # pragma: no cover - env-dependent
        if not (isinstance(exc, ModuleNotFoundError) and exc.name == "nixl"):
            raise RuntimeError(
                f"The 'nixl' GPU transport could not import NIXL: {exc}"
            ) from exc
        package = f"nixl-cu{cuda_api.runtime_version() // 1000}"
        raise RuntimeError(
            "The 'nixl' GPU transport requires NIXL, which is not installed in "
            f"the Python environment at {sys.prefix}. Install the build for its "
            f"CUDA runtime with `pip install {package}` (Linux only). For a "
            f"Tesseract, add {package} to tesseract_requirements.txt and rebuild."
        ) from exc

    # UCX reads its configuration from the environment when the backend is
    # created.
    seg_size = get_config().nixl_ucx_tcp_seg_size
    if seg_size:
        for var in ("UCX_TCP_TX_SEG_SIZE", "UCX_TCP_RX_SEG_SIZE"):
            os.environ.setdefault(var, seg_size)

    agent = nixl_api.nixl_agent(
        _nixl_agent_name(),
        nixl_api.nixl_agent_config(
            backends=[],
            sync_mode=nixl_api.nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
        ),
    )
    agent.create_backend("UCX", {"num_workers": str(num_workers)})
    return agent


def _get_nixl_agent() -> Any:
    """Lazily create this process's NIXL agent, or explain why NIXL is missing."""
    global _NIXL_AGENT
    with _NIXL_AGENT_LOCK:
        if _NIXL_AGENT is None:
            _NIXL_AGENT = _create_nixl_agent(num_workers=1)
        return _NIXL_AGENT


def _get_parallel_read_agent() -> Any:
    """Lazily create the NIXL agent for parallel reads."""
    global _PARALLEL_READ_AGENT
    with _NIXL_AGENT_LOCK:
        if _PARALLEL_READ_AGENT is None:
            _PARALLEL_READ_AGENT = _create_nixl_agent(
                num_workers=get_config().nixl_parallel_reads
            )
        return _PARALLEL_READ_AGENT


def _read_pool() -> ThreadPoolExecutor:
    """Lazily create the threads that run parallel reads."""
    global _READ_POOL
    with _NIXL_AGENT_LOCK:
        if _READ_POOL is None:
            _READ_POOL = ThreadPoolExecutor(
                max_workers=get_config().nixl_parallel_reads,
                thread_name_prefix="tesseract-nixl-read",
            )
        return _READ_POOL


def _read_spans(nbytes: int) -> list[tuple[int, int]]:
    """Split an array's ``nbytes`` into the (offset, size) spans read in parallel."""
    config = get_config()
    parts = config.nixl_parallel_reads
    if parts == 1 or nbytes < config.nixl_parallel_min_bytes:
        return [(0, nbytes)]
    size = -(-nbytes // parts)
    return [(offset, min(size, nbytes - offset)) for offset in range(0, nbytes, size)]


# ---------------------------------------------------------------------------
# Encode side: register the array and describe the transfer
# ---------------------------------------------------------------------------


class NixlExportGroup:
    """The arrays one message exported via NIXL, kept registered until :meth:`release`.

    Only the thread encoding the message adds to a group.
    """

    def __init__(self) -> None:
        # (registration, source array) pairs.
        self.exports: list[tuple[Any, Any]] = []

    def __len__(self) -> int:
        return len(self.exports)

    def release(self) -> None:
        """Deregister the exported arrays and let go of them."""
        exports, self.exports = self.exports, []
        if not exports:
            return
        agent = _get_nixl_agent()
        for registration, _arr in exports:
            # A server releases exports while answering later, unrelated
            # requests, which must not fail over this. A failed deregistration
            # only leaks the registration.
            try:
                agent.deregister_memory(registration)
            except Exception:  # noqa: BLE001, S110
                pass


def dump_nixl_arraydict(arr: Any, group: NixlExportGroup) -> ArrayDict:
    """Register a CUDA array in ``group`` and describe it as a JSON array dict.

    A strided array is first copied on the device, as for cuda_ipc.
    """
    if not has_cuda_array_interface(arr):
        raise ValueError(
            "nixl encoding requires a CUDA array "
            f"(object with __cuda_array_interface__), got {type(arr).__name__}"
        )

    arr, info = contiguous_on_device(without_autograd(arr))

    if info.dtype.kind == "V":
        raise TypeError(
            f"nixl cannot encode arrays of dtype {info.dtype}, which have no "
            "NumPy equivalent (e.g. bfloat16)."
        )

    if info.nbytes == 0:
        # An empty array has no memory to register, so its descriptor is empty
        # and the consumer recreates it without a transfer.
        buffer = ":::"
    else:
        agent = _get_nixl_agent()
        registration = agent.register_memory(
            [(info.data_ptr, info.nbytes, info.device, "")], "VRAM"
        )
        group.exports.append((registration, arr))

        # The consumer reads the memory from another process, outside any stream
        # ordering on this side, so wait for the producer's kernels to finish
        # before the descriptor leaves the process.
        with on_device(info.device):
            cuda_api.device_synchronize()

        meta_b64 = pybase64.b64encode_as_string(agent.get_agent_metadata())
        buffer = f"{agent.name}:{meta_b64}:{info.data_ptr}:{info.device}"

    return {
        "object_type": "array",
        "shape": list(info.shape),
        "dtype": info.dtype.name,
        "data": {"buffer": buffer, "encoding": "nixl"},
    }


# ---------------------------------------------------------------------------
# Decode side: add the remote agent and READ into an owned buffer
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _remote_agent(agent: Any, name: str, metadata: bytes) -> Iterator[None]:
    """Load a producer's agent metadata for the duration of one read.

    NIXL cannot load metadata for a producer it already knows, and a producer's
    metadata changes with every message, as its pooled allocator registers the
    same buffers anew. So each read loads its message's metadata and forgets the
    producer afterwards, and reads from one producer take turns.
    """
    with _PRODUCER_LOCKS_LOCK:
        lock = _PRODUCER_LOCKS.setdefault(name, threading.Lock())
        _PRODUCER_READS[name] += 1
    try:
        with lock:
            loaded_name = agent.add_remote_agent(metadata)
            # NIXL returns the name as bytes, though it annotates it as str.
            if isinstance(loaded_name, bytes):
                loaded_name = loaded_name.decode()
            try:
                if loaded_name != name:
                    raise ValueError(
                        f"nixl descriptor names producer {name!r}, but its "
                        f"metadata describes {loaded_name!r}"
                    )
                yield
            finally:
                agent.remove_remote_agent(loaded_name)
    finally:
        with _PRODUCER_LOCKS_LOCK:
            _PRODUCER_READS[name] -= 1
            if not _PRODUCER_READS[name]:
                del _PRODUCER_READS[name]
                del _PRODUCER_LOCKS[name]


def load_nixl_arraydict(val: ArrayDict) -> IpcDeviceArray:
    """Read the array a NIXL descriptor describes into a new buffer on the active device."""
    peer_name, meta_b64, address, remote_device = val["data"]["buffer"].split(":")
    dtype = np.dtype(val["dtype"])
    shape = tuple(val["shape"])
    nbytes = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
    device = cuda_api.get_device()

    if not meta_b64:
        # An empty array's descriptor carries no transfer (see the encoder).
        if nbytes:
            raise ValueError(
                f"nixl descriptor is empty, but the array of shape {shape} and "
                f"dtype {dtype} is not"
            )
        return IpcDeviceArray(0, device, shape, dtype)

    remote_meta = pybase64.b64decode(meta_b64, validate=True)

    spans = _read_spans(nbytes)
    agent = _get_parallel_read_agent() if len(spans) > 1 else _get_nixl_agent()
    # The producer registered exactly these bytes. Its address and device ordinal
    # identify them to its own agent and need not mean anything in this process.
    remote_address, remote_device = int(address), int(remote_device)

    def read(offset: int, size: int) -> None:
        local = agent.get_xfer_descs([(owned_ptr + offset, size, device)], "VRAM")
        remote = agent.get_xfer_descs(
            [(remote_address + offset, size, remote_device)], "VRAM"
        )
        # Entering the device makes its primary context current on this thread,
        # a pool thread for parallel reads, which UCX's CUDA transport needs.
        with on_device(device):
            handle = agent.initialize_xfer(
                "READ", local, remote, peer_name, b"tesseract"
            )
            try:
                agent.transfer(handle)
                _wait_for_xfer(agent, handle)
            finally:
                agent.release_xfer_handle(handle)

    with _remote_agent(agent, peer_name, remote_meta), on_device(device):
        owned_ptr = cuda_api.malloc(nbytes)
        try:
            local_reg = agent.register_memory([(owned_ptr, nbytes, device, "")], "VRAM")
            try:
                if len(spans) == 1:
                    read(*spans[0])
                else:
                    # Wait for every read before the buffer can be freed, even
                    # when one of them fails.
                    pool = _read_pool()
                    done, _ = wait([pool.submit(read, *span) for span in spans])
                    for future in done:
                        future.result()
            finally:
                agent.deregister_memory(local_reg)
        except Exception:
            cuda_api.free(owned_ptr)
            raise

    return IpcDeviceArray(owned_ptr, device, shape, dtype)


def _wait_for_xfer(agent: Any, handle: Any) -> None:
    """Block until a NIXL transfer completes, raising on error or timeout.

    A posted READ can stall indefinitely if the producer dies mid-transfer, so
    give up after ``_XFER_TIMEOUT`` seconds rather than hang the consumer forever.
    """
    deadline = time.monotonic() + _XFER_TIMEOUT
    while True:
        state = agent.check_xfer_state(handle)
        if state == "DONE":
            return
        if state == "ERR":
            raise RuntimeError("nixl transfer failed")
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"nixl transfer did not complete within {_XFER_TIMEOUT}s"
            )
        time.sleep(0.0005)


# ---------------------------------------------------------------------------
# DeviceTransport backend
# ---------------------------------------------------------------------------


class NixlTransport(DeviceTransport):
    """DeviceTransport backend for the ``nixl`` transport."""

    name = "nixl"
    reach = "both"

    def bootstrap(self, role: Any, peer_offer: Any) -> None:
        """No-op, since each message carries the producer's agent metadata."""

    def new_exports(self) -> NixlExportGroup:
        """A fresh :class:`NixlExportGroup` for one message's exports."""
        return NixlExportGroup()

    def register(self, arr: Any, session: Any = None) -> ArrayDict:
        """Register ``arr`` in ``session`` (a :class:`NixlExportGroup`) and describe it.

        The per-array handle *is* the array dict, so :meth:`descriptor` is a
        passthrough.
        """
        if session is None:
            raise ValueError(
                "nixl exports need a session from new_exports(), which keeps them "
                "registered until it is released"
            )
        return dump_nixl_arraydict(arr, session)

    def descriptor(self, handle: ArrayDict) -> ArrayDict:
        """Return the array dict :meth:`register` already produced."""
        return handle

    def flush(self, session: Any = None) -> None:
        """No-op, since the consumer initiates the READ."""

    def receive(self, val: ArrayDict, session: Any = None) -> IpcDeviceArray:
        """READ the exported bytes into a fresh consumer-owned ``IpcDeviceArray``."""
        return load_nixl_arraydict(val)

    def release(self, session: Any = None) -> None:
        """Deregister the exports of ``session`` (a :class:`NixlExportGroup`)."""
        session.release()
