# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL device transport: GPU array exchange between processes, also across hosts.

NIXL chooses how to move the bytes, using CUDA IPC or shared memory on one host
and UCX across hosts (over RDMA where the hardware supports it, TCP otherwise).
NIXL is only imported once a Tesseract encodes or decodes a ``nixl`` array.

The consumer pulls the array with a NIXL READ. The producer registers the
array's device memory with its NIXL agent and describes it in the JSON message
(see :class:`~tesseract_core.runtime.array_encoding.NixlArrayData`). The
consumer loads the producer's agent metadata and reads the bytes into a buffer
of its own, which it returns as the same
:class:`~tesseract_core.runtime.cuda.arrays.DeviceArray` that ``cuda_ipc``
returns. The producer posts no transfer of its own, so the transport fits the
same request lifecycle as ``cuda_ipc``.

The message carries the address and device as plain numbers rather than NIXL's
descriptor serialization, which is a pickle, so neither process unpickles what
the other sends.
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib
import math
import os
import sys
import threading
import time
import uuid
import weakref
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

import numpy as np
import pybase64

from tesseract_core.runtime.array_encoding import ArrayDict
from tesseract_core.runtime.config import get_config
from tesseract_core.runtime.cuda import api as cuda_api
from tesseract_core.runtime.cuda.arrays import (
    DeviceArray,
    on_device,
    prepare_export,
    take_owned_buffer,
)
from tesseract_core.runtime.device_transport import DeviceTransport

# ---------------------------------------------------------------------------
# Lazy NIXL agents
# ---------------------------------------------------------------------------
#
# Each process has a main NIXL agent with a single UCX worker, which registers
# the arrays the process exports. Concurrent requests share it, so it is
# created with strict thread synchronization, since by default NIXL
# synchronizes nothing.
#
# Arrays read in parallel (see _read_spans) go through a second agent, with one
# UCX worker per read. Every read connects to its producer afresh (see
# _remote_agent), which takes longer the more workers the reading agent has, so
# all other reads use the main agent.

_MAIN_AGENT: Any = None
_PARALLEL_READ_AGENT: Any = None
# Guards the lazily created state of this module.
_INIT_LOCK = threading.Lock()

# The number of parallel reads, taken from the config once so that the spans,
# the parallel-read agent's workers and the read threads always agree.
_PARALLEL_READS: int | None = None

# Threads that run the parallel reads of an array. NIXL runs each thread's
# transfers on one of the agent's UCX workers, so reads only use several
# workers, and connections, when they come from several threads.
_READ_POOL: ThreadPoolExecutor | None = None

# A lock per producer agent, so that reads from one producer take turns (see
# _remote_agent). Each lock is dropped once no read holds it.
_PRODUCER_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = (
    weakref.WeakValueDictionary()
)
_PRODUCER_LOCKS_LOCK = threading.Lock()

# Arrays of at least this many bytes are read in parallel (see _read_spans).
# Smaller ones gain little from it and connect faster on the main agent.
_PARALLEL_MIN_BYTES = 16 * 2**20

# Seconds a consumer waits for a posted READ to complete, so that a stalled
# transfer (e.g. because the producer died) does not hang it forever.
_XFER_TIMEOUT = 120.0


def _nixl_agent_name() -> str:
    """A fresh NIXL agent name, unique even across hosts.

    NIXL refuses a remote agent with the local agent's name, and pids repeat
    across hosts, so the name adds a random part.
    """
    return f"tesseract-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _import_nixl_api() -> Any:
    """Import the NIXL build for the CUDA runtime that :mod:`~tesseract_core.runtime.cuda.api` loaded.

    NIXL ships one build per CUDA major version, and two builds cannot share a
    process. Its ``nixl`` package picks a build from the CUDA framework it finds
    (torch, if installed), which may not match the runtime loaded here. So this
    imports the matching build directly, unless ``nixl`` already loaded one.

    Each build's UCX loads libcudart by bare soname, which fails when the
    runtime comes from a pip wheel, and UCX then treats GPU memory as host
    memory. Loading the runtime with ``RTLD_GLOBAL`` first lets UCX find it.
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
        package = build.replace("_", "-")
        raise RuntimeError(
            "The 'nixl' GPU transport requires NIXL, which is not installed in "
            f"the Python environment at {sys.prefix}. Install the build for its "
            f"CUDA runtime with `pip install {package}` (Linux only). For a "
            f"Tesseract, add {package} to tesseract_requirements.txt and rebuild."
        ) from exc


def _create_nixl_agent(num_workers: int) -> Any:
    """Create a NIXL agent with a UCX backend of ``num_workers`` workers."""
    try:
        nixl_api = _import_nixl_api()
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            f"The 'nixl' GPU transport could not import NIXL: {exc}"
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


def _get_main_agent() -> Any:
    """Return this process's main NIXL agent, creating it on first use."""
    global _MAIN_AGENT
    with _INIT_LOCK:
        if _MAIN_AGENT is None:
            _MAIN_AGENT = _create_nixl_agent(num_workers=1)
        return _MAIN_AGENT


def _parallel_reads() -> int:
    """The configured ``nixl_parallel_reads``, fixed at its first use in this process."""
    global _PARALLEL_READS
    with _INIT_LOCK:
        if _PARALLEL_READS is None:
            _PARALLEL_READS = get_config().nixl_parallel_reads
        return _PARALLEL_READS


def _get_parallel_read_agent() -> Any:
    """Lazily create the NIXL agent for parallel reads."""
    global _PARALLEL_READ_AGENT
    num_workers = _parallel_reads()
    with _INIT_LOCK:
        if _PARALLEL_READ_AGENT is None:
            _PARALLEL_READ_AGENT = _create_nixl_agent(num_workers=num_workers)
        return _PARALLEL_READ_AGENT


def _read_pool() -> ThreadPoolExecutor:
    """Lazily create the threads that run parallel reads."""
    global _READ_POOL
    max_workers = _parallel_reads()
    with _INIT_LOCK:
        if _READ_POOL is None:
            _READ_POOL = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="tesseract-nixl-read"
            )
        return _READ_POOL


def _read_spans(nbytes: int) -> list[tuple[int, int]]:
    """Split ``nbytes`` into the (offset, size) spans to read in parallel, or one span if small."""
    parts = _parallel_reads()
    if parts == 1 or nbytes < _PARALLEL_MIN_BYTES:
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
        agent = _get_main_agent()
        for registration, _arr in exports:
            # A server releases exports while answering later, unrelated
            # requests, which must not fail over this. A failed deregistration
            # only leaks the registration.
            try:
                agent.deregister_memory(registration)
            except Exception:  # noqa: BLE001, S110
                pass


def dump_nixl_arraydict(arr: Any, group: NixlExportGroup) -> ArrayDict:
    """Register a CUDA array in ``group`` and describe it as a JSON array dict."""
    arr, info = prepare_export(arr, "nixl")

    if info.nbytes == 0:
        # An empty array has no memory to register, so its descriptor is empty
        # and the consumer recreates it without a transfer.
        buffer = ":::"
    else:
        agent = _get_main_agent()
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

    NIXL cannot load metadata for a producer it already knows, yet a producer's
    metadata changes with every message, because it registers that message's
    buffers anew even at addresses its pooled allocator reuses. So each read
    loads its own message's metadata and forgets the producer afterwards, and
    reads from one producer take turns.
    """
    with _PRODUCER_LOCKS_LOCK:
        lock = _PRODUCER_LOCKS.setdefault(name, threading.Lock())
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


def load_nixl_arraydict(val: ArrayDict) -> DeviceArray:
    """Read the array a NIXL descriptor describes into a new buffer on the active device."""
    peer_name, meta_b64, address, remote_device = val["data"]["buffer"].split(":")
    dtype = np.dtype(val["dtype"])
    shape = tuple(val["shape"])
    nbytes = math.prod(shape) * dtype.itemsize
    device = cuda_api.get_device()

    if not meta_b64:
        # An empty array's descriptor carries no transfer (see the encoder).
        if nbytes:
            raise ValueError(
                f"nixl descriptor is empty, but the array of shape {shape} and "
                f"dtype {dtype} is not"
            )
        return DeviceArray(0, device, shape, dtype)

    remote_meta = pybase64.b64decode(meta_b64, validate=True)

    spans = _read_spans(nbytes)
    agent = _get_parallel_read_agent() if len(spans) > 1 else _get_main_agent()
    # The producer registered exactly these bytes. Its address and device ordinal
    # identify them to its own agent and need not mean anything in this process.
    remote_address, remote_device = int(address), int(remote_device)

    def read(offset: int, size: int) -> None:
        local = agent.get_xfer_descs([(owned_ptr + offset, size, device)], "VRAM")
        remote = agent.get_xfer_descs(
            [(remote_address + offset, size, remote_device)], "VRAM"
        )
        # UCX's CUDA transport needs the device's primary context current on
        # this thread, which may be a fresh pool thread, so enter the device.
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
        owned_ptr = take_owned_buffer(device, nbytes)
        try:
            local_reg = agent.register_memory([(owned_ptr, nbytes, device, "")], "VRAM")
            try:
                if len(spans) == 1:
                    read(*spans[0])
                else:
                    # Wait for every read, even after one fails, so the buffer
                    # is not freed while another read still writes to it.
                    pool = _read_pool()
                    done, _ = wait([pool.submit(read, *span) for span in spans])
                    for future in done:
                        future.result()
            finally:
                agent.deregister_memory(local_reg)
        except Exception:
            cuda_api.free(owned_ptr)
            raise

    return DeviceArray(owned_ptr, device, shape, dtype)


def _wait_for_xfer(agent: Any, handle: Any) -> None:
    """Block until a NIXL transfer completes, raising on error or after ``_XFER_TIMEOUT``."""
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
        """Register ``arr`` in ``session`` (a :class:`NixlExportGroup`) and return its array dict."""
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

    def receive(self, val: ArrayDict, session: Any = None) -> DeviceArray:
        """READ the exported bytes into a fresh consumer-owned ``DeviceArray``."""
        return load_nixl_arraydict(val)

    def release(self, session: Any = None) -> None:
        """Deregister the exports of ``session`` (a :class:`NixlExportGroup`)."""
        session.release()
