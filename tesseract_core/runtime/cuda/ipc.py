# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA IPC transport: zero-copy GPU array exchange between processes.

This module holds everything specific to the ``cuda_ipc`` GPU transport, kept
separate from the framework-agnostic host encodings in
:mod:`tesseract_core.runtime.array_encoding`. Nothing here is imported unless a
Tesseract actually encodes or decodes a CUDA IPC array, so the CUDA runtime and
driver libraries are only touched on that path.

All low-level CUDA access lives in the :mod:`tesseract_core.runtime.cuda`
package: this module works purely with plain Python values (device pointers as
``int``, IPC handles as ``bytes``) and never imports ctypes. It contributes only
the *transport policy* -- how a GPU array maps to and from the ``cuda_ipc`` JSON
payload, plus the keepalive bookkeeping that the transfer protocol requires and
the pool that recycles staging buffers across requests. What the GPU transports
share, such as the :class:`~tesseract_core.runtime.cuda.arrays.DeviceArray`
they decode into, lives in :mod:`tesseract_core.runtime.cuda.arrays`.

The JSON schema for this encoding (``CudaIpcArrayData``) lives alongside the
other array-data models in :mod:`array_encoding`. The entry points are
:func:`dump_cuda_ipc_arraydict` / :func:`load_cuda_ipc_arraydict`, the
encode/decode pair, and :class:`ExportGroup`, which keeps one message's exports
alive until its consumer is done with them.

Out-of-process consumers that ``dlopen`` libcudart themselves (e.g. the
``tesseract_jax`` C++ FFI shim) can reuse the library discovery -- including the
pip-wheel fallback and forward-compatible version range -- via
:func:`tesseract_core.runtime.cuda.iter_cudart_candidates` instead of
maintaining their own soname list.
"""

import contextlib
import math
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
import pybase64

from tesseract_core.runtime.array_encoding import ArrayDict
from tesseract_core.runtime.cuda import api as cuda_api
from tesseract_core.runtime.cuda.arrays import (
    BufferPool,
    DeviceArray,
    cuda_array_to_host,
    has_cuda_array_interface,
    on_device,
    prepare_export,
    take_owned_buffer,
    validate_cuda_array,
)
from tesseract_core.runtime.device_transport import DeviceTransport

# These live in cuda.arrays, shared by all GPU transports, and stay importable
# from here for code written against earlier releases.
IpcDeviceArray = DeviceArray

__all__ = [
    "CudaIpcTransport",
    "ExportGroup",
    "IpcDeviceArray",
    "cuda_array_to_host",
    "dump_cuda_ipc_arraydict",
    "has_cuda_array_interface",
    "load_cuda_ipc_arraydict",
    "validate_cuda_array",
]


# ---------------------------------------------------------------------------
# CUDA IPC array encode / decode
# ---------------------------------------------------------------------------

# Keeping exports alive.
#
# A CUDA IPC handle is only valid while the *exporting* process keeps the source
# allocation alive. If the exported array were freed the instant it is handed
# off (before the consumer opens and copies it out) a pooled allocator
# (CuPy/PyTorch) could recycle the block, so the consumer would silently read
# *wrong* data.
#
# So each message's exports go into an ExportGroup, which keeps them (and the
# staging buffers they were copied into) until the consumer is done with that
# message, and only that message:
#
#   * Client: the SDK releases a request's group when the HTTP call returns. The
#     server decodes the client's inputs *during* request handling, and
#     :func:`load_cuda_ipc_arraydict` copies each input into server-owned memory
#     and is done with the mapping before the response is sent.
#
#   * Server: a response's group must outlive the response, since the client
#     copies the outputs out only once it has received it. The server keeps the
#     group until the client acknowledges the response (see
#     :mod:`tesseract_core.runtime.cuda.serving`).


class ExportGroup:
    """The arrays one message exported via CUDA IPC, kept alive until :meth:`release`.

    Also holds the staging buffers they were copied into (see
    :func:`_stage_for_export`), which :meth:`release` returns to the pool. Only
    the thread encoding the message adds to a group.
    """

    def __init__(self) -> None:
        self.pins: list[Any] = []
        # Staging buffers as (device pointer, device, size, IPC handle).
        self.staging: list[tuple[int, int, int, bytes]] = []

    def __len__(self) -> int:
        return len(self.pins) + len(self.staging)

    def release(self) -> None:
        """Let go of the pinned arrays and return the staging buffers to the pool."""
        self.pins.clear()
        staging, self.staging = self.staging, []
        for ptr, device, nbytes, handle in staging:
            _STAGING_POOL.put(ptr, device, nbytes, handle)


# Idle staging buffers, each with its IPC handle, so reuse also saves the
# cudaIpcGetMemHandle call.
_STAGING_POOL = BufferPool()


def _stage_for_export(
    src_ptr: int, nbytes: int, device: int, group: ExportGroup
) -> bytes:
    """Copy ``nbytes`` from ``src_ptr`` into a pooled staging buffer and return its IPC handle.

    The buffer stays in ``group`` until the group is released.
    """
    pooled = _STAGING_POOL.take(device, nbytes)
    ptr, handle = pooled if pooled is not None else (cuda_api.malloc(nbytes), None)
    try:
        # The copy runs on the legacy default stream, which is not ordered after
        # the producer's non-blocking streams (JAX's, for one), so wait for the
        # producer's kernels to finish writing the source first.
        cuda_api.device_synchronize()
        cuda_api.memcpy_device_to_device(ptr, src_ptr, nbytes)
        if handle is None:
            handle = cuda_api.ipc_get_mem_handle(ptr)
    except Exception:
        cuda_api.free(ptr)
        raise
    group.staging.append((ptr, device, nbytes, handle))
    return handle


def dump_cuda_ipc_arraydict(arr: Any, group: ExportGroup) -> ArrayDict:
    """Dump a CUDA array to a JSON dict with a CUDA IPC handle.

    Works with any object that implements ``__cuda_array_interface__``, such
    as CuPy arrays, PyTorch tensors, and single-device JAX arrays.

    The IPC handle allows another process on the same host (with --ipc=host)
    to access the GPU memory directly without any CPU round-trip.

    The source array stays pinned in ``group`` until the group is released, so
    its GPU memory is not freed or recycled before the consumer copies it out.

    Frameworks with VMM/pool-backed GPU allocators (e.g. JAX/XLA) hand out
    pointers that the legacy ``cudaIpcGetMemHandle`` API rejects. For those,
    encode stages the array's bytes into a reusable ``cudaMalloc`` buffer with
    one on-GPU copy (see :func:`_stage_for_export`) and exports a handle to that
    instead, which is still far cheaper than a host round-trip.
    """
    arr, info = prepare_export(arr, "cuda_ipc")

    data_ptr, nbytes = info.data_ptr, info.nbytes

    device_uuid = cuda_api.device_uuid(info.device)

    if nbytes == 0:
        # An empty array has no memory to share (frameworks give it a null
        # pointer, which the CUDA calls below reject), so its descriptor
        # carries no handle and the consumer recreates it without opening one.
        return {
            "object_type": "array",
            "shape": list(info.shape),
            "dtype": info.dtype.name,
            "data": {"buffer": f"{device_uuid}::0:0", "encoding": "cuda_ipc"},
        }

    # Keep the source allocation alive until exports are explicitly released.
    # (Not strictly needed on the VMM fallback path below, which copies out of
    # `arr` before the handle leaves the process, but keeping the pin gives both
    # paths identical cleanup.)
    group.pins.append(arr)

    # Synchronization and staging allocations act on the active device, which
    # need not be the array's.
    with on_device(info.device):
        # IPC handles reference the *whole* backing allocation, not the array's
        # (possibly offset) data pointer. Pooled allocators (CuPy, PyTorch) hand
        # out many arrays from a single cudaMalloc block, so we must resolve the
        # allocation base, take the handle on that base, and record the byte
        # offset of this array within the allocation.
        base_ptr, storage_size = cuda_api.get_allocation_base(data_ptr)
        storage_offset = data_ptr - base_ptr

        try:
            handle_bytes = cuda_api.ipc_get_mem_handle(base_ptr)
        except RuntimeError:
            # Legacy IPC rejected this pointer, almost certainly because it's
            # VMM-backed. Stage only this array's own bytes, not the whole
            # (possibly huge) backing allocation.
            handle_bytes = _stage_for_export(data_ptr, nbytes, info.device, group)
            storage_offset = 0
            storage_size = nbytes

        # The consumer reads the memory from another process, outside any
        # stream ordering on this side, so wait for the producer's kernels and
        # the staging copy to finish before the handle leaves the process.
        cuda_api.device_synchronize()

    handle_b64 = pybase64.b64encode_as_string(handle_bytes)
    return {
        "object_type": "array",
        "shape": list(info.shape),
        "dtype": info.dtype.name,
        "data": {
            "buffer": f"{device_uuid}:{handle_b64}:{storage_offset}:{storage_size}",
            "encoding": "cuda_ipc",
        },
    }


# IPC mappings open in this process, by device and handle: the mapped pointer
# and how many decodes use it. CUDA maps an allocation into a process only once,
# and several arrays can share one allocation, so decodes running at the same
# time share one mapping. It is closed when the last of them is done, so no
# mapping outlives the decodes, during which the producer keeps the memory.
_MAPPINGS: dict[tuple[int, bytes], list[int]] = {}
_MAPPINGS_LOCK = threading.Lock()


@contextlib.contextmanager
def _mapped(handle_bytes: bytes, device: int) -> Iterator[int]:
    """Map the allocation behind an IPC handle for the block, sharing open mappings."""
    key = (device, handle_bytes)
    with _MAPPINGS_LOCK:
        entry = _MAPPINGS.get(key)
        if entry is None:
            entry = _MAPPINGS[key] = [
                cuda_api.ipc_open_mem_handle(handle_bytes),
                0,
            ]
        entry[1] += 1
    try:
        yield entry[0]
    finally:
        with _MAPPINGS_LOCK:
            entry[1] -= 1
            if entry[1] == 0:
                del _MAPPINGS[key]
                cuda_api.ipc_close_mem_handle(entry[0])


def _device_with_uuid(device_uuid: str) -> int:
    """This process's ordinal for the CUDA device with UUID ``device_uuid``."""
    for device in range(cuda_api.device_count()):
        if cuda_api.device_uuid(device) == device_uuid:
            return device
    raise ValueError(
        f"The array lives on the GPU with UUID {device_uuid}, which this process "
        "cannot see. Give both processes access to that GPU (e.g. via --gpus or "
        "CUDA_VISIBLE_DEVICES)."
    )


def load_cuda_ipc_arraydict(val: ArrayDict) -> DeviceArray:
    """Load a CUDA array from a JSON dict with a CUDA IPC handle.

    The calling process must share the IPC namespace with the producer
    (e.g. both run with --ipc=host on Docker) and see the same GPU, though they
    may number it differently.

    Returns a caller-owned :class:`DeviceArray`. Decoding maps the IPC handle
    (see :func:`_mapped`), copies the array's own bytes device-to-device into a
    ``cudaMalloc`` buffer owned by this process (reusing a released buffer of
    the same size if there is one), and synchronizes. The borrow of the
    producer's memory therefore lasts only for a single on-GPU copy, so the
    producer is free to reuse or release the exported buffer as soon as this
    call returns.

    The result carries no framework dependency: it exposes both
    ``__cuda_array_interface__`` and ``__dlpack__`` so Torch/JAX/CuPy can adopt
    it zero-copy, plus ``.copy_to_host()`` / ``np.asarray(...)`` for inspection.
    """
    device_uuid, handle_b64, storage_offset_str, _storage_size_str = val["data"][
        "buffer"
    ].split(":")
    handle_bytes = pybase64.b64decode(handle_b64, validate=True)
    device = _device_with_uuid(device_uuid)
    storage_offset = int(storage_offset_str)

    dtype = np.dtype(val["dtype"])
    shape = tuple(val["shape"])
    nbytes = math.prod(shape) * dtype.itemsize

    if not handle_bytes:
        # An empty array's descriptor carries no handle (see the encoder).
        if nbytes:
            raise ValueError(
                f"cuda_ipc descriptor has no handle, but the array of shape "
                f"{shape} and dtype {dtype} is not empty"
            )
        return DeviceArray(0, device, shape, dtype)

    # Allocation and synchronization act on the active device, which need not
    # be the array's.
    with on_device(device):
        # Get the owned buffer up front so that if any later step fails we still
        # close the IPC mapping and free the buffer cleanly.
        owned_ptr = take_owned_buffer(device, nbytes)

        try:
            # Opening the IPC handle can fail too; if it does, we still own the
            # buffer obtained above and must free it (the except below).
            with _mapped(handle_bytes, device) as base_ptr:
                # Copy only this array's own bytes out of the producer's (offset)
                # mapping into the owned buffer, then block until the copy is done so
                # we never unmap mid-copy.
                cuda_api.memcpy_device_to_device(
                    owned_ptr, base_ptr + storage_offset, nbytes
                )
                cuda_api.device_synchronize()
        except Exception:
            cuda_api.free(owned_ptr)
            raise

    return DeviceArray(owned_ptr, device, shape, dtype)


class CudaIpcTransport(DeviceTransport):
    """DeviceTransport backend for the same-host ``cuda_ipc`` transport."""

    name = "cuda_ipc"
    reach = "same_host"

    def bootstrap(self, role: Any, peer_offer: Any) -> None:
        """No-op: the IPC handle is self-contained, so no shared state to set up."""

    def new_exports(self) -> ExportGroup:
        """A fresh :class:`ExportGroup` for one message's exports."""
        return ExportGroup()

    def register(self, arr: Any, session: Any = None) -> ArrayDict:
        """Pin ``arr`` in ``session`` (an :class:`ExportGroup`) and return its IPC descriptor.

        cuda_ipc mints the handle and packs the wire string in one call, so the
        per-array handle *is* the array dict and :meth:`descriptor` is a
        passthrough. Splitting them would take the IPC handle twice for nothing.
        """
        if session is None:
            raise ValueError(
                "cuda_ipc exports need a session from new_exports(), which keeps "
                "them alive until it is released"
            )
        return dump_cuda_ipc_arraydict(arr, session)

    def descriptor(self, handle: ArrayDict) -> ArrayDict:
        """Return the array dict :meth:`register` already produced."""
        return handle

    def flush(self, session: Any = None) -> None:
        """No-op: cuda_ipc is receiver-driven, so there is nothing to post."""

    def receive(self, val: ArrayDict, session: Any = None) -> DeviceArray:
        """Copy the exported bytes into a fresh consumer-owned ``DeviceArray``."""
        return load_cuda_ipc_arraydict(val)

    def release(self, session: Any = None) -> None:
        """Drop the producer-side pins of ``session`` (an :class:`ExportGroup`)."""
        session.release()
