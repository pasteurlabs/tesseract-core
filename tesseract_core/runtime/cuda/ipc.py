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
the pools that recycle device buffers across requests.

The JSON schema for this encoding (``CudaIpcArrayData``) lives alongside the
other array-data models in :mod:`array_encoding`; the public entry points used
by :mod:`array_encoding` are:

* :func:`has_cuda_array_interface` -- detect a GPU leaf,
* :func:`cuda_array_to_host` -- host-copy helper for non-IPC encodings of GPU
  arrays,
* :func:`validate_cuda_array` -- shape/dtype validation without a device copy,
* :func:`dump_cuda_ipc_arraydict` / :func:`load_cuda_ipc_arraydict` -- the
  encode/decode pair,
* :func:`release_pinned_ipc_exports` -- keepalive cleanup, called once per
  request by both the server (for its outputs) and the client (for its inputs),
* :func:`start_transport_check` / :func:`answer_transport_check` /
  :func:`finish_transport_check` -- the exchange that tells a client whether
  ``cuda_ipc`` works between it and a particular server.

Out-of-process consumers that ``dlopen`` libcudart themselves (e.g. the
``tesseract_jax`` C++ FFI shim) can reuse the library discovery -- including the
pip-wheel fallback and forward-compatible version range -- via
:func:`tesseract_core.runtime.cuda.iter_cudart_candidates` instead of
maintaining their own soname list.
"""

import collections
import contextlib
import functools
import math
import os
import threading
import weakref
from collections.abc import Iterator
from typing import Any, NamedTuple

import numpy as np
import pybase64

from tesseract_core.runtime.array_encoding import (
    ArrayDict,
    ShapeType,
    check_shape_dtype_no_cast,
)
from tesseract_core.runtime.config import get_config
from tesseract_core.runtime.cuda import api as cuda_api
from tesseract_core.runtime.cuda import dlpack
from tesseract_core.runtime.device_transport import DeviceTransport

__all__ = [
    "IpcDeviceArray",
    "answer_transport_check",
    "cuda_array_to_host",
    "dump_cuda_ipc_arraydict",
    "finish_transport_check",
    "has_cuda_array_interface",
    "load_cuda_ipc_arraydict",
    "release_pinned_ipc_exports",
    "start_transport_check",
    "validate_cuda_array",
]


def has_cuda_array_interface(obj: Any) -> bool:
    """Check if an object exposes the __cuda_array_interface__ protocol.

    This protocol is supported by PyTorch, CuPy, JAX, Numba, and most
    CUDA-aware Python libraries. It indicates the object holds data in
    GPU device memory.
    """
    return hasattr(obj, "__cuda_array_interface__")


FORBID_DEVICE_HOST_COPY_ENV = "TESSERACT_FORBID_DEVICE_HOST_COPY"


def check_device_host_copy(what: str) -> None:
    """Raise if implicit device-to-host copies are forbidden.

    Setting ``TESSERACT_FORBID_DEVICE_HOST_COPY`` to ``1`` or ``true`` makes
    silent host copies of GPU data fail loudly, so tests can assert that a path
    stays on-device. Subprocess servers inherit the variable, and the SDK client
    checks it too. The guarded paths are GPU arrays serialized without a device
    transport and ``np.asarray`` on an :class:`IpcDeviceArray`. Explicit copies
    via :meth:`IpcDeviceArray.copy_to_host` are never blocked.
    """
    if os.environ.get(FORBID_DEVICE_HOST_COPY_ENV, "").lower() in {"1", "true"}:
        raise RuntimeError(
            f"Implicit device-to-host copy of {what} "
            f"({FORBID_DEVICE_HOST_COPY_ENV} is set)."
        )


def cuda_array_to_host(arr: Any) -> np.ndarray:
    """Copy a GPU array to a host NumPy array.

    Used for non-IPC encodings, where the bytes must reach the host, and so
    subject to :func:`check_device_host_copy`. Handles CuPy (``.get()``) and
    PyTorch (``.cpu().numpy()``) explicitly, then falls back to ``np.asarray``
    for any other framework whose arrays support ``__array__`` (e.g. JAX, which
    fetches to host on conversion).
    """
    check_device_host_copy(f"a {type(arr).__name__} GPU array")
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


class _CudaArrayInfo(NamedTuple):
    """Device-array metadata read from ``__cuda_array_interface__``.

    ``strides`` is in bytes; ``None`` means row-major contiguous.
    """

    data_ptr: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: np.dtype
    device: int
    strides: tuple[int, ...] | None

    def is_c_contiguous(self) -> bool:
        """Whether the array's memory is row-major contiguous.

        cuda_ipc moves ``nbytes`` consecutive bytes and the decoder rebuilds the
        array from shape and dtype alone, so callers must reject non-contiguous
        sources, which would otherwise be silently misread.
        """
        if self.strides is None:
            return True
        expected = []
        acc = self.dtype.itemsize
        for dim in reversed(self.shape):
            expected.append(acc)
            acc *= dim
        expected.reverse()
        return self.strides == tuple(expected)


def _read_cuda_array_info(arr: Any) -> _CudaArrayInfo:
    """Read a CUDA array's metadata from ``__cuda_array_interface__``.

    The protocol carries no device ordinal, so it is read from the framework's
    ``.device`` attribute and defaults to 0.
    """
    iface = arr.__cuda_array_interface__
    shape = tuple(iface["shape"])
    dtype = np.dtype(iface["typestr"])  # e.g. "<f4", "|b1"
    strides = iface.get("strides")

    device = 0
    dev = getattr(arr, "device", None)
    if isinstance(dev, int):
        device = dev  # IpcDeviceArray
    elif hasattr(dev, "id"):
        device = dev.id  # CuPy
    elif getattr(dev, "index", None) is not None:
        device = dev.index  # PyTorch

    return _CudaArrayInfo(
        data_ptr=iface["data"][0],
        nbytes=math.prod(shape) * dtype.itemsize,
        shape=shape,
        dtype=dtype,
        device=device,
        strides=None if strides is None else tuple(strides),
    )


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
#     client's inputs *during* request handling, and :func:`load_cuda_ipc_arraydict`
#     copies each input into server-owned memory and closes the mapping before
#     the response is sent. By the time the HTTP call returns (with the body
#     buffered), the inputs are provably dead and can be released immediately.
#
# Both rely on the same two assumptions:
#   1. Requests are issued *serially* (never concurrently).
#   2. The consumer copies decoded arrays into consumer-owned memory before it
#      releases the exporter's buffer (which the decode path does
#      unconditionally; see :func:`load_cuda_ipc_arraydict`).
_CUDA_IPC_EXPORT_REGISTRY: list[Any] = []


class _BufferPool:
    """Idle ``cudaMalloc`` buffers kept for reuse, keyed by device and size.

    Iterative callers send the same shapes on every request, so without a pool
    both sides of a cuda_ipc exchange would free and reallocate the same large
    buffers each time. When two processes share a GPU, a ``cudaMalloc`` that
    receives memory the other process just freed can take tens of milliseconds
    per GiB.

    Each buffer can carry a value, such as a staging buffer's IPC handle. Idle
    buffers beyond :func:`_pool_max_bytes` per device are freed, least recently
    released first.
    """

    def __init__(self) -> None:
        # Device pointer -> (device, size, value), least recently released first.
        self._idle: collections.OrderedDict[int, tuple[int, int, Any]] = (
            collections.OrderedDict()
        )
        self._idle_bytes: collections.Counter[int] = collections.Counter()
        self._lock = threading.Lock()

    def take(self, device: int, nbytes: int) -> tuple[int, Any] | None:
        """Remove and return ``(ptr, value)`` for an idle buffer of this device and size, or ``None``."""
        with self._lock:
            for ptr in reversed(self._idle):
                buf_device, buf_nbytes, value = self._idle[ptr]
                if buf_device == device and buf_nbytes == nbytes:
                    del self._idle[ptr]
                    self._idle_bytes[device] -= nbytes
                    return ptr, value
        return None

    def put(self, ptr: int, device: int, nbytes: int, value: Any = None) -> None:
        """Keep a buffer for reuse, then free the oldest idle buffers beyond the cap."""
        if not ptr:
            return
        max_bytes = _pool_max_bytes(device)
        evicted = []
        with self._lock:
            self._idle[ptr] = (device, nbytes, value)
            self._idle_bytes[device] += nbytes
            for old_ptr, (old_device, old_nbytes, _) in list(self._idle.items()):
                if self._idle_bytes[device] <= max_bytes:
                    break
                if old_device == device:
                    del self._idle[old_ptr]
                    self._idle_bytes[device] -= old_nbytes
                    evicted.append(old_ptr)
        for old_ptr in evicted:
            cuda_api.free(old_ptr)


def _pool_max_bytes(device: int) -> int:
    """Cap on a :class:`_BufferPool`'s idle bytes on ``device``, set by ``cuda_ipc_pool_fraction``."""
    fraction = get_config().cuda_ipc_pool_fraction
    return int(cuda_api.device_total_memory(device) * fraction / 2)


# Staging buffers in use by the current exports, as (device pointer, device,
# size, IPC handle). Released together with _CUDA_IPC_EXPORT_REGISTRY, but
# returned to _STAGING_POOL instead of dropped.
_CUDA_IPC_STAGING_BUFFERS: list[tuple[int, int, int, bytes]] = []

# Idle staging buffers, each with its IPC handle, so reuse also saves the
# cudaIpcGetMemHandle call.
_STAGING_POOL = _BufferPool()


def _stage_for_export(src_ptr: int, nbytes: int, device: int) -> bytes:
    """Copy ``nbytes`` from ``src_ptr`` into a pooled staging buffer and return its IPC handle."""
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
    _CUDA_IPC_STAGING_BUFFERS.append((ptr, device, nbytes, handle))
    return handle


def release_pinned_ipc_exports() -> None:
    """Release arrays pinned for CUDA IPC export and recycle their staging buffers.

    Both sides of an exchange call this, at the points the registry comment
    above explains.
    """
    _CUDA_IPC_EXPORT_REGISTRY.clear()
    staging, _CUDA_IPC_STAGING_BUFFERS[:] = list(_CUDA_IPC_STAGING_BUFFERS), []
    for ptr, device, nbytes, handle in staging:
        _STAGING_POOL.put(ptr, device, nbytes, handle)


def _pin_cuda_ipc_export(arr: Any) -> None:
    """Retain a reference to a source array so its GPU memory stays valid.

    The reference is held until the exporting side calls
    :func:`release_pinned_ipc_exports`.
    """
    _CUDA_IPC_EXPORT_REGISTRY.append(arr)


@contextlib.contextmanager
def _on_device(device: int) -> Iterator[None]:
    """Make ``device`` the active CUDA device for the block, then restore the caller's."""
    previous = cuda_api.get_device()
    if previous == device:
        yield
        return
    cuda_api.set_device(device)
    try:
        yield
    finally:
        cuda_api.set_device(previous)


def dump_cuda_ipc_arraydict(arr: Any) -> ArrayDict:
    """Dump a CUDA array to a JSON dict with a CUDA IPC handle.

    Works with any object that implements ``__cuda_array_interface__``, such
    as CuPy arrays, PyTorch tensors, and single-device JAX arrays.

    The IPC handle allows another process on the same host (with --ipc=host)
    to access the GPU memory directly without any CPU round-trip.

    The source array is pinned in a process-global registry (see
    :func:`_pin_cuda_ipc_export`) so its GPU memory is not freed or recycled
    before the consumer copies it out; the pin is released once the exporting
    side calls :func:`release_pinned_ipc_exports`.

    Frameworks with VMM/pool-backed GPU allocators (e.g. JAX/XLA) hand out
    pointers that the legacy ``cudaIpcGetMemHandle`` API rejects. For those,
    encode stages the array's bytes into a reusable ``cudaMalloc`` buffer with
    one on-GPU copy (see :func:`_stage_for_export`) and exports a handle to that
    instead, which is still far cheaper than a host round-trip.
    """
    return _dump_cuda_ipc_arraydict(arr, pin=True)


def _dump_cuda_ipc_arraydict(arr: Any, *, pin: bool) -> ArrayDict:
    """See :func:`dump_cuda_ipc_arraydict`, which pins ``arr`` (``pin=True``)."""
    if not has_cuda_array_interface(arr):
        raise ValueError(
            "cuda_ipc encoding requires a CUDA array "
            f"(object with __cuda_array_interface__), got {type(arr).__name__}"
        )

    info = _read_cuda_array_info(arr)
    if not info.is_c_contiguous():
        raise ValueError(
            "cuda_ipc encoding requires a C-contiguous array; got one with "
            f"strides {info.strides}. Make a contiguous copy first (e.g. "
            "cupy.ascontiguousarray / torch.Tensor.contiguous)."
        )

    data_ptr, nbytes = info.data_ptr, info.nbytes

    # Keep the source allocation alive until exports are explicitly released.
    # (Not strictly needed on the VMM fallback path below, which copies out of
    # `arr` before the handle leaves the process, but keeping the pin gives both
    # paths identical cleanup.)
    if pin:
        _pin_cuda_ipc_export(arr)

    # Synchronization and staging allocations act on the active device, which
    # need not be the array's.
    with _on_device(info.device):
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
            handle_bytes = _stage_for_export(data_ptr, nbytes, info.device)
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
            "buffer": f"{info.device}:{handle_b64}:{storage_offset}:{storage_size}",
            "encoding": "cuda_ipc",
        },
    }


# ---------------------------------------------------------------------------
# Decoded device-array wrapper
# ---------------------------------------------------------------------------
#
# The decode path returns an object (:class:`IpcDeviceArray`) that owns a device
# buffer and exposes it via both ``__cuda_array_interface__`` and DLPack, so
# Torch/JAX/CuPy can all adopt it zero-copy without CuPy being a decode-time
# dependency. The DLPack ABI machinery lives in
# :mod:`tesseract_core.runtime.cuda.dlpack`.

# Idle owned buffers of decoded arrays. An array's buffer returns here once the
# array (or the framework tensor that adopted it) is released, and the next
# decode of the same size reuses it.
_DECODE_POOL = _BufferPool()


def _finalize_ipc_device_array(state: dict) -> None:
    """Release an :class:`IpcDeviceArray`'s device buffer exactly once.

    Registered via :func:`weakref.finalize`, so it runs when the array is
    garbage-collected *and* at interpreter shutdown, and can fire at most once.
    ``state`` is the array's mutable ownership record, shared by reference with
    the live object so ``__dlpack__`` can hand ownership off before this runs:

    * ``dlpack_token is None`` and not ``released``: we still own the buffer, so
      return it to the pool.
    * ``dlpack_token`` set: ``__dlpack__`` moved the buffer into a DLPack bundle;
      drop the bundle iff its capsule was never consumed (a consumer that took
      the capsule already owns the free).
    """
    try:
        if state["dlpack_token"] is None:
            if not state["released"]:
                state["release"]()
                state["released"] = True
        else:
            dlpack.drop_unconsumed_bundle(state["dlpack_token"])
    except Exception:  # noqa: BLE001, S110
        # Finalizers must never raise.
        pass


class IpcDeviceArray:
    """Owns a device buffer decoded from a CUDA IPC handle.

    The buffer is a process-owned ``cudaMalloc`` allocation holding the array's
    own bytes (the producer's IPC mapping is copied into it and then closed by
    the decoder). The object is framework-agnostic:

    * ``__cuda_array_interface__`` (v3) lets CuPy / Numba / PyTorch adopt it,
    * ``__dlpack__`` / ``__dlpack_device__`` let Torch and JAX adopt it,

    both zero-copy. ``.copy_to_host()`` / ``np.asarray(...)`` materialise a host
    NumPy copy so it can be inspected without any GPU framework installed. Only
    ``np.asarray`` is subject to :func:`check_device_host_copy`.

    The device buffer returns to the decode pool exactly once. Either a
    :func:`weakref.finalize` callback releases it (see
    :func:`_finalize_ipc_device_array`), or a DLPack consumer takes it (the
    capsule is renamed to ``"used_dltensor"`` on consumption, transferring the
    release to the consumer's deleter). ``_state["released"]`` guards against a
    double release.
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
        #   release:      returns the buffer to the pool.
        #   released:     True once the buffer is released or ownership was
        #                 handed to a DLPack capsule. Prevents a double release.
        #   dlpack_token: token of the DLPack bundle produced by __dlpack__, or
        #                 None if __dlpack__ was never called. Ownership of the
        #                 buffer moves into that bundle when it is created.
        self._state: dict = {
            "release": functools.partial(_DECODE_POOL.put, ptr, device, self._nbytes),
            "released": False,
            "dlpack_token": None,
        }
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
        if self._state["released"]:
            raise RuntimeError("device buffer has been released")
        host = np.empty(self.shape, dtype=self.dtype)
        cuda_api.memcpy_device_to_host(host.ctypes.data, self._ptr, self._nbytes)
        cuda_api.device_synchronize()
        return host

    def __array__(self, dtype: Any = None) -> np.ndarray:
        check_device_host_copy("an IpcDeviceArray")
        host = self.copy_to_host()
        return host if dtype is None else host.astype(dtype)

    # -- DLPack ----------------------------------------------------------

    def __dlpack_device__(self) -> tuple[int, int]:
        return (dlpack.DLDEVICE_CUDA, self.device)

    def __dlpack__(self, stream: Any = None, **kwargs: Any) -> Any:
        """Return a ``"dltensor"`` PyCapsule wrapping the owned buffer.

        Ownership of the device buffer moves into a self-contained DLPack bundle
        (see :func:`tesseract_core.runtime.cuda.dlpack.make_dlpack_capsule`)
        whose deleter returns it to the pool. The bundle's lifetime is
        deliberately *not* tied to this object's, because a consumer (Torch/JAX)
        may keep the tensor long after this ``IpcDeviceArray`` is gone and will
        call the deleter then. Whoever ends up owning the capsule (the consumer,
        or the finalizer for an un-consumed capsule) releases the buffer exactly
        once.
        """
        if self._state["released"]:
            raise RuntimeError("device buffer has been released")

        capsule, token = dlpack.make_dlpack_capsule(
            self._ptr,
            self.device,
            self.shape,
            self.dtype,
            release=self._state["release"],
        )
        # The buffer now belongs to the bundle; this object must not free it.
        # The finalizer reads this shared state to drop the bundle iff its
        # capsule is never consumed.
        self._state["dlpack_token"] = token
        self._state["released"] = True
        return capsule


def load_cuda_ipc_arraydict(val: ArrayDict) -> "IpcDeviceArray":
    """Load a CUDA array from a JSON dict with a CUDA IPC handle.

    The calling process must share the IPC namespace with the producer
    (e.g. both run with --ipc=host on Docker) and see the same GPU.

    Returns a caller-owned :class:`IpcDeviceArray`. Decoding opens the IPC
    handle, copies the array's own bytes device-to-device into a ``cudaMalloc``
    buffer owned by this process (reusing a released buffer of the same size if
    there is one), synchronizes, and closes the mapping before returning. The
    borrow of the producer's memory therefore lasts only for a single on-GPU
    copy, so the producer is free to reuse or release the exported buffer as
    soon as this call returns.

    The result carries no framework dependency: it exposes both
    ``__cuda_array_interface__`` and ``__dlpack__`` so Torch/JAX/CuPy can adopt
    it zero-copy, plus ``.copy_to_host()`` / ``np.asarray(...)`` for inspection.
    """
    device_str, handle_b64, storage_offset_str, _storage_size_str = val["data"][
        "buffer"
    ].split(":")
    handle_bytes = pybase64.b64decode(handle_b64, validate=True)
    device = int(device_str)
    storage_offset = int(storage_offset_str)

    dtype = np.dtype(val["dtype"])
    shape = tuple(val["shape"])
    nbytes = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize

    # Allocation and synchronization act on the active device, which need not
    # be the array's.
    with _on_device(device):
        # Get the owned buffer up front so that if any later step fails we still
        # close the IPC mapping and free the buffer cleanly.
        pooled = _DECODE_POOL.take(device, nbytes)
        if pooled is None:
            owned_ptr = cuda_api.malloc(nbytes)
        else:
            owned_ptr, _ = pooled
            # The buffer's previous array was released when its consumer dropped
            # it, but kernels the consumer queued on its own streams may still be
            # reading it, and the copy below is not ordered after them. Wait for
            # them first. This costs little because the synchronize after the copy
            # waits for the same work.
            cuda_api.device_synchronize()

        try:
            # Opening the IPC handle can fail too; if it does, we still own the
            # buffer obtained above and must free it (the except below).
            base_ptr = cuda_api.ipc_open_mem_handle(handle_bytes)
            try:
                # Copy only this array's own bytes out of the producer's (offset)
                # mapping into the owned buffer, then block until the copy is done so
                # we never unmap mid-copy.
                cuda_api.memcpy_device_to_device(
                    owned_ptr, base_ptr + storage_offset, nbytes
                )
                cuda_api.device_synchronize()
            finally:
                # Only reached once the mapping was opened; always unmap it.
                cuda_api.ipc_close_mem_handle(base_ptr)
        except Exception:
            cuda_api.free(owned_ptr)
            raise

    return IpcDeviceArray(owned_ptr, device, shape, dtype)


# ---------------------------------------------------------------------------
# Transport check
# ---------------------------------------------------------------------------
#
# Whether cuda_ipc works between two processes depends on things neither can
# see from the other side, such as whether they share a host and a GPU and how a
# container runtime isolates them. A client therefore finds out by trying,
# once per server: it exports random bytes, the server opens the handle and
# checks them, then exports the bytes reversed for the client to open and check
# in turn. Comparing contents, rather than only whether the handles open, also
# catches a handle that opens onto the wrong memory.
#
# The check can run while other calls are in flight (Tesseract-JAX may lower
# one function while a compiled one runs), so it never touches the per-request
# export registry above: releasing that would unpin another call's arrays.
# Each side keeps its own check buffers alive instead.

# Size of the buffer each side exports during a check.
_CHECK_NBYTES = 256

# Arrays the server exported in reply to recent checks. Kept alive until enough
# later checks have replaced them that the client is certainly done reading,
# even when several clients check at about the same time.
_CHECK_REPLIES: collections.deque = collections.deque(maxlen=16)


def _device_array_from_bytes(data: bytes, device: int) -> IpcDeviceArray:
    """Copy ``data`` into a fresh ``uint8`` device array owned by this process."""
    host = np.frombuffer(data, dtype=np.uint8)
    with _on_device(device):
        ptr = cuda_api.malloc(host.nbytes)
        try:
            cuda_api.memcpy_host_to_device(ptr, host.ctypes.data, host.nbytes)
            cuda_api.device_synchronize()
        except Exception:
            cuda_api.free(ptr)
            raise
    return IpcDeviceArray(ptr, device, host.shape, host.dtype)


def _export_for_check(arr: IpcDeviceArray) -> ArrayDict:
    """Export ``arr`` by IPC handle, leaving the per-request export registry alone.

    The caller keeps ``arr`` alive until the other side has read it. ``arr`` is
    a plain ``cudaMalloc`` buffer, so legacy IPC exports it without the staging
    copy, whose buffers the next request's release would recycle.
    """
    return _dump_cuda_ipc_arraydict(arr, pin=False)


def _read_check_array(payload: Any) -> bytes:
    """Open a check array exported by the other side and return its bytes."""
    if not isinstance(payload, dict) or payload.get("dtype") != "uint8":
        raise ValueError("expected a uint8 array")
    if list(payload.get("shape", ())) != [_CHECK_NBYTES]:
        raise ValueError(f"expected an array of {_CHECK_NBYTES} bytes")
    if payload.get("data", {}).get("encoding") != "cuda_ipc":
        raise ValueError("expected a cuda_ipc array")
    return load_cuda_ipc_arraydict(payload).copy_to_host().tobytes()


def start_transport_check() -> tuple[dict, bytes, IpcDeviceArray]:
    """Client side: export random bytes from the active CUDA device.

    Returns the request body to send to the server, the bytes its reply must
    hold, and the exported array, which the caller keeps alive until the server
    has answered.
    """
    data = os.urandom(_CHECK_NBYTES)
    arr = _device_array_from_bytes(data, cuda_api.get_device())
    request = {
        "gpu_transport": "cuda_ipc",
        "array": _export_for_check(arr),
        "expected": pybase64.b64encode_as_string(data),
    }
    return request, data[::-1], arr


def answer_transport_check(request: dict) -> ArrayDict:
    """Server side: check the client's bytes, then export them reversed.

    Raises if the client's handle cannot be opened or holds the wrong bytes.
    """
    data = _read_check_array(request.get("array"))
    if data != pybase64.b64decode(request.get("expected", ""), validate=True):
        raise ValueError(
            "the bytes read through the client's IPC handle differ from the ones "
            "it wrote"
        )
    device = int(request["array"]["data"]["buffer"].split(":", 1)[0])
    reply = _device_array_from_bytes(data[::-1], device)
    _CHECK_REPLIES.append(reply)
    return _export_for_check(reply)


def finish_transport_check(reply: Any, expected: bytes) -> None:
    """Client side: open the server's reply and check it holds ``expected``."""
    if _read_check_array(reply) != expected:
        raise ValueError(
            "the bytes read through the server's IPC handle differ from the ones "
            "it wrote"
        )


def validate_cuda_array(
    val: Any, expected_shape: ShapeType, expected_dtype: str | None
) -> Any:
    """Validate a GPU array's shape/dtype without pulling it off the device.

    Returns the object unchanged so it can later be encoded via CUDA IPC (see
    :func:`tesseract_core.runtime.array_encoding.encode_array`). It reads only
    the ``__cuda_array_interface__`` metadata, so no device-to-host copy or
    kernel launch occurs. Never casts, because a cast would need a device copy
    the caller did not ask for.
    """
    info = _read_cuda_array_info(val)
    check_shape_dtype_no_cast(
        info.shape,
        info.dtype.name,
        expected_shape,
        expected_dtype,
        no_cast_reason="cuda_ipc does not cast on device",
    )
    return val


class CudaIpcTransport(DeviceTransport):
    """DeviceTransport backend for the same-host ``cuda_ipc`` transport."""

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
