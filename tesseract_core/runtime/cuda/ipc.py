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
* :class:`ExportGroup` -- keeps one message's exports alive until its consumer
  is done with them,
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
    "ExportGroup",
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
    try:
        return hasattr(obj, "__cuda_array_interface__")
    except RuntimeError:
        # PyTorch raises this rather than AttributeError for a CUDA tensor that
        # requires grad, which is still a GPU array (see _without_autograd).
        return True


def _without_autograd(arr: Any) -> Any:
    """``arr``, detached if it is a PyTorch tensor that requires grad.

    Such a tensor refuses ``__cuda_array_interface__`` and ``.numpy()``.
    Encoding only reads its values, so a detached view of the same memory
    serves instead.
    """
    if getattr(arr, "requires_grad", False) and callable(getattr(arr, "detach", None)):
        return arr.detach()
    return arr


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
    arr = _without_autograd(arr)
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


def _contiguous_on_device(arr: Any) -> Any:
    """``arr``, or a row-major copy of it on the same device if it is strided.

    cuda_ipc moves a flat byte range, so a strided GPU array is first copied on
    the device by the framework that made it: PyTorch's ``.contiguous()`` or
    CuPy's ``.copy(order="C")``. An array that offers neither is rejected rather
    than copied through the host, so GPU data never silently leaves the device.
    """
    info = _read_cuda_array_info(arr)
    if info.is_c_contiguous():
        return arr
    for make_copy in (lambda: arr.contiguous(), lambda: arr.copy(order="C")):
        try:
            copy = make_copy()
        except (AttributeError, TypeError):
            continue
        if (
            has_cuda_array_interface(copy)
            and _read_cuda_array_info(copy).is_c_contiguous()
        ):
            return copy
    raise ValueError(
        "cuda_ipc encoding requires a C-contiguous array; got one with "
        f"strides {info.strides}, and {type(arr).__name__} offers no way to copy "
        "it on the device (.contiguous() or .copy(order='C')). Make a "
        "contiguous copy first."
    )


def _read_cuda_array_info(arr: Any) -> _CudaArrayInfo:
    """Read a CUDA array's metadata from ``__cuda_array_interface__``.

    The protocol carries no device ordinal, so it is read from the framework's
    ``.device`` attribute and defaults to 0.
    """
    arr = _without_autograd(arr)
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
#     group until the client acknowledges the response (see serve.py).
#
# Callers that pass no group use a process-wide one, released by
# :func:`release_pinned_ipc_exports`, which is safe only if exports never
# overlap in time.


class ExportGroup:
    """The arrays one message exported via CUDA IPC, kept alive until :meth:`release`.

    Also holds the staging buffers they were copied into (see
    :func:`_stage_for_export`), which :meth:`release` returns to the pool.
    """

    def __init__(self) -> None:
        self.pins: list[Any] = []
        # Staging buffers as (device pointer, device, size, IPC handle).
        self.staging: list[tuple[int, int, int, bytes]] = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.pins) + len(self.staging)

    def pin(self, arr: Any) -> None:
        """Keep ``arr``, so its memory stays valid, until :meth:`release`."""
        with self._lock:
            self.pins.append(arr)

    def add_staging(self, ptr: int, device: int, nbytes: int, handle: bytes) -> None:
        """Keep a staging buffer until :meth:`release` returns it to the pool."""
        with self._lock:
            self.staging.append((ptr, device, nbytes, handle))

    def release(self) -> None:
        """Let go of the pinned arrays and return the staging buffers to the pool."""
        with self._lock:
            self.pins.clear()
            staging, self.staging[:] = list(self.staging), []
        for ptr, device, nbytes, handle in staging:
            _STAGING_POOL.put(ptr, device, nbytes, handle)


# The group for callers that pass none.
_DEFAULT_EXPORTS = ExportGroup()


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
        # Buffers handed back by release() and not yet added to the pool.
        self._released: collections.deque[tuple[int, int, int]] = collections.deque()

    def release(self, ptr: int, device: int, nbytes: int) -> None:
        """Hand back a buffer from a finalizer, which may run at any point.

        A garbage-collection pass can run a finalizer while this thread holds
        the pool's lock inside :meth:`take` or :meth:`put`, so this never waits
        for the lock. It queues the buffer, which joins the pool now if the lock
        is free and otherwise at the next :meth:`take` or :meth:`put`.
        Finalizers must never raise, so neither does this.
        """
        try:
            if not ptr:
                return
            self._released.append((ptr, device, nbytes))
            if self._lock.acquire(blocking=False):
                self._lock.release()
                self._drain()
        except Exception:  # noqa: BLE001, S110
            pass

    def _drain(self) -> None:
        """Add the buffers queued by :meth:`release` to the pool."""
        while True:
            try:
                ptr, device, nbytes = self._released.popleft()
            except IndexError:
                return
            self._put(ptr, device, nbytes, None)

    def take(self, device: int, nbytes: int) -> tuple[int, Any] | None:
        """Remove and return ``(ptr, value)`` for an idle buffer of this device and size, or ``None``."""
        self._drain()
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
        self._drain()
        self._put(ptr, device, nbytes, value)

    def _put(self, ptr: int, device: int, nbytes: int, value: Any) -> None:
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


# Idle staging buffers, each with its IPC handle, so reuse also saves the
# cudaIpcGetMemHandle call.
_STAGING_POOL = _BufferPool()


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
    group.add_staging(ptr, device, nbytes, handle)
    return handle


def release_pinned_ipc_exports() -> None:
    """Release the exports of callers that passed no :class:`ExportGroup`."""
    _DEFAULT_EXPORTS.release()


@contextlib.contextmanager
def _on_device(device: int) -> Iterator[None]:
    """Make ``device`` the active CUDA device for the block, then restore the caller's."""
    previous = cuda_api.get_device()
    # Set it even when it is already active, because cudaSetDevice also makes
    # the device's primary context current on this thread, which driver calls
    # in the block need (a fresh thread has none).
    cuda_api.set_device(device)
    if previous == device:
        yield
        return
    try:
        yield
    finally:
        cuda_api.set_device(previous)


def dump_cuda_ipc_arraydict(arr: Any, group: ExportGroup | None = None) -> ArrayDict:
    """Dump a CUDA array to a JSON dict with a CUDA IPC handle.

    Works with any object that implements ``__cuda_array_interface__``, such
    as CuPy arrays, PyTorch tensors, and single-device JAX arrays.

    The IPC handle allows another process on the same host (with --ipc=host)
    to access the GPU memory directly without any CPU round-trip.

    The source array stays pinned in ``group`` (by default the process-wide
    one, see :func:`release_pinned_ipc_exports`) until the group is released,
    so its GPU memory is not freed or recycled before the consumer copies it out.

    Frameworks with VMM/pool-backed GPU allocators (e.g. JAX/XLA) hand out
    pointers that the legacy ``cudaIpcGetMemHandle`` API rejects. For those,
    encode stages the array's bytes into a reusable ``cudaMalloc`` buffer with
    one on-GPU copy (see :func:`_stage_for_export`) and exports a handle to that
    instead, which is still far cheaper than a host round-trip.
    """
    if group is None:
        group = _DEFAULT_EXPORTS
    if not has_cuda_array_interface(arr):
        raise ValueError(
            "cuda_ipc encoding requires a CUDA array "
            f"(object with __cuda_array_interface__), got {type(arr).__name__}"
        )

    arr = _contiguous_on_device(_without_autograd(arr))
    info = _read_cuda_array_info(arr)

    if info.dtype.kind == "V":
        raise TypeError(
            f"cuda_ipc cannot encode arrays of dtype {info.dtype}, which have no "
            "NumPy equivalent (e.g. bfloat16)."
        )

    data_ptr, nbytes = info.data_ptr, info.nbytes

    if nbytes == 0:
        # An empty array has no memory to share (frameworks give it a null
        # pointer, which the CUDA calls below reject), so its descriptor
        # carries no handle and the consumer recreates it without opening one.
        return {
            "object_type": "array",
            "shape": list(info.shape),
            "dtype": info.dtype.name,
            "data": {"buffer": f"{info.device}::0:0", "encoding": "cuda_ipc"},
        }

    # Keep the source allocation alive until exports are explicitly released.
    # (Not strictly needed on the VMM fallback path below, which copies out of
    # `arr` before the handle leaves the process, but keeping the pin gives both
    # paths identical cleanup.)
    group.pin(arr)

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

    The device buffer returns to the decode pool once this object is garbage
    collected. Every DLPack capsule it hands out, and the tensor a framework
    adopts from one, keeps it alive until the framework releases the tensor.
    A framework adopting it through ``__cuda_array_interface__`` keeps a
    reference to it for the same reason.
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
        self._finalizer = weakref.finalize(
            self, _DECODE_POOL.release, ptr, device, self._nbytes
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
        host = np.empty(self.shape, dtype=self.dtype)
        if self._nbytes == 0:
            return host
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
        """Return a new ``"dltensor"`` PyCapsule onto the owned buffer.

        The capsule keeps this object alive until the consumer releases the tensor.
        """
        return dlpack.make_dlpack_capsule(
            self._ptr, self.device, self.shape, self.dtype, owner=self
        )


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


def load_cuda_ipc_arraydict(val: ArrayDict) -> "IpcDeviceArray":
    """Load a CUDA array from a JSON dict with a CUDA IPC handle.

    The calling process must share the IPC namespace with the producer
    (e.g. both run with --ipc=host on Docker) and see the same GPU.

    Returns a caller-owned :class:`IpcDeviceArray`. Decoding maps the IPC handle
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
    device_str, handle_b64, storage_offset_str, _storage_size_str = val["data"][
        "buffer"
    ].split(":")
    handle_bytes = pybase64.b64decode(handle_b64, validate=True)
    device = int(device_str)
    storage_offset = int(storage_offset_str)

    dtype = np.dtype(val["dtype"])
    shape = tuple(val["shape"])
    nbytes = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize

    if not handle_bytes:
        # An empty array's descriptor carries no handle (see the encoder).
        if nbytes:
            raise ValueError(
                f"cuda_ipc descriptor has no handle, but the array of shape "
                f"{shape} and dtype {dtype} is not empty"
            )
        return IpcDeviceArray(0, device, shape, dtype)

    # Allocation and synchronization act on the active device, which need not
    # be the array's.
    with _on_device(device):
        # Free the buffers of decoded arrays whose only remaining reference was
        # a DLPack capsule nobody consumed, so the pool can offer them below.
        dlpack.drop_abandoned_capsules()
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

    return IpcDeviceArray(owned_ptr, device, shape, dtype)


# ---------------------------------------------------------------------------
# Transport check
# ---------------------------------------------------------------------------
#
# Whether cuda_ipc works between two processes depends on things neither can
# see from the other side, such as whether they share a host and a GPU and how a
# container runtime isolates them. A client therefore finds out by trying,
# once per server. It exports random bytes, the server opens the handle and
# checks them, then exports the bytes reversed for the client to open and check
# in turn. Comparing contents also catches a handle that opens onto the wrong
# memory.
#
# The check can run while other calls are in flight (Tesseract-JAX may lower
# one function while a compiled one runs), so its arrays stay out of the export
# groups above, and each side keeps its own check buffers alive instead.

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
    """Export ``arr`` by IPC handle, outside every request's export group.

    The caller keeps ``arr`` alive until the other side has read it. ``arr`` is
    a plain ``cudaMalloc`` buffer, which legacy IPC exports without staging, so
    the throwaway group holds nothing that needs releasing.
    """
    return dump_cuda_ipc_arraydict(arr, ExportGroup())


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

    def new_exports(self) -> ExportGroup:
        """A fresh :class:`ExportGroup` for one message's exports."""
        return ExportGroup()

    def register(self, arr: Any, session: Any = None) -> ArrayDict:
        """Pin ``arr`` in ``session`` (an :class:`ExportGroup`) and return its IPC descriptor.

        cuda_ipc mints the handle and packs the wire string in one call, so the
        per-array handle *is* the array dict and :meth:`descriptor` is a
        passthrough. Splitting them would take the IPC handle twice for nothing.
        """
        return dump_cuda_ipc_arraydict(arr, group=session)

    def descriptor(self, handle: ArrayDict) -> ArrayDict:
        """Return the array dict :meth:`register` already produced."""
        return handle

    def flush(self, session: Any = None) -> None:
        """No-op: cuda_ipc is receiver-driven, so there is nothing to post."""

    def receive(self, val: ArrayDict, session: Any = None) -> "IpcDeviceArray":
        """Copy the exported bytes into a fresh consumer-owned ``IpcDeviceArray``."""
        return load_cuda_ipc_arraydict(val)

    def release(self, session: Any = None) -> None:
        """Drop the producer-side pins of ``session``, or of callers that passed none."""
        if session is None:
            release_pinned_ipc_exports()
        else:
            session.release()
