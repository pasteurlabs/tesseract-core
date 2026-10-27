# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU array handling shared by the GPU transports.

This covers reading a GPU array's metadata from ``__cuda_array_interface__``,
preparing it for export, copying it to the host for host encodings, and the
pooled :class:`DeviceArray` that :mod:`~tesseract_core.runtime.cuda.ipc` and
:mod:`~tesseract_core.runtime.cuda.nixl` decode into.
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

from tesseract_core.runtime.array_encoding import ShapeType, check_shape_dtype_no_cast
from tesseract_core.runtime.config import get_config
from tesseract_core.runtime.cuda import api as cuda_api
from tesseract_core.runtime.cuda import dlpack

__all__ = [
    "FORBID_DEVICE_HOST_COPY_ENV",
    "BufferPool",
    "CudaArrayInfo",
    "DeviceArray",
    "check_device_host_copy",
    "contiguous_on_device",
    "cuda_array_to_host",
    "device_array_from_bytes",
    "has_cuda_array_interface",
    "on_device",
    "prepare_export",
    "take_owned_buffer",
    "validate_cuda_array",
    "without_autograd",
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
        # requires grad, which is still a GPU array (see without_autograd).
        return True


def without_autograd(arr: Any) -> Any:
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
    transport and ``np.asarray`` on a :class:`DeviceArray`. Explicit copies
    via :meth:`DeviceArray.copy_to_host` are never blocked.
    """
    if os.environ.get(FORBID_DEVICE_HOST_COPY_ENV, "").lower() in {"1", "true"}:
        raise RuntimeError(
            f"Implicit device-to-host copy of {what} "
            f"({FORBID_DEVICE_HOST_COPY_ENV} is set)."
        )


def cuda_array_to_host(arr: Any) -> np.ndarray:
    """Copy a GPU array to a host NumPy array.

    Used for host encodings, where the bytes must reach the host, and so
    subject to :func:`check_device_host_copy`. Handles CuPy (``.get()``) and
    PyTorch (``.cpu().numpy()``) explicitly, then falls back to ``np.asarray``
    for any other framework whose arrays support ``__array__`` (e.g. JAX, which
    fetches to host on conversion).
    """
    check_device_host_copy(f"a {type(arr).__name__} GPU array")
    arr = without_autograd(arr)
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


class CudaArrayInfo(NamedTuple):
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

        The GPU transports move ``nbytes`` consecutive bytes and the decoder
        rebuilds the array from shape and dtype alone, so callers must reject
        non-contiguous sources, which would otherwise be silently misread.
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


def contiguous_on_device(arr: Any) -> tuple[Any, CudaArrayInfo]:
    """``arr``, or a row-major copy of it on the same device if it is strided, with its info.

    The GPU transports move a flat byte range, so a strided GPU array is first
    copied on the device by the framework that made it, with PyTorch's
    ``.contiguous()`` or CuPy's ``.copy(order="C")``. An array that offers
    neither is rejected rather than copied through the host, so GPU data never
    silently leaves the device.
    """
    info = _read_cuda_array_info(arr)
    if info.is_c_contiguous():
        return arr, info
    for make_copy in (lambda: arr.contiguous(), lambda: arr.copy(order="C")):
        try:
            copy = make_copy()
        except (AttributeError, TypeError):
            continue
        if has_cuda_array_interface(copy):
            copy_info = _read_cuda_array_info(copy)
            if copy_info.is_c_contiguous():
                return copy, copy_info
    raise ValueError(
        "Passing a GPU array by reference requires a C-contiguous array; got one with "
        f"strides {info.strides}, and {type(arr).__name__} offers no way to copy "
        "it on the device (.contiguous() or .copy(order='C')). Make a "
        "contiguous copy first."
    )


def _read_cuda_array_info(arr: Any) -> CudaArrayInfo:
    """Read a CUDA array's metadata from ``__cuda_array_interface__``.

    The protocol carries no device ordinal, so it is read from the framework's
    ``.device`` attribute and defaults to 0.
    """
    arr = without_autograd(arr)
    iface = arr.__cuda_array_interface__
    shape = tuple(iface["shape"])
    dtype = np.dtype(iface["typestr"])  # e.g. "<f4", "|b1"
    strides = iface.get("strides")

    device = 0
    dev = getattr(arr, "device", None)
    if isinstance(dev, int):
        device = dev  # DeviceArray
    elif hasattr(dev, "id"):
        device = dev.id  # CuPy
    elif getattr(dev, "index", None) is not None:
        device = dev.index  # PyTorch

    return CudaArrayInfo(
        data_ptr=iface["data"][0],
        nbytes=math.prod(shape) * dtype.itemsize,
        shape=shape,
        dtype=dtype,
        device=device,
        strides=None if strides is None else tuple(strides),
    )


def validate_cuda_array(
    val: Any, expected_shape: ShapeType, expected_dtype: str | None
) -> Any:
    """Validate a GPU array's shape/dtype without pulling it off the device.

    Returns the object unchanged so a GPU transport can later encode it (see
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
        no_cast_reason="GPU arrays are not cast on device",
    )
    return val


def prepare_export(arr: Any, transport: str) -> tuple[Any, CudaArrayInfo]:
    """Return ``arr`` as a contiguous CUDA array with its info, or raise if it cannot be exported.

    ``transport`` names the transport in error messages. A strided array is
    first copied on the device (see :func:`contiguous_on_device`).
    """
    if not has_cuda_array_interface(arr):
        raise ValueError(
            f"{transport} encoding requires a CUDA array "
            f"(object with __cuda_array_interface__), got {type(arr).__name__}"
        )

    arr, info = contiguous_on_device(without_autograd(arr))

    if info.dtype.kind == "V":
        raise TypeError(
            f"{transport} cannot encode arrays of dtype {info.dtype}, which have no "
            "NumPy equivalent (e.g. bfloat16)."
        )
    return arr, info


class BufferPool:
    """Idle ``cudaMalloc`` buffers kept for reuse, keyed by device and size.

    Iterative callers send the same shapes on every request, so without a pool
    both sides of a GPU transport exchange would free and reallocate the same
    large buffers each time. When two processes share a GPU, a ``cudaMalloc`` that
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
    """Cap on a :class:`BufferPool`'s idle bytes on ``device``, set by ``cuda_ipc_pool_fraction``."""
    fraction = get_config().cuda_ipc_pool_fraction
    return int(cuda_api.device_total_memory(device) * fraction / 2)


@contextlib.contextmanager
def on_device(device: int) -> Iterator[None]:
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


# ---------------------------------------------------------------------------
# Decoded device-array wrapper
# ---------------------------------------------------------------------------
#
# The decode path returns an object (:class:`DeviceArray`) that owns a device
# buffer and exposes it via both ``__cuda_array_interface__`` and DLPack, so
# Torch/JAX/CuPy can all adopt it zero-copy without CuPy being a decode-time
# dependency. The DLPack ABI machinery lives in
# :mod:`tesseract_core.runtime.cuda.dlpack`.

# Idle owned buffers of decoded arrays. An array's buffer returns here once the
# array (or the framework tensor that adopted it) is released, and the next
# decode of the same size reuses it.
_DECODE_POOL = BufferPool()


class DeviceArray:
    """Owns a device buffer a GPU transport decoded an array into.

    The buffer is a process-owned ``cudaMalloc`` allocation holding the array's
    own bytes, which the transport copied out of the producer's memory. The
    object is framework-agnostic:

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
        self._nbytes = math.prod(self.shape) * self.dtype.itemsize
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
        check_device_host_copy("a DeviceArray")
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


def take_owned_buffer(device: int, nbytes: int) -> int:
    """A buffer of ``nbytes`` on ``device`` to decode an array into.

    Reuses a released buffer of the same size if there is one, and allocates
    one otherwise. ``device`` must be the active device. The caller frees the
    buffer if decoding fails, and otherwise hands it to a :class:`DeviceArray`,
    which returns it to the pool.
    """
    # Free the buffers of decoded arrays whose only remaining reference was a
    # DLPack capsule nobody consumed, so the pool can offer them below.
    dlpack.drop_abandoned_capsules()
    pooled = _DECODE_POOL.take(device, nbytes)
    if pooled is None:
        return cuda_api.malloc(nbytes)
    # The buffer's previous array was released when its consumer dropped it,
    # but kernels the consumer queued on its own streams may still be reading
    # it, and the transport's copy into it is not ordered after them. Wait for
    # them first. This costs little for cuda_ipc, which synchronizes after its
    # copy anyway and so waits for the same work.
    cuda_api.device_synchronize()
    return pooled[0]


def device_array_from_bytes(data: bytes, device: int) -> DeviceArray:
    """Copy ``data`` into a fresh ``uint8`` device array owned by this process."""
    host = np.frombuffer(data, dtype=np.uint8)
    with on_device(device):
        ptr = cuda_api.malloc(host.nbytes)
        try:
            cuda_api.memcpy_host_to_device(ptr, host.ctypes.data, host.nbytes)
            cuda_api.device_synchronize()
        except Exception:
            cuda_api.free(ptr)
            raise
    return DeviceArray(ptr, device, host.shape, host.dtype)
