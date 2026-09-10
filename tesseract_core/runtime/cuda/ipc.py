# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA IPC array encoding: zero-copy GPU array exchange between processes.

This module holds everything specific to the ``json+cuda_ipc`` encoding, kept
separate from the framework-agnostic host encodings in
:mod:`tesseract_core.runtime.array_encoding`. Nothing here is imported unless a
Tesseract actually encodes or decodes a CUDA IPC array, so the CUDA runtime and
driver libraries are only touched on that path.

All low-level CUDA access lives in the :mod:`tesseract_core.runtime.cuda`
package: this module works purely with plain Python values (device pointers as
``int``, IPC handles as ``bytes``) and never imports ctypes. It contributes only
the *encoding policy* -- how a GPU array maps to and from the ``cuda_ipc`` JSON
payload, plus the keepalive bookkeeping that the transfer protocol requires.

The JSON schema for this encoding (``CudaIpcArrayData``) lives alongside the
other array-data models in :mod:`array_encoding`; the public entry points used
by :mod:`array_encoding` are:

* :func:`has_cuda_array_interface` / :func:`cuda_array_to_host` -- host-copy
  helpers for non-IPC encodings of GPU arrays,
* :func:`validate_cuda_array` -- shape/dtype validation without a device copy,
* :func:`dump_cuda_ipc_arraydict` / :func:`load_cuda_ipc_arraydict` -- the
  encode/decode pair,
* :func:`release_pinned_ipc_exports` -- keepalive cleanup, called once per
  request by both the server (for its outputs) and the client (for its inputs).

Out-of-process consumers that ``dlopen`` libcudart themselves (e.g. the
``tesseract_jax`` C++ FFI shim) can reuse the library discovery -- including the
pip-wheel fallback and forward-compatible version range -- via
:func:`tesseract_core.runtime.cuda.iter_cudart_candidates` instead of
maintaining their own soname list.
"""

import weakref
from typing import Any, get_args

import numpy as np
import pybase64
from pydantic_core import PydanticCustomError

from tesseract_core.runtime.array_encoding import AllowedDtypes, ArrayDict, ShapeType
from tesseract_core.runtime.cuda import api as cuda_api
from tesseract_core.runtime.cuda import dlpack
from tesseract_core.runtime.device_transport import DeviceTransport

__all__ = [
    "IpcDeviceArray",
    "cuda_array_to_host",
    "dump_cuda_ipc_arraydict",
    "has_cuda_array_interface",
    "load_cuda_ipc_arraydict",
    "release_pinned_ipc_exports",
    "validate_cuda_array",
]


def has_cuda_array_interface(obj: Any) -> bool:
    """Check if an object exposes the __cuda_array_interface__ protocol.

    This protocol is supported by PyTorch, CuPy, JAX, Numba, and any
    CUDA-aware Python library. It indicates the object holds data in
    GPU device memory.
    """
    return hasattr(obj, "__cuda_array_interface__")


def cuda_array_to_host(arr: Any) -> np.ndarray:
    """Copy a GPU array to a host NumPy array (explicit device-to-host copy).

    Used for non-IPC encodings, where the bytes must reach the host. Handles
    CuPy (``.get()``) and PyTorch (``.cpu().numpy()``) explicitly, then falls
    back to ``np.asarray`` for any other framework whose arrays support
    ``__array__`` (e.g. JAX, which fetches to host on conversion).
    """
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


def _get_cuda_array_info(arr: Any) -> tuple[int, int, tuple[int, ...], str]:
    """Extract (device_ptr, nbytes, shape, numpy_dtype_str) from a CUDA array.

    Works with any object that implements __cuda_array_interface__ (v2+):
    PyTorch tensors, CuPy arrays, JAX DeviceArrays, Numba device arrays, etc.
    """
    iface = arr.__cuda_array_interface__
    data_ptr = iface["data"][0]
    shape = tuple(iface["shape"])
    typestr = iface["typestr"]  # e.g. "<f4", "|b1"
    dtype = np.dtype(typestr)
    nbytes = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
    return data_ptr, nbytes, shape, dtype.name


def _is_c_contiguous(arr: Any) -> bool:
    """Whether a CUDA array's memory is C-contiguous per __cuda_array_interface__.

    cuda_ipc transfers a flat, contiguous byte range: encode copies (or hands
    off) ``prod(shape) * itemsize`` consecutive bytes and decode rebuilds a
    contiguous array from shape/dtype alone (the payload carries no strides). A
    non-contiguous source would therefore be silently misread, so callers must
    reject it.

    Per the protocol, ``strides = None`` means row-major contiguous. A non-None
    ``strides`` is still contiguous iff it equals the row-major strides implied
    by shape and itemsize.
    """
    iface = arr.__cuda_array_interface__
    strides = iface.get("strides")
    if strides is None:
        return True
    shape = tuple(iface["shape"])
    itemsize = np.dtype(iface["typestr"]).itemsize
    expected = []
    acc = itemsize
    for dim in reversed(shape):
        expected.append(acc)
        acc *= dim
    expected.reverse()
    return tuple(strides) == tuple(expected)


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
#     client's inputs *during* request handling -- :func:`load_cuda_ipc_arraydict`
#     copies each input into server-owned memory and closes the mapping before
#     the response is sent -- so by the time the HTTP call returns (with the body
#     buffered) the inputs are provably dead and can be released immediately.
#
# Both rely on the same two assumptions:
#   1. Requests are issued *serially* (never concurrently).
#   2. The consumer copies decoded arrays into consumer-owned memory before it
#      releases the exporter's buffer (which the decode path does
#      unconditionally; see :func:`load_cuda_ipc_arraydict`).
_CUDA_IPC_EXPORT_REGISTRY: list[Any] = []

# Device pointers of VMM-fallback staging buffers (see runtime.stage_for_legacy_ipc)
# awaiting free. Kept separate from _CUDA_IPC_EXPORT_REGISTRY (which holds plain
# pinned array references) since these need an explicit free call instead of just
# dropping a reference, but are released at the same point and for the same
# reasons.
_CUDA_IPC_STAGING_BUFFERS: list[int] = []


def release_pinned_ipc_exports() -> None:
    """Release arrays pinned for CUDA IPC export and free their staging buffers.

    Drops the keepalive references held since the last release and frees any
    VMM-fallback staging buffers. Driven by both sides of a cuda_ipc exchange
    but at different points in the request lifecycle (server: start of the next
    request; client: end of the current request); see the registry comment above
    for why each timing is safe.
    """
    _CUDA_IPC_EXPORT_REGISTRY.clear()

    staging_ptrs, _CUDA_IPC_STAGING_BUFFERS[:] = list(_CUDA_IPC_STAGING_BUFFERS), []
    for ptr in staging_ptrs:
        cuda_api.free(ptr)


def _pin_cuda_ipc_export(arr: Any) -> None:
    """Retain a reference to a source array so its GPU memory stays valid.

    The reference is held until the exporting side calls
    :func:`release_pinned_ipc_exports`.
    """
    _CUDA_IPC_EXPORT_REGISTRY.append(arr)


def _pin_cuda_ipc_staging_buffer(device_ptr: int) -> None:
    """Register a VMM-fallback staging buffer for a later free.

    Freed at the same point ordinary pinned arrays are released (when the
    exporting side calls :func:`release_pinned_ipc_exports`), for the same
    reasons (see the module comment above).
    """
    _CUDA_IPC_STAGING_BUFFERS.append(device_ptr)


def dump_cuda_ipc_arraydict(arr: Any) -> ArrayDict:
    """Dump a CUDA array to a JSON dict with a CUDA IPC handle.

    Works with any object that implements __cuda_array_interface__:
    PyTorch tensors, CuPy arrays, JAX DeviceArrays, etc.

    The IPC handle allows another process on the same host (with --ipc=host)
    to access the GPU memory directly without any CPU round-trip.

    The source array is pinned in a process-global registry (see
    :func:`_pin_cuda_ipc_export`) so its GPU memory is not freed or recycled
    before the consumer copies it out; the pin is released once the exporting
    side calls :func:`release_pinned_ipc_exports`.

    Frameworks with VMM/pool-backed GPU allocators (e.g. JAX/XLA) hand out
    pointers that the legacy ``cudaIpcGetMemHandle`` API rejects; in that case
    this transparently falls back to staging the array's bytes into a fresh
    ``cudaMalloc`` buffer via one on-GPU copy (see
    :func:`tesseract_core.runtime.cuda.api.stage_for_legacy_ipc`) and
    exports a handle to that instead. Still far cheaper than a host round-trip.
    """
    if not has_cuda_array_interface(arr):
        raise ValueError(
            "cuda_ipc encoding requires a CUDA array "
            f"(object with __cuda_array_interface__), got {type(arr).__name__}"
        )

    if not _is_c_contiguous(arr):
        raise ValueError(
            "cuda_ipc encoding requires a C-contiguous array; got one with "
            f"strides {arr.__cuda_array_interface__.get('strides')}. Make a "
            "contiguous copy first (e.g. cupy.ascontiguousarray / "
            "torch.Tensor.contiguous)."
        )

    data_ptr, nbytes, shape, dtype_name = _get_cuda_array_info(arr)

    # Keep the source allocation alive until exports are explicitly released.
    # (Still needed even on the VMM fallback path below: stage_for_legacy_ipc
    # reads from `arr`'s memory synchronously before returning, but keeping the
    # pin simplifies the two paths to an identical cleanup story.)
    _pin_cuda_ipc_export(arr)

    # IPC handles reference the *whole* backing allocation, not the array's
    # (possibly offset) data pointer. Pooled allocators (CuPy, PyTorch) hand
    # out many arrays from a single cudaMalloc block, so we must resolve the
    # allocation base, take the handle on that base, and record the byte offset
    # of this array within the allocation.
    base_ptr, storage_size = cuda_api.get_allocation_base(data_ptr)
    storage_offset = data_ptr - base_ptr

    # Get the IPC handle for the base of the allocation.
    try:
        handle_bytes = cuda_api.ipc_get_mem_handle(base_ptr)
    except RuntimeError:
        # Legacy IPC rejected this pointer, almost certainly because it's
        # VMM/pool-backed. Copy just this array's own bytes (not the whole,
        # possibly huge, backing allocation) into a fresh cudaMalloc buffer and
        # export a handle to *that* instead.
        staging_ptr = cuda_api.stage_for_legacy_ipc(data_ptr, nbytes)
        _pin_cuda_ipc_staging_buffer(staging_ptr)
        storage_offset = 0
        storage_size = nbytes
        handle_bytes = cuda_api.ipc_get_mem_handle(staging_ptr)

    # Determine device ordinal
    device = 0
    # CuPy arrays expose .device.id, torch tensors expose .device.index
    if hasattr(arr, "device"):
        dev = arr.device
        if hasattr(dev, "id"):
            device = dev.id  # CuPy
        elif hasattr(dev, "index") and dev.index is not None:
            device = dev.index  # PyTorch

    handle_b64 = pybase64.b64encode_as_string(handle_bytes)
    return {
        "object_type": "array",
        "shape": list(shape),
        "dtype": dtype_name,
        "data": {
            "buffer": f"{device}:{handle_b64}:{storage_offset}:{storage_size}",
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


def _finalize_ipc_device_array(state: dict) -> None:
    """Release an :class:`IpcDeviceArray`'s device buffer exactly once.

    Registered via :func:`weakref.finalize`, so it runs when the array is
    garbage-collected *and* at interpreter shutdown, and can fire at most once.
    ``state`` is the array's mutable ownership record, shared by reference with
    the live object so ``__dlpack__`` can hand ownership off before this runs:

    * ``dlpack_token is None`` and not ``freed``: we still own the buffer, so
      free it.
    * ``dlpack_token`` set: ``__dlpack__`` moved the buffer into a DLPack bundle;
      drop the bundle iff its capsule was never consumed (a consumer that took
      the capsule already owns the free).
    """
    try:
        if state["dlpack_token"] is None:
            if not state["freed"]:
                cuda_api.free(state["ptr"])
                state["freed"] = True
        else:
            dlpack.drop_unconsumed_bundle(state["dlpack_token"])
    except Exception:
        # Finalizers must never raise.
        pass


class IpcDeviceArray:
    """Owns a device buffer decoded from a CUDA IPC handle.

    The buffer is a fresh, process-owned ``cudaMalloc`` allocation holding the
    array's own bytes (the producer's IPC mapping is copied into it and then
    closed by the decoder). The object is framework-agnostic:

    * ``__cuda_array_interface__`` (v3) lets CuPy / Numba / PyTorch adopt it,
    * ``__dlpack__`` / ``__dlpack_device__`` let Torch and JAX adopt it,

    both zero-copy. ``.copy_to_host()`` / ``np.asarray(...)`` materialise a host
    NumPy copy so it can be inspected without any GPU framework installed.

    Ownership of the device buffer is released exactly once: either a
    :func:`weakref.finalize` callback frees it (see
    :func:`_finalize_ipc_device_array`), or a DLPack consumer takes it (the
    capsule is renamed to ``"used_dltensor"`` on consumption, transferring the
    free to the consumer's deleter). ``_state["freed"]`` guards against a double
    free.
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
        #   freed:        True once the buffer is freed or ownership was handed
        #                 to a DLPack capsule; prevents a double free.
        #   dlpack_token: token of the DLPack bundle produced by __dlpack__, or
        #                 None if __dlpack__ was never called. Ownership of the
        #                 buffer moves into that bundle when it is created.
        self._state: dict = {"ptr": ptr, "freed": False, "dlpack_token": None}
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
        if self._state["freed"]:
            raise RuntimeError("device buffer has been released")
        host = np.empty(self.shape, dtype=self.dtype)
        cuda_api.memcpy_device_to_host(host.ctypes.data, self._ptr, self._nbytes)
        cuda_api.device_synchronize()
        return host

    def __array__(self, dtype: Any = None) -> np.ndarray:
        host = self.copy_to_host()
        return host if dtype is None else host.astype(dtype)

    # -- DLPack ----------------------------------------------------------

    def __dlpack_device__(self) -> tuple[int, int]:
        return (dlpack.DLDEVICE_CUDA, self.device)

    def __dlpack__(self, stream: Any = None, **kwargs: Any) -> Any:
        """Return a ``"dltensor"`` PyCapsule wrapping the owned buffer.

        Ownership of the device buffer moves into a self-contained DLPack bundle
        (see :func:`tesseract_core.runtime.cuda.dlpack.make_dlpack_capsule`)
        whose deleter frees it. The bundle's lifetime is deliberately *not* tied
        to this object's, because a consumer (Torch/JAX) may keep the tensor long
        after this ``IpcDeviceArray`` is gone and will call the deleter then.
        Whoever ends up owning the capsule (the consumer, or the finalizer for an
        un-consumed capsule) frees the buffer exactly once.
        """
        if self._state["freed"]:
            raise RuntimeError("device buffer has been released")

        capsule, token = dlpack.make_dlpack_capsule(
            self._ptr, self.device, self.shape, self.dtype
        )
        # The buffer now belongs to the bundle; this object must not free it.
        # The finalizer reads this shared state to drop the bundle iff its
        # capsule is never consumed.
        self._state["dlpack_token"] = token
        self._state["freed"] = True
        return capsule


def load_cuda_ipc_arraydict(val: ArrayDict) -> "IpcDeviceArray":
    """Load a CUDA array from a JSON dict with a CUDA IPC handle.

    The calling process must share the IPC namespace with the producer
    (e.g. both run with --ipc=host on Docker) and see the same GPU.

    Returns a freshly-allocated, caller-owned :class:`IpcDeviceArray`: the IPC
    handle is opened, the array's own bytes are copied device-to-device into a
    ``cudaMalloc`` buffer owned by this process, the copy is synchronised, and
    the IPC mapping is closed before returning. The borrow of the producer's
    memory therefore lasts only for a single on-GPU copy, so the producer is
    free to reuse or release the exported buffer as soon as this call returns.

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

    # Allocate the owned buffer up front (on the target device) so that if any
    # later step fails we still close the IPC mapping and free the buffer cleanly.
    cuda_api.set_device(device)
    owned_ptr = cuda_api.malloc(nbytes)

    try:
        # Opening the IPC handle can fail too; if it does, we still own the
        # buffer allocated above and must free it (the except below).
        base_ptr = cuda_api.ipc_open_mem_handle(handle_bytes, device)
        try:
            # Copy only this array's own bytes out of the producer's (offset)
            # mapping into our fresh buffer, then block until the copy is done so
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


def validate_cuda_array(
    val: Any, expected_shape: ShapeType, expected_dtype: str | None
) -> Any:
    """Validate a GPU array's shape/dtype without pulling it off the device.

    Returns the object unchanged so it can later be encoded via CUDA IPC (see
    :func:`tesseract_core.runtime.array_encoding.encode_array`). Only the
    ``__cuda_array_interface__`` metadata is inspected -- no device-to-host copy
    or kernel launch occurs. Mirrors the shape/dtype checks in
    :func:`tesseract_core.runtime.array_encoding._coerce_shape_dtype`, but never
    casts (a cast would need a device copy the caller did not ask for).
    """
    _, _, shape, dtype_name = _get_cuda_array_info(val)

    # Shape: Ellipsis means "no check"; otherwise each dim must match unless the
    # expected dim is None (a polymorphic wildcard).
    if expected_shape is not Ellipsis:
        if len(shape) != len(expected_shape) or any(
            exp is not None and got != exp
            for got, exp in zip(shape, expected_shape, strict=False)
        ):
            raise PydanticCustomError(
                "array_shape_mismatch",
                "Array shape {actual_shape} is incompatible with expected "
                "shape {expected_shape}",
                {"actual_shape": shape, "expected_shape": tuple(expected_shape)},
            )

    allowed_dtypes = [dtype.lower() for dtype in get_args(AllowedDtypes)]
    if dtype_name not in allowed_dtypes:
        raise PydanticCustomError(
            "array_invalid_dtype",
            "Array has unsupported dtype '{actual_dtype}'; must be one of: "
            "{allowed_dtypes}",
            {"actual_dtype": dtype_name, "allowed_dtypes": ", ".join(allowed_dtypes)},
        )

    if expected_dtype is not None and dtype_name != expected_dtype:
        raise PydanticCustomError(
            "array_dtype_mismatch",
            "GPU array dtype '{actual_dtype}' does not match expected dtype "
            "'{expected_dtype}' (cuda_ipc does not cast on device)",
            {"actual_dtype": dtype_name, "expected_dtype": expected_dtype},
        )

    return val


class CudaIpcTransport(DeviceTransport):
    """DeviceTransport backend for the same-host ``json+cuda_ipc`` mode."""

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
