# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DLPack ABI: exchange CUDA device buffers with Torch/JAX/CuPy zero-copy.

This module mirrors just enough of the DLPack C ABI to move buffers in both
directions without depending on any GPU framework:

* *export* -- :func:`make_dlpack_capsule` / :func:`drop_unconsumed_bundle` wrap an
  owned device buffer as a ``"dltensor"`` PyCapsule, used by
  :class:`tesseract_core.runtime.cuda.ipc.IpcDeviceArray` to expose its buffer
  via ``__dlpack__``.
* *import* -- :func:`is_dlpack_cuda` / :func:`read_dlpack_cuda_metadata` borrow a
  foreign producer's device pointer, shape, and dtype, used by the encode path to
  route a CAI-less array (e.g. JAX) over a CUDA transport.

All ctypes and the CPython capsule API stay here; the buffer is always freed
through :func:`tesseract_core.runtime.cuda.api.free`.
"""

import ctypes
from typing import Any

import numpy as np

from tesseract_core.runtime.cuda import api

_kDLCUDA = 2  # DLDeviceType for CUDA global memory


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


# void (*)(struct DLManagedTensor *self)
_DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DLManagedTensorDeleter),
    ]


# DLDataTypeCode values (kDLInt, kDLUInt, kDLFloat, ..., kDLBool, kDLComplex).
_DLPACK_TYPE_CODES = {
    "i": 0,  # kDLInt
    "u": 1,  # kDLUInt
    "f": 2,  # kDLFloat
    "b": 6,  # kDLBool
    "c": 5,  # kDLComplex
}

# Reverse of _DLPACK_TYPE_CODES: DLDataTypeCode -> NumPy dtype kind, used to
# reconstruct a NumPy dtype from a DLPack tensor's (code, bits) when reading
# metadata off a foreign array (e.g. a JAX buffer that exposes only DLPack).
_NUMPY_KINDS = {code: kind for kind, code in _DLPACK_TYPE_CODES.items()}

# Keep PyCapsule_* usable from ctypes for the DLPack capsule handshake.
_pythonapi = ctypes.pythonapi
_pythonapi.PyCapsule_New.restype = ctypes.py_object
_pythonapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
]
_pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
_pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_pythonapi.PyCapsule_SetName.restype = ctypes.c_int
_pythonapi.PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]
_pythonapi.PyCapsule_IsValid.restype = ctypes.c_int
_pythonapi.PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]

# DLDeviceType for CUDA, re-exported for __dlpack_device__ callers.
DLDEVICE_CUDA = _kDLCUDA


def _dlpack_dtype(dtype: np.dtype) -> _DLDataType:
    """Map a NumPy dtype to a DLPack ``DLDataType`` (code/bits/lanes)."""
    code = _DLPACK_TYPE_CODES.get(dtype.kind)
    if code is None:
        raise TypeError(f"dtype {dtype!r} has no DLPack type code")
    return _DLDataType(code=code, bits=dtype.itemsize * 8, lanes=1)


# ---------------------------------------------------------------------------
# DLPack bundle registry
# ---------------------------------------------------------------------------
#
# A DLPack capsule must outlive the object that produced it: the consumer may
# hold the borrowed tensor arbitrarily long and only calls the deleter when it
# is done. We therefore keep each capsule's backing ctypes state (the
# DLManagedTensor, the shape array, the CFUNCTYPE deleter trampoline) alive in a
# process-global registry keyed by an integer token, rather than on the
# producing IpcDeviceArray. The deleter removes its own entry when invoked, so
# the state is reclaimed exactly when the consumer releases the tensor.

_BUNDLES: dict[int, Any] = {}
_NEXT_TOKEN = 0


def make_dlpack_capsule(
    ptr: int, device: int, shape: tuple[int, ...], dtype: np.dtype
) -> tuple[Any, int]:
    """Build a ``"dltensor"`` capsule that owns ``ptr`` and register its state.

    Returns ``(capsule, token)``. The buffer is freed exactly once, by the
    deleter, whether the capsule is consumed by a framework or dropped
    un-consumed via :func:`drop_unconsumed_bundle`.
    """
    global _NEXT_TOKEN
    token = _NEXT_TOKEN
    _NEXT_TOKEN += 1

    shape_arr = (ctypes.c_int64 * len(shape))(*shape)

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.dl_tensor.device = _DLDevice(device_type=_kDLCUDA, device_id=device)
    managed.dl_tensor.ndim = len(shape)
    managed.dl_tensor.dtype = _dlpack_dtype(dtype)
    managed.dl_tensor.shape = shape_arr
    managed.dl_tensor.strides = ctypes.cast(None, ctypes.POINTER(ctypes.c_int64))
    managed.dl_tensor.byte_offset = 0

    def _deleter(_managed_ptr: int) -> None:
        # Runs when the consumer releases the tensor. Free the buffer and drop
        # our registry entry so the ctypes state can be reclaimed. Guard against
        # a second invocation (bundle already gone).
        bundle = _BUNDLES.pop(token, None)
        if bundle is not None:
            api.free(ptr)

    c_deleter = _DLManagedTensorDeleter(_deleter)
    managed.deleter = c_deleter
    managed.manager_ctx = None

    capsule = _pythonapi.PyCapsule_New(ctypes.byref(managed), b"dltensor", None)

    # Keep every object the capsule/consumer may still touch alive until the
    # deleter drops the entry.
    _BUNDLES[token] = (managed, shape_arr, c_deleter, capsule)
    return capsule, token


def drop_unconsumed_bundle(token: int) -> None:
    """Free a bundle's buffer iff its capsule was never consumed.

    Called from the :class:`IpcDeviceArray` finalizer. If the capsule is still
    named ``"dltensor"`` no framework adopted it, so we invoke the deleter to
    free the buffer. If it was renamed to ``"used_dltensor"`` a consumer owns it
    and will (or already did) free it via the deleter, so we leave it alone.
    """
    bundle = _BUNDLES.get(token)
    if bundle is None:
        return
    _managed, _shape_arr, c_deleter, capsule = bundle
    still_dltensor = bool(_pythonapi.PyCapsule_IsValid(capsule, b"dltensor"))
    if still_dltensor:
        # Nobody adopted it -> free now (the deleter pops the registry entry).
        c_deleter(0)


# ---------------------------------------------------------------------------
# Reading metadata off a foreign DLPack producer
# ---------------------------------------------------------------------------
#
# Some frameworks (notably JAX) expose their device buffers only through DLPack
# (``__dlpack__`` / ``__dlpack_device__``) and never implement
# ``__cuda_array_interface__``. To route such an array over a CUDA device
# transport we need its device pointer, shape, dtype, and contiguity -- exactly
# what a DLPack capsule already carries. The helpers below borrow that metadata
# without adopting the buffer, leaving the producer as sole owner.


def is_dlpack_cuda(obj: Any) -> bool:
    """Whether ``obj`` is a DLPack producer whose buffer lives in CUDA device memory.

    Checks the cheap ``__dlpack_device__`` handshake (no capsule is requested, so
    no buffer is exported). Only plain CUDA global memory (``kDLCUDA``) qualifies:
    the CUDA transports export a device pointer by reference, which pinned host
    memory (``kDLCUDAHost``) is not, matching the device-only
    ``__cuda_array_interface__`` path.
    """
    dlpack_device = getattr(obj, "__dlpack_device__", None)
    if not callable(dlpack_device) or not callable(getattr(obj, "__dlpack__", None)):
        return False
    try:
        device_type, _device_id = dlpack_device()
    except Exception:
        return False
    return device_type == _kDLCUDA


def read_dlpack_cuda_metadata(
    obj: Any,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...] | None, np.dtype]:
    """Read ``(data_ptr, device, shape, strides, dtype)`` from a DLPack producer.

    Requests one DLPack capsule from ``obj``, reads the ``DLTensor`` fields, then
    consumes the capsule and runs its deleter so ``obj`` retains sole ownership
    of the underlying buffer (we only borrowed its metadata). ``strides`` is in
    elements, or ``None`` for a C-contiguous tensor. Raises ``TypeError`` if the
    capsule is not on a CUDA device or carries an unsupported dtype.
    """
    device_type, device_id = obj.__dlpack_device__()
    if device_type != _kDLCUDA:
        raise TypeError(
            f"DLPack tensor is not on a CUDA device (device_type={device_type})"
        )

    capsule = obj.__dlpack__()
    managed_ptr = _pythonapi.PyCapsule_GetPointer(capsule, b"dltensor")
    if not managed_ptr:
        raise TypeError("object did not return a valid 'dltensor' DLPack capsule")

    managed = ctypes.cast(managed_ptr, ctypes.POINTER(_DLManagedTensor)).contents
    tensor = managed.dl_tensor
    try:
        ndim = tensor.ndim
        shape = tuple(tensor.shape[i] for i in range(ndim))
        strides = (
            None
            if not tensor.strides
            else tuple(tensor.strides[i] for i in range(ndim))
        )
        dtype = _numpy_dtype(tensor.dtype)
        data_ptr = (tensor.data or 0) + tensor.byte_offset
    finally:
        # We borrowed metadata only; hand ownership back by consuming the capsule
        # (so the producer's own deleter is disarmed) and releasing the tensor.
        _pythonapi.PyCapsule_SetName(capsule, b"used_dltensor")
        if managed.deleter:
            managed.deleter(managed_ptr)

    return data_ptr, device_id, shape, strides, dtype


def _numpy_dtype(dl_dtype: _DLDataType) -> np.dtype:
    """Map a DLPack ``DLDataType`` back to a NumPy dtype.

    Rejects vectorized lanes (``lanes != 1``): the CUDA transports move a flat
    scalar-typed byte range, so a packed multi-lane element has no NumPy dtype
    to rebuild it from.
    """
    if dl_dtype.lanes != 1:
        raise TypeError(f"DLPack dtype with {dl_dtype.lanes} lanes is not supported")
    kind = _NUMPY_KINDS.get(dl_dtype.code)
    if kind is None:
        raise TypeError(f"DLPack dtype code {dl_dtype.code} is not supported")
    if dl_dtype.bits % 8 != 0:
        raise TypeError(f"DLPack dtype with {dl_dtype.bits} bits is not supported")
    return np.dtype(f"{kind}{dl_dtype.bits // 8}")
