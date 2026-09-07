# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DLPack ABI: hand an owned device buffer to Torch/JAX/CuPy zero-copy.

This module mirrors just enough of the DLPack C ABI to export a device buffer
as a ``"dltensor"`` PyCapsule, plus the capsule handshake and lifetime
bookkeeping. It is used by :class:`tesseract_core.runtime.cuda_ipc.IpcDeviceArray`
to expose its buffer via ``__dlpack__`` without depending on any GPU framework.

All ctypes and the CPython capsule API stay here. The public surface is two
functions -- :func:`make_dlpack_capsule` and :func:`drop_unconsumed_bundle` --
and the buffer is always freed through
:func:`tesseract_core.runtime.cuda.runtime.free`.
"""

import ctypes
from typing import Any

import numpy as np

from tesseract_core.runtime.cuda import runtime

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
            runtime.free(ptr)

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
