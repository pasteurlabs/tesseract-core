# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand owned CUDA device buffers to Torch/JAX/CuPy as DLPack capsules.

Mirrors just enough of the DLPack C ABI to wrap a device buffer in a
``"dltensor"`` PyCapsule without depending on any GPU framework.
:class:`tesseract_core.runtime.cuda.ipc.IpcDeviceArray` implements
``__dlpack__`` with :func:`make_dlpack_capsule`. All ctypes and CPython capsule
calls stay in this module. A capsule, and the tensor a framework adopts from
it, keep the object that owns the buffer alive until the framework is done.
"""

import ctypes
import sys
from typing import Any

import numpy as np

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

# Private prototypes for the PyCapsule calls, so the shared ctypes.pythonapi
# functions keep whatever argtypes other code gave them.
_capsule_new = ctypes.PYFUNCTYPE(
    ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p
)(("PyCapsule_New", ctypes.pythonapi))
_capsule_is_valid = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_IsValid", ctypes.pythonapi)
)

# Capsule name for an unconsumed DLPack tensor. Kept here because a capsule
# borrows its name for its whole life.
_DLTENSOR = b"dltensor"

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
# is done. Each capsule's backing ctypes state (the DLManagedTensor and its
# shape array), together with the object that owns the device buffer and the
# capsule itself, is kept in a process-global registry keyed by the
# DLManagedTensor's address. The deleter removes the entry, which lets the
# owner go and free the buffer once nothing else uses it either.

_BUNDLES: dict[int, Any] = {}

# Addresses of capsules no framework has consumed yet. A framework consumes a
# capsule as soon as it gets it, so this stays small.
_UNCONSUMED: set[int] = set()


@_DLManagedTensorDeleter
def _deleter(managed_ptr: int | None) -> None:
    # Runs when the consumer releases the tensor, possibly at interpreter
    # shutdown, and a ctypes callback must not raise.
    try:
        _BUNDLES.pop(managed_ptr, None)
    except BaseException:  # noqa: BLE001, S110
        pass


def drop_abandoned_capsules() -> None:
    """Let go of the owners of capsules that were dropped without being consumed.

    A capsule a framework refused, or that was never passed to one, has no
    consumer to call the deleter, so its bundle would keep the owner alive for
    good. Capsules without a destructor are used because a destructor would run
    Python code from inside the consumer's error handling; instead, a capsule
    that is still unconsumed and that nothing but its bundle references is
    dropped here, which callers do before each export and decode.
    """
    for address in list(_UNCONSUMED):
        bundle = _BUNDLES.get(address)
        if bundle is None:
            _UNCONSUMED.discard(address)
            continue
        capsule = bundle[3]
        if not _capsule_is_valid(capsule, _DLTENSOR):
            # Consumed: the consumer calls the deleter when it is done.
            _UNCONSUMED.discard(address)
        elif sys.getrefcount(capsule) <= 3:
            # Referenced only by the bundle, this local and the call's argument.
            _UNCONSUMED.discard(address)
            _BUNDLES.pop(address, None)


def make_dlpack_capsule(
    ptr: int,
    device: int,
    shape: tuple[int, ...],
    dtype: np.dtype,
    owner: Any,
) -> Any:
    """Build a ``"dltensor"`` capsule for the C-contiguous buffer at ``ptr``.

    ``owner`` is the object whose lifetime the buffer follows. The capsule keeps
    it alive until the consumer releases the tensor, or, for a capsule dropped
    without being consumed, until :func:`drop_abandoned_capsules` next runs.
    """
    drop_abandoned_capsules()
    shape_arr = (ctypes.c_int64 * len(shape))(*shape)

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.dl_tensor.device = _DLDevice(device_type=_kDLCUDA, device_id=device)
    managed.dl_tensor.ndim = len(shape)
    managed.dl_tensor.dtype = _dlpack_dtype(dtype)
    managed.dl_tensor.shape = shape_arr
    managed.dl_tensor.strides = ctypes.cast(None, ctypes.POINTER(ctypes.c_int64))
    managed.dl_tensor.byte_offset = 0
    managed.deleter = _deleter
    managed.manager_ctx = None

    address = ctypes.addressof(managed)
    capsule = _capsule_new(address, _DLTENSOR, None)
    _BUNDLES[address] = (managed, shape_arr, owner, capsule)
    _UNCONSUMED.add(address)
    return capsule
