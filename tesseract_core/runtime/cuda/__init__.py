# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-agnostic access to CUDA, wrapping cudart/libcuda via ctypes.

This package isolates every low-level detail of talking to CUDA -- library
discovery and loading, ctypes signatures, the IPC handle ABI, and the DLPack
capsule handshake -- so the rest of the codebase works with plain Python values
(device pointers as ``int``, IPC handles as ``bytes``) and never imports ctypes.

Layers:

* :mod:`~tesseract_core.runtime.cuda.loader` -- find and load libcudart/libcuda.
* :mod:`~tesseract_core.runtime.cuda.runtime` -- the plain-Python CUDA API
  (memory management, IPC, device-to-device/host copies).
* :mod:`~tesseract_core.runtime.cuda.dlpack` -- export an owned device buffer
  as a DLPack capsule.

The main consumer is :mod:`tesseract_core.runtime.cuda_ipc`, which layers the
``json+cuda_ipc`` array-encoding policy on top of this API.
"""

from tesseract_core.runtime.cuda import dlpack, loader, runtime
from tesseract_core.runtime.cuda.loader import iter_cudart_candidates

__all__ = [
    "dlpack",
    "iter_cudart_candidates",
    "loader",
    "runtime",
]
