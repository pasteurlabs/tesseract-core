# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch compatibility layer for Tesseract.

Provides :func:`apply_tesseract`, which wraps any Tesseract as a differentiable
PyTorch operation supporting both reverse-mode (``.backward()``) and forward-mode
(``torch.autograd.forward_ad``) automatic differentiation.
"""

from .function import apply_tesseract

__all__ = ["apply_tesseract"]
