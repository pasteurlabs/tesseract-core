# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fakes shared by the GPU-free tests of the GPU transports."""

from __future__ import annotations

from typing import Any


class FakeCudaArray:
    """Mimics a GPU array's metadata surface without any device memory."""

    def __init__(
        self,
        shape: tuple[int, ...],
        typestr: str = "<f4",
        data_ptr: int = 0x1000,
        device: Any = None,
        strides: tuple[int, ...] | None = None,
    ) -> None:
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": typestr,
            "data": (data_ptr, False),
            "strides": strides,
            "version": 3,
        }
        if device is not None:
            self.device = device
