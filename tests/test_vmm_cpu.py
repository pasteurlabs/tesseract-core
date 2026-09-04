# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the copy-free VMM path behind ``json+cuda_ipc``.

The VMM path is reached transparently from ``dump_cuda_ipc_arraydict`` when the
source memory is VMM-exportable, and decoded when the wire buffer is
``vmm:``-prefixed. These cover the plumbing that decides *whether* to take that
path and that a non-VMM host keeps the legacy path -- without a GPU. The real
cross-process VMM transfer lives in ``test_vmm.py`` (marked ``gpu``).
"""

from __future__ import annotations

import pytest


def test_non_vmm_pointer_is_not_exportable():
    """A bogus / non-VMM pointer is reported not-exportable, never raising."""
    from tesseract_core.runtime import vmm_transport

    # An arbitrary integer is not a VMM allocation; the driver rejects it.
    assert vmm_transport.is_vmm_exportable(0x5000) is False


def test_is_vmm_exportable_false_without_driver(monkeypatch):
    """Without a loadable CUDA driver, exportability is False (legacy fallback)."""
    from tesseract_core.runtime import vmm_transport

    def _no_driver():
        raise RuntimeError("no libcuda")

    monkeypatch.setattr(vmm_transport, "_get_cuda_driver", _no_driver)
    assert vmm_transport.is_vmm_exportable(0x1000) is False


def test_cuda_ipc_schema_accepts_vmm_variant():
    from tesseract_core.runtime.array_encoding import CudaIpcArrayData

    # legacy form still valid
    CudaIpcArrayData(buffer="0:YWJj:0:64", encoding="cuda_ipc")
    # vmm variant: vmm:<sockpath_b64>:<export_id>:<offset>:<size>:<device>
    CudaIpcArrayData(buffer="vmm:L3RtcC9z:3:128:2097152:0", encoding="cuda_ipc")


@pytest.mark.parametrize(
    "bad",
    [
        "vmm:onlyone",
        "vmm:a:b:c",  # non-numeric fields
        "vmm:L3Rt:0:0:0",  # too few fields
        "vmm::0:0:0:0",  # empty sockpath
    ],
)
def test_cuda_ipc_schema_rejects_malformed_vmm(bad):
    from pydantic import ValidationError

    from tesseract_core.runtime.array_encoding import CudaIpcArrayData

    with pytest.raises(ValidationError):
        CudaIpcArrayData(buffer=bad, encoding="cuda_ipc")


def test_load_dispatches_vmm_prefix(monkeypatch):
    """A vmm:-prefixed buffer routes load_cuda_ipc_arraydict to the VMM loader."""
    import numpy as np

    from tesseract_core.runtime import cuda_ipc, vmm_transport
    from tesseract_core.runtime.cuda_ipc import IpcDeviceArray

    # A well-formed (non-device) IpcDeviceArray so the return type checks out; we
    # only assert dispatch happened, never touch its buffer.
    sentinel = IpcDeviceArray(0, 0, (4,), np.dtype("float32"))
    called = {}

    def fake_load(val):
        called["val"] = val
        return sentinel

    monkeypatch.setattr(vmm_transport, "load_vmm_arraydict", fake_load)

    val = {
        "object_type": "array",
        "shape": [4],
        "dtype": "float32",
        "data": {"buffer": "vmm:L3RtcC9z:0:0:16:0", "encoding": "cuda_ipc"},
    }
    assert cuda_ipc.load_cuda_ipc_arraydict(val) is sentinel
    assert called["val"] is val


def test_dump_routes_to_vmm_when_exportable(monkeypatch):
    """dump_cuda_ipc_arraydict delegates to the VMM path for exportable memory."""
    from tesseract_core.runtime import cuda_ipc, vmm_transport

    class FakeCudaArray:
        def __init__(self):
            self.__cuda_array_interface__ = {
                "shape": (4,),
                "typestr": "<f4",
                "data": (0x7000, False),
                "strides": None,
                "version": 3,
            }

    marker = {
        "object_type": "array",
        "shape": [4],
        "dtype": "float32",
        "data": {"buffer": "vmm:L3Rt:0:0:16:0", "encoding": "cuda_ipc"},
    }
    monkeypatch.setattr(vmm_transport, "is_vmm_exportable", lambda ptr: True)
    monkeypatch.setattr(vmm_transport, "dump_vmm_arraydict", lambda arr: marker)

    assert cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray()) is marker
