# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the copy-free VMM device transport (``json+cuda_vmm``).

The VMM transport is selected per-format at the dispatcher (``json+cuda_vmm`` ->
``array_encoding='vmm'``), a sibling of ``json+cuda_ipc`` -- not auto-selected by
probing the pointer inside ``dump_cuda_ipc_arraydict``. Both share the cuda_ipc
wire encoding (decode sniffs the ``vmm:`` prefix). These cover the selection
plumbing, the fd-passing server framing/reuse, and the explicit-opt-in error
contract -- without a GPU. The real cross-process VMM transfer lives in
``test_vmm.py`` (marked ``gpu``).
"""

from __future__ import annotations

import types

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


def test_cuda_ipc_dump_does_not_auto_route_to_vmm(monkeypatch):
    """cuda_ipc encode never selects VMM, even for VMM-exportable memory.

    Selection is per-format at the dispatcher now, not a probe inside
    dump_cuda_ipc_arraydict. json+cuda_ipc must stay the always-works legacy
    path; the copy-free VMM export is only reached via json+cuda_vmm.
    """
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

    # If cuda_ipc encode ever consulted the VMM path, these would fire; they must
    # not be touched at all.
    def _boom(*a, **k):
        raise AssertionError("cuda_ipc encode must not touch the VMM path")

    monkeypatch.setattr(vmm_transport, "is_vmm_exportable", _boom)
    monkeypatch.setattr(vmm_transport, "dump_vmm_arraydict", _boom)

    # It proceeds down the legacy path and fails there (no real CUDA runtime),
    # proving it never diverted to VMM. A RuntimeError from the legacy machinery
    # (not the AssertionError above) is the pass condition.
    with pytest.raises((RuntimeError, OSError, ValueError)):
        cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray())


def test_format_selects_device_transport():
    """json+cuda_ipc / json+cuda_vmm map to their transports at the dispatcher."""
    from tesseract_core.sdk.tesseract import _DEVICE_TRANSPORT_BY_FORMAT

    assert _DEVICE_TRANSPORT_BY_FORMAT["json+cuda_ipc"] == "cuda_ipc"
    assert _DEVICE_TRANSPORT_BY_FORMAT["json+cuda_vmm"] == "vmm"


def test_available_formats_gates_cuda_vmm(monkeypatch):
    """json+cuda_vmm is offered only when the experimental flag is enabled."""
    from tesseract_core.runtime import config, file_interactions

    monkeypatch.setattr(
        file_interactions,
        "get_config",
        lambda: types.SimpleNamespace(enable_experimental_cuda_ipc=True),
        raising=False,
    )
    # get_config is imported lazily inside available_formats; patch at source.
    monkeypatch.setattr(
        config,
        "get_config",
        lambda: types.SimpleNamespace(enable_experimental_cuda_ipc=True),
    )
    formats = file_interactions.available_formats()
    assert "json+cuda_vmm" in formats
    assert "json+cuda_ipc" in formats

    monkeypatch.setattr(
        config,
        "get_config",
        lambda: types.SimpleNamespace(enable_experimental_cuda_ipc=False),
    )
    assert "json+cuda_vmm" not in file_interactions.available_formats()


def test_vmm_export_raises_on_non_vmm_memory(monkeypatch):
    """An explicit VMM export of non-VMM memory fails loudly and actionably."""
    from tesseract_core.runtime import vmm_transport

    class _Driver:
        def cuMemGetAddressRange_v2(self, base, size, ptr):
            base._obj.value = 0x1000
            size._obj.value = 4096
            return 0

        def cuMemRetainAllocationHandle(self, handle, base):
            return 1  # non-VMM memory: the driver rejects it

    monkeypatch.setattr(vmm_transport, "_get_cuda_driver", lambda: _Driver())

    class FakeCudaArray:
        def __init__(self):
            self.__cuda_array_interface__ = {
                "shape": (4,),
                "typestr": "<f4",
                "data": (0x1000, False),
                "strides": None,
                "version": 3,
            }

    # A real server (cheap: just a socket), so the export fails at the retain
    # step -- before any handle is registered -- rather than at type-checking.
    server = vmm_transport._FdPassServer()
    try:
        with pytest.raises(RuntimeError, match="requires VMM-backed device memory"):
            vmm_transport._build_vmm_descriptor(FakeCudaArray(), server)
        # Nothing was registered, since retain failed before server.register().
        assert not server._registry
    finally:
        server.close()


def test_vmm_registered_as_transport():
    """The vmm backend is discoverable by name and satisfies the protocol."""
    from tesseract_core.runtime.device_transport import DeviceTransport, get_transport

    transport = get_transport("vmm")
    assert transport.name == "vmm"
    assert transport.reach == "same_host"
    assert isinstance(transport, DeviceTransport)


def test_vmm_consumer_bootstrap_is_noop():
    """Only the producer holds server state; the consumer's bootstrap is None."""
    from tesseract_core.runtime.device_transport import get_transport

    assert get_transport("vmm").bootstrap("consumer") is None


# ---------------------------------------------------------------------------
# fd-passing server framing + connection reuse (I1/I3), exercised without a GPU
# by stubbing the single driver call that turns a handle into a fd.
# ---------------------------------------------------------------------------


def _pipe_fd() -> int:
    """A real, sendable fd (the read end of a pipe) standing in for a VMM fd."""
    import os

    r, w = os.pipe()
    os.close(w)
    return r


def _stub_driver_export(monkeypatch):
    """Make cuMemExportToShareableHandle hand back a fresh pipe fd, ret 0."""
    import ctypes

    from tesseract_core.runtime import vmm_transport

    class _Driver:
        def cuMemExportToShareableHandle(self, fd_ptr, _handle, _type, _flags):
            fd_ptr._obj.value = _pipe_fd()
            return 0

        def cuMemRelease(self, _handle):
            return 0

    monkeypatch.setattr(vmm_transport, "_get_cuda_driver", lambda: _Driver())
    # ctypes.byref returns a lightweight object exposing ._obj; the stub above
    # writes through it, matching how the real ctypes out-param is filled.
    return ctypes


def test_fd_server_roundtrip_and_reuse(monkeypatch):
    """A single connection fetches an fd, then reuses the same socket for more."""
    import os

    from tesseract_core.runtime import vmm_transport

    _stub_driver_export(monkeypatch)
    server = vmm_transport._FdPassServer()
    try:
        id_a = server.register(handle=111, keepalive=None)
        id_b = server.register(handle=222, keepalive=None)

        fd1 = vmm_transport._fetch_fd(server.path, id_a)
        assert fd1 >= 0
        os.close(fd1)

        # A second fetch to the same path must reuse the cached connection.
        conn_before = vmm_transport._FETCH_CONNS.get(server.path)
        fd2 = vmm_transport._fetch_fd(server.path, id_b)
        conn_after = vmm_transport._FETCH_CONNS.get(server.path)
        assert conn_before is conn_after is not None
        os.close(fd2)
    finally:
        vmm_transport._drop_conn(server.path)
        server.close()


def test_fd_server_batch_fetch(monkeypatch):
    """One request fetches many fds in a single SCM_RIGHTS message (I1)."""
    import os

    from tesseract_core.runtime import vmm_transport

    _stub_driver_export(monkeypatch)
    server = vmm_transport._FdPassServer()
    try:
        ids = [server.register(handle=h, keepalive=None) for h in (1, 2, 3, 4)]
        fds = vmm_transport._fetch_fds(server.path, ids)
        assert len(fds) == len(ids)
        assert all(fd >= 0 for fd in fds)
        for fd in fds:
            os.close(fd)
    finally:
        vmm_transport._drop_conn(server.path)
        server.close()


def test_fd_server_miss_raises(monkeypatch):
    """An unknown export id is reported as a miss and raises, leaking no fd."""
    from tesseract_core.runtime import vmm_transport

    _stub_driver_export(monkeypatch)
    server = vmm_transport._FdPassServer()
    try:
        with pytest.raises(RuntimeError, match="no fd"):
            vmm_transport._fetch_fd(server.path, 99999)
    finally:
        vmm_transport._drop_conn(server.path)
        server.close()


def test_fd_server_reconnects_after_close(monkeypatch):
    """A dropped/broken cached connection is transparently reconnected."""
    import os

    from tesseract_core.runtime import vmm_transport

    _stub_driver_export(monkeypatch)
    server = vmm_transport._FdPassServer()
    try:
        eid = server.register(handle=7, keepalive=None)
        fd1 = vmm_transport._fetch_fd(server.path, eid)
        os.close(fd1)
        # Simulate a stale connection by closing the cached socket underneath.
        stale = vmm_transport._FETCH_CONNS[server.path]
        stale.close()
        fd2 = vmm_transport._fetch_fd(server.path, eid)  # must recover
        assert fd2 >= 0
        os.close(fd2)
    finally:
        vmm_transport._drop_conn(server.path)
        server.close()


def test_recv_exactly_clean_eof(monkeypatch):
    """A clean hang-up at a message boundary returns None (not an error)."""
    import socket

    from tesseract_core.runtime import vmm_transport

    a, b = socket.socketpair()
    b.close()
    try:
        assert vmm_transport._recv_exactly(a, 8) is None
    finally:
        a.close()


def test_recv_exactly_partial_frame_raises():
    """A truncated frame is a protocol error, not a graceful hang-up."""
    import socket

    from tesseract_core.runtime import vmm_transport

    a, b = socket.socketpair()
    b.sendall(b"\x01\x02\x03")  # fewer than the 8 requested
    b.close()
    try:
        with pytest.raises(OSError, match="mid-frame"):
            vmm_transport._recv_exactly(a, 8)
    finally:
        a.close()


def test_shutdown_closes_server(monkeypatch):
    """The transport shutdown releases and closes the fallback server."""
    from tesseract_core.runtime import vmm_transport
    from tesseract_core.runtime.device_transport import get_transport

    _stub_driver_export(monkeypatch)
    server = vmm_transport._FdPassServer()
    vmm_transport._set_fallback_server(server)
    server.register(handle=5, keepalive=None)

    get_transport("vmm").shutdown()
    assert vmm_transport._FALLBACK_SERVER is None
    assert server._running is False
