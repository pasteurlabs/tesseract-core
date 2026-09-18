# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the copy-free ``cuda_vmm`` device transport.

``cuda_vmm`` is a sibling of ``cuda_ipc``, selected by the ``gpu_transport``
config (not auto-selected by probing a pointer inside ``cuda_ipc``). It has its
own wire encoding (``encoding: "cuda_vmm"``, ``vmm:``-prefixed buffer). These
cover the schema, the fd-passing server framing/reuse, and the explicit-opt-in
error contract -- without a GPU. The real cross-process VMM transfer lives in
``test_vmm.py`` (marked ``gpu``).
"""

from __future__ import annotations

import types

import pytest


def test_non_vmm_pointer_is_not_exportable():
    """A bogus / non-VMM pointer is reported not-exportable, never raising."""
    from tesseract_core.runtime.cuda import vmm

    # An arbitrary integer is not a VMM allocation; the driver rejects it.
    assert vmm.is_vmm_exportable(0x5000) is False


def test_is_vmm_exportable_false_without_driver(monkeypatch):
    """Without a loadable CUDA driver, exportability is False (legacy fallback)."""
    from tesseract_core.runtime.cuda import api as cuda_api
    from tesseract_core.runtime.cuda import vmm

    def _no_driver(_base_ptr):
        raise RuntimeError("no libcuda")

    monkeypatch.setattr(cuda_api, "retain_allocation_handle", _no_driver)
    assert vmm.is_vmm_exportable(0x1000) is False


def test_cuda_vmm_schema_accepts_vmm_buffer():
    from tesseract_core.runtime.array_encoding import CudaVmmArrayData

    # vmm form: vmm:<sockpath_b64>:<export_id>:<offset>:<size>:<device>
    CudaVmmArrayData(buffer="vmm:L3RtcC9z:3:128:2097152:0", encoding="cuda_vmm")


@pytest.mark.parametrize(
    "bad",
    [
        "vmm:onlyone",
        "vmm:a:b:c",  # non-numeric fields
        "vmm:L3Rt:0:0:0",  # too few fields
        "vmm::0:0:0:0",  # empty sockpath
        "0:YWJj:0:64",  # legacy cuda_ipc form is not a cuda_vmm buffer
    ],
)
def test_cuda_vmm_schema_rejects_malformed(bad):
    from pydantic import ValidationError

    from tesseract_core.runtime.array_encoding import CudaVmmArrayData

    with pytest.raises(ValidationError):
        CudaVmmArrayData(buffer=bad, encoding="cuda_vmm")


def test_vmm_export_raises_on_non_vmm_memory(monkeypatch):
    """An explicit VMM export of non-VMM memory fails loudly and actionably."""
    from tesseract_core.runtime.cuda import api as cuda_api
    from tesseract_core.runtime.cuda import vmm

    monkeypatch.setattr(cuda_api, "get_allocation_base", lambda _ptr: (0x1000, 4096))
    # non-VMM memory: the driver rejects retaining a handle for it.
    monkeypatch.setattr(cuda_api, "retain_allocation_handle", lambda _base: None)

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
    server = vmm._FdPassServer()
    try:
        with pytest.raises(RuntimeError, match="requires VMM-backed device memory"):
            vmm._build_vmm_descriptor(FakeCudaArray(), server)
        # Nothing was registered, since retain failed before server.register().
        assert not server._registry
    finally:
        server.close()


def test_cuda_vmm_registered_as_transport():
    """The cuda_vmm backend is discoverable by name and satisfies the interface."""
    from tesseract_core.runtime.device_transport import DeviceTransport, get_transport

    transport = get_transport("cuda_vmm")
    assert transport.name == "cuda_vmm"
    assert transport.reach == "same_host"
    assert isinstance(transport, DeviceTransport)


def test_cuda_vmm_consumer_bootstrap_is_noop():
    """Only the producer holds server state; the consumer's bootstrap is None."""
    from tesseract_core.runtime.device_transport import get_transport

    assert get_transport("cuda_vmm").bootstrap("consumer") is None


def test_cuda_vmm_gated_by_gpu_transport(monkeypatch):
    """cuda_vmm is offered only when configured via gpu_transport."""
    from tesseract_core.runtime import config, file_interactions

    monkeypatch.setattr(
        config, "get_config", lambda: types.SimpleNamespace(gpu_transport="cuda_vmm")
    )
    assert file_interactions.available_gpu_transports() == ("none", "cuda_vmm")

    monkeypatch.setattr(
        config, "get_config", lambda: types.SimpleNamespace(gpu_transport="none")
    )
    assert file_interactions.available_gpu_transports() == ("none",)


# ---------------------------------------------------------------------------
# fd-passing server framing + connection reuse, exercised without a GPU by
# stubbing the single driver call that turns a handle into a fd.
# ---------------------------------------------------------------------------


def _pipe_fd() -> int:
    """A real, sendable fd (the read end of a pipe) standing in for a VMM fd."""
    import os

    r, w = os.pipe()
    os.close(w)
    return r


def _stub_driver_export(monkeypatch):
    """Make export_to_shareable_fd hand back a fresh pipe fd, and mem_release a no-op."""
    from tesseract_core.runtime.cuda import api as cuda_api

    monkeypatch.setattr(cuda_api, "export_to_shareable_fd", lambda _handle: _pipe_fd())
    monkeypatch.setattr(cuda_api, "mem_release", lambda _handle: None)


def test_fd_server_roundtrip_and_reuse(monkeypatch):
    """A single connection fetches an fd, then reuses the same socket for more."""
    import os

    from tesseract_core.runtime.cuda import vmm

    _stub_driver_export(monkeypatch)
    server = vmm._FdPassServer()
    try:
        id_a = server.register(handle=111, keepalive=None)
        id_b = server.register(handle=222, keepalive=None)

        fd1 = vmm._fetch_fd(server.path, id_a)
        assert fd1 >= 0
        os.close(fd1)

        # A second fetch to the same path must reuse the cached connection.
        conn_before = vmm._FETCH_CONNS.get(server.path)
        fd2 = vmm._fetch_fd(server.path, id_b)
        conn_after = vmm._FETCH_CONNS.get(server.path)
        assert conn_before is conn_after is not None
        os.close(fd2)
    finally:
        vmm._drop_conn(server.path)
        server.close()


def test_fd_server_batch_fetch(monkeypatch):
    """One request fetches many fds in a single SCM_RIGHTS message."""
    import os

    from tesseract_core.runtime.cuda import vmm

    _stub_driver_export(monkeypatch)
    server = vmm._FdPassServer()
    try:
        ids = [server.register(handle=h, keepalive=None) for h in (1, 2, 3, 4)]
        fds = vmm._fetch_fds(server.path, ids)
        assert len(fds) == len(ids)
        assert all(fd >= 0 for fd in fds)
        for fd in fds:
            os.close(fd)
    finally:
        vmm._drop_conn(server.path)
        server.close()


def test_fd_server_miss_raises(monkeypatch):
    """An unknown export id is reported as a miss and raises, leaking no fd."""
    from tesseract_core.runtime.cuda import vmm

    _stub_driver_export(monkeypatch)
    server = vmm._FdPassServer()
    try:
        with pytest.raises(RuntimeError, match="no fd"):
            vmm._fetch_fd(server.path, 99999)
    finally:
        vmm._drop_conn(server.path)
        server.close()


def test_fd_server_reconnects_after_close(monkeypatch):
    """A dropped/broken cached connection is transparently reconnected."""
    import os

    from tesseract_core.runtime.cuda import vmm

    _stub_driver_export(monkeypatch)
    server = vmm._FdPassServer()
    try:
        eid = server.register(handle=7, keepalive=None)
        fd1 = vmm._fetch_fd(server.path, eid)
        os.close(fd1)
        # Simulate a stale connection by closing the cached socket underneath.
        stale = vmm._FETCH_CONNS[server.path]
        stale.close()
        fd2 = vmm._fetch_fd(server.path, eid)  # must recover
        assert fd2 >= 0
        os.close(fd2)
    finally:
        vmm._drop_conn(server.path)
        server.close()


def test_recv_exactly_clean_eof():
    """A clean hang-up at a message boundary returns None (not an error)."""
    import socket

    from tesseract_core.runtime.cuda import vmm

    a, b = socket.socketpair()
    b.close()
    try:
        assert vmm._recv_exactly(a, 8) is None
    finally:
        a.close()


def test_recv_exactly_partial_frame_raises():
    """A truncated frame is a protocol error, not a graceful hang-up."""
    import socket

    from tesseract_core.runtime.cuda import vmm

    a, b = socket.socketpair()
    b.sendall(b"\x01\x02\x03")  # fewer than the 8 requested
    b.close()
    try:
        with pytest.raises(OSError, match="mid-frame"):
            vmm._recv_exactly(a, 8)
    finally:
        a.close()


def test_session_opens_and_closes_active_server(monkeypatch):
    """session() installs the active server on enter and tears it down on exit."""
    from tesseract_core.runtime.cuda import vmm
    from tesseract_core.runtime.device_transport import get_transport

    _stub_driver_export(monkeypatch)
    assert vmm._active_server() is None
    with get_transport("cuda_vmm").session("producer") as server:
        assert server is not None
        assert vmm._active_server() is server
        assert server._running is True
    assert vmm._active_server() is None
    assert server._running is False


def test_consumer_session_is_noop():
    """A consumer session holds no server state and yields None."""
    from tesseract_core.runtime.cuda import vmm
    from tesseract_core.runtime.device_transport import get_transport

    with get_transport("cuda_vmm").session("consumer") as server:
        assert server is None
        assert vmm._active_server() is None


def test_register_without_active_session_raises(monkeypatch):
    """Exporting with no open session raises rather than exporting silently."""
    from tesseract_core.runtime.cuda import vmm
    from tesseract_core.runtime.device_transport import get_transport

    _stub_driver_export(monkeypatch)
    assert vmm._active_server() is None
    fake_arr = types.SimpleNamespace(__cuda_array_interface__={})
    with pytest.raises(RuntimeError, match="no active fd-passing session"):
        get_transport("cuda_vmm").register(fake_arr)
    assert vmm._active_server() is None
