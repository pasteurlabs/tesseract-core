# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the cuda_ipc encoding logic.

These run on ordinary (GPU-less) CI runners. They cover the Python
*orchestration* around CUDA IPC -- payload assembly, base/offset arithmetic,
device-ordinal detection, shape/dtype validation, export groups, the
serve-side release hook, the ``--ipc=host`` wiring, and the CLI guard -- by

  * feeding fake objects that expose ``__cuda_array_interface__`` (no device
    memory), and
  * running against the ``mocked_cuda`` fixture, which swaps the plain-Python
    CUDA runtime layer (``tesseract_core.runtime.cuda.api``) for an
    in-process fake. Because the fake replaces the module's real public seam --
    not scattered ctypes internals -- the encoding policy is exercised exactly as
    it ships.

They deliberately do NOT verify that IPC transfers the correct bytes, that the
by-value handle marshalling is right, or that offsets read the right data: those
are CUDA-runtime properties with no meaning against a fake. Those guarantees are
covered by the GPU tests in ``test_cuda_ipc.py`` (marked ``@pytest.mark.gpu``).

Library-discovery tests (wheel/soname resolution) live at the bottom and target
``tesseract_core.runtime.cuda.loader`` directly, since that is where discovery
now lives.
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from tesseract_core.runtime import array_encoding
from tesseract_core.runtime.cuda import api as cuda_api
from tesseract_core.runtime.cuda import ipc as cuda_ipc
from tesseract_core.runtime.cuda import loader


def _unpack_cuda_ipc(data: dict) -> dict:
    """Split a packed cuda_ipc ``buffer`` back into its named components."""
    device, handle, storage_offset, storage_size = data["buffer"].split(":")
    return {
        "device": int(device),
        "handle": handle,
        "storage_offset": int(storage_offset),
        "storage_size": int(storage_size),
    }


# ── Fakes ───────────────────────────────────────────────────────────────


class FakeCudaArray:
    """Mimics a GPU array's metadata surface without any device memory."""

    def __init__(
        self,
        shape: tuple[int, ...],
        typestr: str,
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


class _CuPyDevice:
    """Stand-in for ``cupy.ndarray.device`` (exposes ``.id``)."""

    def __init__(self, id: int) -> None:
        self.id = id


class _TorchDevice:
    """Stand-in for ``torch.Tensor.device`` (exposes ``.index``)."""

    def __init__(self, index: int | None) -> None:
        self.index = index


# ── Encode-side orchestration (mocked CUDA) ─────────────────────────────


def test_dump_assembles_payload_and_offset(mocked_cuda):
    arr = FakeCudaArray((4, 8), "<f4", data_ptr=0x5000)
    out = cuda_ipc.dump_cuda_ipc_arraydict(arr)

    assert out["object_type"] == "array"
    assert out["shape"] == [4, 8]
    assert out["dtype"] == "float32"
    data = out["data"]
    assert data["encoding"] == "cuda_ipc"
    unpacked = _unpack_cuda_ipc(data)
    # base = 0x5000 - 256; offset = data_ptr - base = 256; size from fake = 4096
    assert unpacked["storage_offset"] == 256
    assert unpacked["storage_size"] == 4096
    # The handle must be taken on the allocation *base*, not the data pointer.
    assert mocked_cuda.calls["get_handle"] == [0x5000 - 256]
    # Handle is base64 of the 64 raw bytes.
    import pybase64

    assert len(pybase64.b64decode(unpacked["handle"])) == cuda_api.IPC_HANDLE_SIZE


def test_dump_device_detection_cupy(mocked_cuda):
    arr = FakeCudaArray((3,), "<f4", device=_CuPyDevice(id=2))
    out = cuda_ipc.dump_cuda_ipc_arraydict(arr)
    assert _unpack_cuda_ipc(out["data"])["device"] == 2


def test_dump_device_detection_torch(mocked_cuda):
    arr = FakeCudaArray((3,), "<f4", device=_TorchDevice(index=3))
    out = cuda_ipc.dump_cuda_ipc_arraydict(arr)
    assert _unpack_cuda_ipc(out["data"])["device"] == 3


def test_dump_device_defaults_to_zero(mocked_cuda):
    # No .device attribute, and torch tensors with device.index == None.
    assert (
        _unpack_cuda_ipc(
            cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((3,), "<f4"))["data"]
        )["device"]
        == 0
    )
    arr = FakeCudaArray((3,), "<f4", device=_TorchDevice(index=None))
    assert (
        _unpack_cuda_ipc(cuda_ipc.dump_cuda_ipc_arraydict(arr)["data"])["device"] == 0
    )


def test_dump_rejects_non_cuda_array(mocked_cuda):
    with pytest.raises(ValueError, match="cuda_ipc encoding requires a CUDA array"):
        cuda_ipc.dump_cuda_ipc_arraydict(np.zeros((2, 2), dtype=np.float32))


@pytest.mark.parametrize(
    "shape, strides",
    [
        ((50,), (8,)),  # every-other-element view of <f4 (itemsize 4)
        ((4, 3), (4, 16)),  # transposed 3x4 float32
    ],
)
def test_dump_rejects_non_contiguous(mocked_cuda, shape, strides):
    arr = FakeCudaArray(shape, "<f4", strides=strides)
    with pytest.raises(ValueError, match="C-contiguous"):
        cuda_ipc.dump_cuda_ipc_arraydict(arr)


def test_dump_accepts_explicit_contiguous_strides(mocked_cuda):
    # strides given but equal to the row-major strides -> still contiguous.
    arr = FakeCudaArray((3, 4), "<f4", strides=(16, 4))
    out = cuda_ipc.dump_cuda_ipc_arraydict(arr)
    assert out["data"]["encoding"] == "cuda_ipc"


# ── VMM staging fallback (legacy IPC reject) ────────────────────────────


def test_dump_falls_back_to_staging_on_ipc_reject(mocked_cuda):
    """When legacy IPC rejects the allocation, encode stages into a fresh buffer.

    The staged handle uses offset 0 / size == the array's own nbytes, and the
    staging buffer is registered for a later release.
    """
    # Reject the array's pointer (VMM-backed) but let the staging buffer succeed,
    # matching real behavior where the fresh cudaMalloc buffer is IPC-exportable.
    mocked_cuda.reject_foreign_ipc = True

    arr = FakeCudaArray((4, 8), "<f4", data_ptr=0x5000)  # nbytes = 4*8*4 = 128
    out = cuda_ipc.dump_cuda_ipc_arraydict(arr)

    # Only the staging buffer's handle was taken.
    assert mocked_cuda.calls["get_handle"] == [0xD000]
    # The array's own bytes are copied into a fresh buffer.
    assert mocked_cuda.calls["malloc"] == [128]
    assert mocked_cuda.calls["memcpy_d2d"] == [(0xD000, 0x5000, 128)]
    # Payload reflects the staging buffer: offset 0, size == nbytes.
    unpacked = _unpack_cuda_ipc(out["data"])
    assert unpacked["storage_offset"] == 0
    assert unpacked["storage_size"] == 128
    # Staging buffer registered for a later release.
    assert cuda_ipc._DEFAULT_EXPORTS.staging == [
        (0xD000, 0, 128, b"\x01" * cuda_api.IPC_HANDLE_SIZE)
    ]


def test_dump_synchronizes_before_returning_handle(mocked_cuda):
    """The consumer reads from another process, so encode waits for the device."""
    cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((3,), "<f4"))
    assert mocked_cuda.calls["sync"] == [True]


def test_dump_works_on_the_arrays_device(mocked_cuda, monkeypatch):
    """Staging and synchronization run on the array's device, not the caller's."""
    mocked_cuda.reject_foreign_ipc = True
    seen = []

    def record_active_device(name):
        original = getattr(cuda_api, name)

        def wrapper(*args):
            seen.append((name, mocked_cuda.current_device))
            return original(*args)

        monkeypatch.setattr(cuda_api, name, wrapper)

    record_active_device("malloc")
    record_active_device("device_synchronize")

    out = cuda_ipc.dump_cuda_ipc_arraydict(
        FakeCudaArray((4,), "<f4", data_ptr=0x5000, device=_TorchDevice(index=1))
    )

    assert seen == [("malloc", 1), ("device_synchronize", 1), ("device_synchronize", 1)]
    assert mocked_cuda.current_device == 0
    cuda_ipc.release_pinned_ipc_exports()
    assert cuda_ipc._STAGING_POOL.take(1, 16) == (
        0xD000,
        b"\x01" * cuda_api.IPC_HANDLE_SIZE,
    )
    assert _unpack_cuda_ipc(out["data"])["device"] == 1


def test_dump_on_the_active_device_does_not_switch(mocked_cuda):
    # The device is still set once, to make its context current on this thread
    # (a fresh thread has none), but nothing switches or needs restoring.
    cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((3,), "<f4"))
    assert mocked_cuda.calls["set_device"] == [0]


# ── Export registry / ring-1 lifetime ───────────────────────────────────


def test_export_registry_pins_and_releases(mocked_cuda):
    assert cuda_ipc._DEFAULT_EXPORTS.pins == []
    arr = FakeCudaArray((3,), "<f4")
    cuda_ipc.dump_cuda_ipc_arraydict(arr)
    # The source array is retained so its (would-be) GPU memory stays valid.
    assert arr in cuda_ipc._DEFAULT_EXPORTS.pins
    cuda_ipc.release_pinned_ipc_exports()
    assert cuda_ipc._DEFAULT_EXPORTS.pins == []


def test_release_recycles_staging_buffers(mocked_cuda):
    """Released staging buffers are reused, handle included, by the next export."""
    mocked_cuda.reject_foreign_ipc = True
    first = cuda_ipc.dump_cuda_ipc_arraydict(
        FakeCudaArray((4,), "<f4", data_ptr=0x5000)
    )
    cuda_ipc.release_pinned_ipc_exports()
    assert mocked_cuda.calls["free"] == []

    second = cuda_ipc.dump_cuda_ipc_arraydict(
        FakeCudaArray((4,), "<f4", data_ptr=0x6000)
    )

    assert mocked_cuda.calls["malloc"] == [16]
    assert mocked_cuda.calls["get_handle"] == [0xD000]
    assert mocked_cuda.calls["memcpy_d2d"][-1] == (0xD000, 0x6000, 16)
    assert second["data"]["buffer"] == first["data"]["buffer"]


def test_release_frees_staging_beyond_pool_limit(mocked_cuda, monkeypatch):
    """Idle staging buffers beyond the pool's byte limit are freed, oldest first."""
    monkeypatch.setattr(cuda_ipc, "_pool_max_bytes", lambda device: 16)
    mocked_cuda.reject_foreign_ipc = True
    cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((4,), "<f4", data_ptr=0x5000))
    cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((4,), "<f4", data_ptr=0x6000))
    cuda_ipc.release_pinned_ipc_exports()
    assert mocked_cuda.calls["free"] == [0xD000]

    # The next export reuses the pooled buffer, not the freed one.
    cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((4,), "<f4", data_ptr=0x7000))
    assert mocked_cuda.calls["memcpy_d2d"][-1] == (0xE000, 0x7000, 16)
    assert mocked_cuda.calls["malloc"] == [16, 16]


def test_staging_pool_makes_room_for_a_new_size(mocked_cuda, monkeypatch):
    """Releasing a buffer of a new size evicts idle buffers of other sizes to make room."""
    monkeypatch.setattr(cuda_ipc, "_pool_max_bytes", lambda device: 64)
    mocked_cuda.reject_foreign_ipc = True
    cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((4,), "<f4", data_ptr=0x5000))
    cuda_ipc.release_pinned_ipc_exports()
    cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((16,), "<f4", data_ptr=0x6000))
    cuda_ipc.release_pinned_ipc_exports()
    assert mocked_cuda.calls["free"] == [0xD000]

    cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((16,), "<f4", data_ptr=0x7000))
    assert mocked_cuda.calls["malloc"] == [16, 64]
    assert mocked_cuda.calls["memcpy_d2d"][-1] == (0xE000, 0x7000, 64)


@pytest.mark.parametrize(
    "fraction, expected", [(0.25, 1 << 30), (0.5, 2 << 30), (0, 0)]
)
def test_pool_limit_follows_runtime_config(mocked_cuda, fraction, expected):
    """Each of the two pools gets half of ``cuda_ipc_pool_fraction`` of the device."""
    from tesseract_core.runtime.config import override_config, update_config

    with override_config():
        update_config(cuda_ipc_pool_fraction=fraction)
        assert cuda_ipc._pool_max_bytes(0) == expected  # fake device has 8 GiB


def test_zero_pool_fraction_disables_reuse(mocked_cuda):
    from tesseract_core.runtime.config import override_config, update_config

    mocked_cuda.reject_foreign_ipc = True
    with override_config():
        update_config(cuda_ipc_pool_fraction=0)
        cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((4,), "<f4", data_ptr=0x5000))
        cuda_ipc.release_pinned_ipc_exports()
        assert mocked_cuda.calls["free"] == [0xD000]
        cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((4,), "<f4", data_ptr=0x6000))
    assert mocked_cuda.calls["malloc"] == [16, 16]


def test_client_request_releases_input_exports(mocked_cuda, monkeypatch):
    """HTTPClient._request must release the GPU inputs it pinned while encoding.

    Encoding a GPU input pins it in the request's own export group, so that
    concurrent requests do not release each other's inputs. If _request does
    not release afterward each call's inputs leak. The pin must survive until
    the server has decoded the inputs, i.e. until the response body is
    buffered, so we assert it is still present when the (fake) request is
    dispatched, and gone once _request returns.
    """
    from tesseract_core.sdk.tesseract import HTTPClient, ServerCapabilities

    seen_during_request = {}

    response = Mock(status_code=200, ok=True, content=b"{}")

    class FakeSession:
        def __init__(self) -> None:
            self.headers = {}

        def request(self, **kwargs):
            # The input must still be pinned here: a real server has not yet
            # decoded and copied it out.
            seen_during_request["pinned"] = list(groups[0].pins)
            return response

    client = HTTPClient.__new__(HTTPClient)
    client._url = "http://localhost:8000"
    client._output_path = None
    client._output_format = "json+base64"
    client._gpu_transport = "cuda_ipc"
    client._timeout = None
    client._session = FakeSession()
    client.server_capabilities = ServerCapabilities(
        ("json+base64",), ("none", "cuda_ipc"), ("none",)
    )

    groups = _record_export_groups(monkeypatch)
    arr = FakeCudaArray((3,), "<f4")
    client._request("apply", method="POST", payload={"a": arr})

    # Pinned during the request (so the server can copy it out) ...
    assert arr in seen_during_request["pinned"]
    # ... and released once the request returned (no leak across calls).
    assert len(groups) == 1
    assert len(groups[0]) == 0
    assert cuda_ipc._DEFAULT_EXPORTS.pins == []


def test_client_request_cpu_only_payload_skips_release(monkeypatch):
    """A cuda_ipc-transport request with no GPU inputs skips the release path.

    Nothing gets pinned, so _request must not import/call the cuda_ipc runtime
    for cleanup -- otherwise a base install (no runtime extra) would spuriously
    fail on an all-CPU payload. Guard by making the runtime import explode.
    """
    from tesseract_core.sdk import tesseract as sdk
    from tesseract_core.sdk.tesseract import HTTPClient, ServerCapabilities

    def _boom():
        raise AssertionError(
            "the cuda_ipc runtime must not be used for a CPU-only payload"
        )

    monkeypatch.setattr(sdk, "_import_cuda_ipc", _boom)

    response = Mock(status_code=200, ok=True, content=b"{}")

    class FakeSession:
        def __init__(self) -> None:
            self.headers = {}

        def request(self, **kwargs):
            return response

    client = HTTPClient.__new__(HTTPClient)
    client._url = "http://localhost:8000"
    client._output_path = None
    client._output_format = "json+base64"
    client._gpu_transport = "cuda_ipc"
    client._timeout = None
    client._session = FakeSession()
    client.server_capabilities = ServerCapabilities(
        ("json+base64",), ("none", "cuda_ipc"), ("none",)
    )

    # Plain host array -> encodes as base64, pins nothing, releases nothing.
    client._request("apply", method="POST", payload={"a": np.zeros(3)})


def test_import_cuda_ipc_explains_missing_runtime_extra(monkeypatch):
    """Without the runtime extra, cuda_ipc use points at tesseract-core[runtime].

    Simulate a base SDK install (no runtime deps) by making the cuda_ipc import
    fail, and assert the friendly ImportError naming the extra is raised instead
    of a bare ModuleNotFoundError from deep in the import chain.
    """
    import builtins

    from tesseract_core.sdk import tesseract as sdk

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Mimic the module being unimportable on a base install (its deep
        # dependencies, e.g. fsspec, are absent). Covers `from
        # tesseract_core.runtime.cuda import ipc`.
        if name == "tesseract_core.runtime.cuda.ipc" or (
            name == "tesseract_core.runtime.cuda" and "ipc" in (fromlist or ())
        ):
            raise ImportError("No module named 'fsspec'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"tesseract-core\[runtime\]"):
        sdk._import_cuda_ipc()


def test_decode_cuda_ipc_failure_gives_actionable_error(monkeypatch):
    """A cuda_ipc response this client can't open yields a helpful RuntimeError.

    cuda_ipc is opt-in, so a cuda_ipc-encoded array only comes back when the
    caller asked for it. If this process has no usable CUDA context (no driver,
    no matching device, or the runtime extra missing), opening the handle fails
    deep in the runtime; the SDK decode must translate that into a message
    naming the fix (drop the transport to get a host copy) instead of leaking a
    bare CUDA/import error.
    """
    from tesseract_core.sdk import tesseract as sdk

    def boom_load(_val):
        raise RuntimeError("cudaIpcOpenMemHandle failed: simulated")

    monkeypatch.setattr(
        sdk,
        "_import_cuda_ipc",
        lambda: types.SimpleNamespace(load_cuda_ipc_arraydict=boom_load),
    )

    encoded = _encoded((2,), "float32", device=0, offset=0, storage_size=8)
    with pytest.raises(RuntimeError, match="gpu_transport='cuda_ipc'"):
        sdk._decode_array(encoded)


# ── Decode-side orchestration (mocked CUDA, no CuPy) ─────────────────────
#
# Decoding no longer depends on CuPy: it uses only the plain-Python CUDA runtime
# primitives (set_device/malloc/memcpy/device_synchronize/free) plus the IPC
# open/close helpers. The mocked_cuda fixture replaces those so the *Python
# orchestration* (offset arithmetic, own-nbytes copy, synchronize-before-close,
# mapping close, buffer free, DLPack ownership) is exercised on a GPU-less box;
# the real device-copy correctness lives in the GPU tests in test_cuda_ipc.py.


def _encoded(shape, dtype, device, offset, storage_size, fill=b"\x02"):
    import pybase64

    handle = pybase64.b64encode_as_string(fill * cuda_api.IPC_HANDLE_SIZE)
    return {
        "object_type": "array",
        "shape": list(shape),
        "dtype": dtype,
        "data": {
            "buffer": f"{device}:{handle}:{offset}:{storage_size}",
            "encoding": "cuda_ipc",
        },
    }


def test_load_copies_own_bytes_at_offset_and_closes(mocked_cuda):
    """Decode allocates the array's own nbytes, copies from base+offset, closes."""
    handle = b"\x02" * cuda_api.IPC_HANDLE_SIZE
    encoded = _encoded((4, 8), "float32", device=1, offset=128, storage_size=4096)

    out = cuda_ipc.load_cuda_ipc_arraydict(encoded)

    nbytes = 4 * 8 * 4  # 128
    # Opened on the requested device with the decoded handle.
    assert mocked_cuda.calls["open"] == [(handle, 1)]
    # Allocated exactly the array's own byte size (not the whole storage_size).
    assert mocked_cuda.calls["malloc"] == [nbytes]
    # One device->device copy of nbytes, from base(0x2000)+offset(128) into the
    # owned buffer (0xD000).
    assert len(mocked_cuda.calls["memcpy_d2d"]) == 1
    dst, src, size = mocked_cuda.calls["memcpy_d2d"][0]
    assert dst == 0xD000
    assert src == 0x2000 + 128
    assert size == nbytes
    # Synchronised before the mapping was closed.
    assert mocked_cuda.calls["sync"] == [True]
    assert mocked_cuda.calls["close"] == [0x2000]
    # The caller's active device is restored.
    assert mocked_cuda.current_device == 0
    # Returned wrapper is framework-agnostic and correctly shaped.
    assert isinstance(out, cuda_ipc.IpcDeviceArray)
    assert out.shape == (4, 8)
    assert out.dtype == np.float32
    assert hasattr(out, "__cuda_array_interface__")
    assert hasattr(out, "__dlpack__")
    iface = out.__cuda_array_interface__
    assert iface["version"] == 3
    assert iface["data"] == (0xD000, False)
    assert iface["strides"] is None
    assert iface["typestr"] == np.dtype("float32").str


def test_load_reuses_owned_buffer_released_on_del(mocked_cuda):
    """A decoded array's buffer is reused by the next decode of the same size."""
    encoded = _encoded((2,), "float32", device=0, offset=0, storage_size=8)
    out = cuda_ipc.load_cuda_ipc_arraydict(encoded)
    del out
    import gc

    gc.collect()
    assert mocked_cuda.calls["free"] == []

    cuda_ipc.load_cuda_ipc_arraydict(encoded)
    assert mocked_cuda.calls["malloc"] == [8]
    assert mocked_cuda.calls["memcpy_d2d"][-1] == (0xD000, 0x2000, 8)
    # One sync after each copy, plus one before copying into the reused buffer.
    assert mocked_cuda.calls["sync"] == [True] * 3


def test_load_reuses_owned_buffer_released_by_dlpack_deleter(mocked_cuda):
    """A buffer handed out via DLPack returns to the pool once its capsule is gone.

    The capsule keeps the array alive even after the array's last other
    reference is dropped, so the buffer is reused only after both are gone.
    """
    import gc

    encoded = _encoded((2,), "float32", device=0, offset=0, storage_size=8)
    out = cuda_ipc.load_cuda_ipc_arraydict(encoded)
    capsule = out.__dlpack__()
    del out
    gc.collect()
    cuda_ipc.load_cuda_ipc_arraydict(encoded)
    assert mocked_cuda.calls["malloc"] == [8, 8]

    # Nobody consumed the capsule, so destroying it runs the deleter.
    del capsule
    gc.collect()
    assert mocked_cuda.calls["free"] == []
    cuda_ipc.load_cuda_ipc_arraydict(encoded)
    assert mocked_cuda.calls["malloc"] == [8, 8]


def test_owned_buffers_beyond_pool_limit_are_freed(mocked_cuda, monkeypatch):
    monkeypatch.setattr(cuda_ipc, "_pool_max_bytes", lambda device: 0)
    out = cuda_ipc.load_cuda_ipc_arraydict(
        _encoded((2,), "float32", device=0, offset=0, storage_size=8)
    )
    del out
    import gc

    gc.collect()
    assert mocked_cuda.calls["free"] == [0xD000]


def test_load_closes_handle_even_on_copy_failure(mocked_cuda, monkeypatch):
    """The IPC mapping is released and the owned buffer freed if the copy fails."""

    def boom_memcpy(dst, src, nbytes):
        raise RuntimeError("cudaMemcpy (device->device) failed: simulated")

    monkeypatch.setattr(cuda_api, "memcpy_device_to_device", boom_memcpy)

    with pytest.raises(RuntimeError, match="cudaMemcpy"):
        cuda_ipc.load_cuda_ipc_arraydict(
            _encoded((2,), "float32", device=0, offset=0, storage_size=8)
        )
    # Owned buffer freed and the IPC mapping closed despite the failure.
    assert mocked_cuda.calls["free"] == [0xD000]
    assert mocked_cuda.calls["close"] == [0x2000]


def test_load_frees_owned_buffer_on_open_failure(mocked_cuda, monkeypatch):
    """A failed IPC open still frees the already-allocated owned buffer.

    The owned buffer is allocated before the mapping is opened; if the open
    fails there is no mapping to close, but the owned buffer must not leak.
    """

    def boom_open(handle_bytes):
        raise RuntimeError("cudaIpcOpenMemHandle failed: simulated")

    monkeypatch.setattr(cuda_api, "ipc_open_mem_handle", boom_open)

    with pytest.raises(RuntimeError, match="cudaIpcOpenMemHandle"):
        cuda_ipc.load_cuda_ipc_arraydict(
            _encoded((2,), "float32", device=0, offset=0, storage_size=8)
        )
    # Owned buffer freed; nothing to close since the mapping never opened.
    assert mocked_cuda.calls["free"] == [0xD000]
    assert mocked_cuda.calls["close"] == []


def test_copy_to_host_reads_device_bytes(mocked_cuda, allow_device_host_copy):
    """copy_to_host performs a device->host memcpy + sync and returns the bytes."""
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    mocked_cuda.device_bytes = expected.tobytes()

    out = cuda_ipc.load_cuda_ipc_arraydict(
        _encoded((2, 3), "float32", device=0, offset=0, storage_size=24)
    )
    host = out.copy_to_host()
    np.testing.assert_array_equal(host, expected)
    # A device->host copy happened.
    assert len(mocked_cuda.calls["memcpy_d2h"]) == 1
    # np.asarray goes through __array__ -> copy_to_host too.
    np.testing.assert_array_equal(np.asarray(out), expected)


# ── GPU-array validation (no CUDA calls at all) ─────────────────────────


def test_validate_cuda_array_passthrough():
    """A matching GPU array validates and is returned unchanged (not copied)."""
    arr = FakeCudaArray((4, 8), "<f4")
    assert cuda_ipc.validate_cuda_array(arr, (None, 8), "float32") is arr
    # Ellipsis shape means "no shape check".
    assert cuda_ipc.validate_cuda_array(arr, ..., None) is arr


@pytest.mark.parametrize(
    "expected_shape, expected_dtype, match",
    [
        ((4, 4), "float32", "shape"),  # dim mismatch
        ((4, 8, 1), "float32", "shape"),  # rank mismatch
        ((None, 8), "float64", "dtype"),  # dtype mismatch
    ],
)
def test_validate_cuda_array_rejections(expected_shape, expected_dtype, match):
    from pydantic_core import PydanticCustomError

    arr = FakeCudaArray((4, 8), "<f4")
    with pytest.raises(PydanticCustomError, match=match):
        cuda_ipc.validate_cuda_array(arr, expected_shape, expected_dtype)


# ── encode_array dispatch (Python-side, no CUDA calls) ──────────────────


def _info(json_mode: bool, ctx: dict):
    return types.SimpleNamespace(context=ctx, mode_is_json=lambda: json_mode)


def test_encode_array_cuda_ipc_falls_back_to_host_for_cpu_array():
    """A host array under a cuda_ipc device transport falls back to host encoding.

    array_encoding (CPU) and device_transport (GPU) are orthogonal: a GPU leaf is
    exported by handle, but a plain CPU leaf in the same response is serialized
    over the host encoding (base64 here) rather than failing the whole response.
    """
    import pybase64

    out = array_encoding.encode_array(
        np.arange(3, dtype=np.int64),
        _info(True, {"array_encoding": "base64", "device_transport": "cuda_ipc"}),
        (None,),
        "int64",
    )
    assert out["data"]["encoding"] == "base64"
    decoded = np.frombuffer(pybase64.b64decode(out["data"]["buffer"]), dtype=np.int64)
    np.testing.assert_array_equal(decoded, np.arange(3))


def test_encode_array_cuda_ipc_exports_gpu_leaf(mocked_cuda):
    """A GPU leaf under a cuda_ipc device transport is exported by handle."""
    out = array_encoding.encode_array(
        FakeCudaArray((3,), "<f4"),
        _info(True, {"array_encoding": "base64", "device_transport": "cuda_ipc"}),
        (None,),
        "float32",
    )
    assert out["data"]["encoding"] == "cuda_ipc"


def test_output_to_bytes_mixed_gpu_and_cpu_arrays(mocked_cuda):
    """A single response carries a GPU leaf and a CPU leaf together.

    The GPU array is exported over the cuda_ipc device transport; the plain host
    array in the same model falls back to the base64 host encoding.

    The config only makes cuda_ipc *available*; the per-call ``gpu_transport``
    kwarg is what *selects* it for this response.
    """
    import orjson
    import pybase64
    from pydantic import BaseModel

    from tesseract_core.runtime import config
    from tesseract_core.runtime.file_interactions import output_to_bytes
    from tesseract_core.runtime.schema_types import Array, Float32

    config.update_config(gpu_transport="cuda_ipc")

    class MixedModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        gpu: Array[(3,), Float32]
        cpu: Array[(3,), Float32]

    model = MixedModel.model_construct(
        gpu=FakeCudaArray((3,), "<f4"),
        cpu=np.arange(3, dtype=np.float32),
    )
    payload = orjson.loads(
        output_to_bytes(model, "json+base64", gpu_transport="cuda_ipc")
    )

    assert payload["gpu"]["data"]["encoding"] == "cuda_ipc"
    assert payload["cpu"]["data"]["encoding"] == "base64"
    decoded_cpu = np.frombuffer(
        pybase64.b64decode(payload["cpu"]["data"]["buffer"]), dtype=np.float32
    )
    np.testing.assert_array_equal(decoded_cpu, np.arange(3, dtype=np.float32))


def _record_export_groups(monkeypatch):
    """Record the ExportGroups created from now on."""
    groups = []
    init = cuda_ipc.ExportGroup.__init__

    def recording_init(self):
        init(self)
        groups.append(self)

    monkeypatch.setattr(cuda_ipc.ExportGroup, "__init__", recording_init)
    return groups


def test_encode_payload_mixed_gpu_and_binref(mocked_cuda, tmp_path, monkeypatch):
    """The SDK client encodes mixed GPU and binref input arrays in a single request.

    GPU leaves are exported by handle via cuda_ipc (without host copies), while host
    leaves are written as .bin files to the input directory. Both the pinned GPU
    allocations and the temporary disk files are released on context exit.
    """
    from tesseract_core.sdk.tesseract import _encode_payload

    groups = _record_export_groups(monkeypatch)
    payload = {
        "gpu": FakeCudaArray((3,), "<f4"),
        "cpu": np.arange(3, dtype=np.float32),
    }

    bin_file = None
    with _encode_payload(
        payload,
        gpu_transport="cuda_ipc",
        input_path=tmp_path,
        output_format="json+binref",
    ) as encoded:
        assert encoded["gpu"]["data"]["encoding"] == "cuda_ipc"
        assert encoded["cpu"]["data"]["encoding"] == "binref"
        bin_name = encoded["cpu"]["data"]["buffer"].split(":")[0]
        bin_file = tmp_path / bin_name
        assert bin_file.exists()
        # This request's group tracks the exported allocation during context
        assert len(groups) == 1
        assert len(groups[0].pins) == 1

    # Context exit should unlink disk files and release pinned allocations
    assert not bin_file.exists()
    assert len(groups[0]) == 0
    assert cuda_ipc._DEFAULT_EXPORTS.pins == []


def test_encode_array_cuda_ipc_missing_context_raises(mocked_cuda):
    """_encode_array with encoding='cuda_ipc' on a GPU array requires EncodingContext."""
    from tesseract_core.sdk.tesseract import _encode_array

    gpu_arr = FakeCudaArray((3,), "<f4")
    with pytest.raises(
        ValueError, match="EncodingContext is required when encoding is 'cuda_ipc'"
    ):
        _encode_array(gpu_arr, encoding="cuda_ipc")


def test_encode_payload_mixed_gpu_and_binref_cleanup_on_exception(
    mocked_cuda, tmp_path, monkeypatch
):
    """Context exit releases both CUDA IPC exports and binref files even on error."""
    from tesseract_core.sdk.tesseract import _encode_payload

    groups = _record_export_groups(monkeypatch)
    payload = {
        "gpu": FakeCudaArray((3,), "<f4"),
        "cpu": np.arange(3, dtype=np.float32),
    }

    bin_file = None
    with (
        pytest.raises(RuntimeError, match="simulated failure during request"),
        _encode_payload(
            payload,
            gpu_transport="cuda_ipc",
            input_path=tmp_path,
            output_format="json+binref",
        ) as encoded,
    ):
        bin_name = encoded["cpu"]["data"]["buffer"].split(":")[0]
        bin_file = tmp_path / bin_name
        assert bin_file.exists()
        assert len(groups[0].pins) == 1
        raise RuntimeError("simulated failure during request")

    # Context exit should unlink disk files and release pinned allocations despite the error
    assert not bin_file.exists()
    assert len(groups[0]) == 0


def test_cuda_array_to_host_branches(allow_device_host_copy):
    """cuda_array_to_host handles CuPy-, torch-, __array__-like, and rejects others."""

    class CupyLike:
        def get(self):
            return np.array([1.0, 2.0])

    class TorchLike:
        def cpu(self):
            return self

        def numpy(self):
            return np.array([3.0, 4.0])

    class ArrayLike:
        # Mirrors JAX arrays, which expose neither .get() nor .cpu() but fetch to
        # host via __array__ / np.asarray.
        def __array__(self, dtype=None):
            out = np.array([5.0, 6.0])
            return out if dtype is None else out.astype(dtype)

    assert cuda_ipc.cuda_array_to_host(CupyLike()).tolist() == [1.0, 2.0]
    assert cuda_ipc.cuda_array_to_host(TorchLike()).tolist() == [3.0, 4.0]
    assert cuda_ipc.cuda_array_to_host(ArrayLike()).tolist() == [5.0, 6.0]
    with pytest.raises(TypeError, match="Cannot copy GPU array"):
        cuda_ipc.cuda_array_to_host(object())


def test_forbid_device_host_copy_blocks_implicit_copies(mocked_cuda):
    """Implicit host copies raise in the runtime and SDK, explicit ones don't.

    The flag is set by the autouse ``forbid_device_host_copy`` fixture.
    """
    from tesseract_core.sdk.tesseract import _encode_array

    match = "TESSERACT_FORBID_DEVICE_HOST_COPY"
    with pytest.raises(RuntimeError, match=match):
        array_encoding.encode_array(
            FakeCudaArray((3,), "<f4"),
            _info(True, {"array_encoding": "base64"}),
            (None,),
            "float32",
        )
    with pytest.raises(RuntimeError, match=match):
        _encode_array(FakeCudaArray((3,), "<f4"), encoding="base64")

    mocked_cuda.device_bytes = np.arange(3, dtype=np.float32).tobytes()
    decoded = cuda_ipc.load_cuda_ipc_arraydict(
        _encoded((3,), "float32", device=0, offset=0, storage_size=12)
    )
    with pytest.raises(RuntimeError, match=match):
        np.asarray(decoded)
    np.testing.assert_array_equal(decoded.copy_to_host(), np.arange(3))


# ── GPU-transport gating ────────────────────────────────────────────────


def test_output_to_bytes_rejects_cuda_ipc_transport_by_default():
    """Without a configured gpu_transport, cuda_ipc is not an accepted transport."""
    from tesseract_core.runtime import config, file_interactions

    config.update_config(gpu_transport="none")
    with pytest.raises(ValueError, match=r"Unsupported GPU transport cuda_ipc"):
        file_interactions.output_to_bytes(
            {"y": 1}, "json+base64", gpu_transport="cuda_ipc"
        )


def test_available_gpu_transports_reflects_config():
    from tesseract_core.runtime import config
    from tesseract_core.runtime.file_interactions import available_gpu_transports

    config.update_config(gpu_transport="none")
    assert available_gpu_transports() == ("none",)

    config.update_config(gpu_transport="cuda_ipc")
    assert "cuda_ipc" in available_gpu_transports()


def test_output_formats_never_include_cuda_ipc():
    """The host-array output formats are the three stable ones, always."""
    from tesseract_core.runtime.file_interactions import available_formats

    assert available_formats() == ("json", "json+base64", "json+binref")


# ── format + transport -> encoding-context mapping ──────────────────────


def test_output_to_bytes_splits_host_encoding_and_device_transport(monkeypatch):
    """Format sets array_encoding (CPU); gpu_transport sets device_transport (GPU)."""
    from tesseract_core.runtime import config, file_interactions

    config.update_config(gpu_transport="cuda_ipc")
    captured = {}

    class FakeAdapter:
        def __init__(self, _type):
            pass

        def dump_python(self, obj, mode, context, exclude_unset):
            captured["context"] = context
            return {}

    monkeypatch.setattr(file_interactions, "TypeAdapter", FakeAdapter)
    monkeypatch.setattr(file_interactions.orjson, "dumps", lambda d: b"{}")

    file_interactions.output_to_bytes({"y": 1}, "json+base64", gpu_transport="cuda_ipc")
    assert captured["context"] == {
        "array_encoding": "base64",
        "compression": None,
        "device_transport": "cuda_ipc",
        "device_exports": None,
    }

    # Default gpu_transport leaves device_transport unset (None).
    file_interactions.output_to_bytes({"y": 1}, "json")
    assert captured["context"] == {
        "array_encoding": "json",
        "device_transport": None,
        "device_exports": None,
    }


# ── libcudart discovery (wheel-installed CUDA) ──────────────────────────
#
# The pip CUDA wheels (nvidia-cuda-runtime-cuXX, pulled in by jax[cudaXX] /
# cupy-cudaXXx) install libcudart under site-packages/nvidia/cuda_runtime/lib/,
# which is on neither LD_LIBRARY_PATH nor the ldconfig cache. Discovery lives in
# tesseract_core.runtime.cuda.loader and must still find it there after the
# system-loader probes come up empty.


def _fake_cudart_wheel(tmp_path, soname="libcudart.so.12"):
    """Create a fake ``nvidia/cuda_runtime/lib/<soname>`` layout; return its lib dir."""
    lib_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / soname).write_bytes(b"")
    return lib_dir


def test_iter_wheel_cudart_paths_finds_runtime_wheel(tmp_path, monkeypatch):
    """The runtime wheel's lib dir is discovered via its importlib spec."""
    lib_dir = _fake_cudart_wheel(tmp_path)

    real_find_spec = loader.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "nvidia.cuda_runtime":
            spec = types.SimpleNamespace()
            spec.submodule_search_locations = [str(lib_dir.parent)]
            return spec
        return real_find_spec(name)

    monkeypatch.setattr(loader.importlib.util, "find_spec", fake_find_spec)

    found = list(loader._iter_wheel_cudart_paths())
    assert str(lib_dir / "libcudart.so.12") in found


def test_find_cudart_prefers_wheel_over_system(tmp_path, monkeypatch):
    """The wheel copy is loaded even when a system soname would also resolve.

    The wheel is tried before the system loader, so on a host with both a venv
    runtime and a system one the venv copy wins -- matching how JAX/torch load
    libcudart.
    """
    lib_dir = _fake_cudart_wheel(tmp_path)
    wheel_path = str(lib_dir / "libcudart.so.12")

    # A system runtime is also resolvable, so both sources could satisfy the load.
    monkeypatch.setattr(
        loader.ctypes.util,
        "find_library",
        lambda name: "libcudart.so.12" if name == "cudart" else None,
    )
    monkeypatch.setattr(loader, "_iter_wheel_cudart_paths", lambda: [wheel_path])

    # Everything loads; _find_cudart returns the first candidate it tries.
    loaded = {}

    def fake_cdll(path):
        loaded["path"] = path
        return object()

    monkeypatch.setattr(loader.ctypes, "CDLL", fake_cdll)

    handle = loader._find_cudart()
    assert handle is not None
    # The wheel path is first in the candidate order, so it is what gets loaded.
    assert loaded["path"] == wheel_path


def test_iter_wheel_cudart_paths_finds_unenumerated_future_major(tmp_path, monkeypatch):
    """A wheel shipping a CUDA major we never hardcoded is still discovered.

    The wheel search globs by filename rather than probing a fixed set, so a
    runtime released after this code was written (here ``.so.99``) is picked up
    without any change to the version lists.
    """
    future_soname = f"libcudart.so.{loader.CUDART_MAJOR_NEWEST + 79}"
    lib_dir = _fake_cudart_wheel(tmp_path, soname=future_soname)

    real_find_spec = loader.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "nvidia.cuda_runtime":
            spec = types.SimpleNamespace()
            spec.submodule_search_locations = [str(lib_dir.parent)]
            return spec
        return real_find_spec(name)

    monkeypatch.setattr(loader.importlib.util, "find_spec", fake_find_spec)

    found = list(loader._iter_wheel_cudart_paths())
    assert str(lib_dir / future_soname) in found


def test_iter_wheel_cudart_paths_prefers_newest_major(tmp_path, monkeypatch):
    """When a wheel lib dir holds several runtimes, the newest major comes first."""
    lib_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
    lib_dir.mkdir(parents=True)
    for soname in ("libcudart.so.11", "libcudart.so.13", "libcudart.so.12"):
        (lib_dir / soname).write_bytes(b"")

    real_find_spec = loader.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "nvidia.cuda_runtime":
            spec = types.SimpleNamespace()
            spec.submodule_search_locations = [str(lib_dir.parent)]
            return spec
        return real_find_spec(name)

    monkeypatch.setattr(loader.importlib.util, "find_spec", fake_find_spec)

    found = list(loader._iter_wheel_cudart_paths())
    assert found[0] == str(lib_dir / "libcudart.so.13")


# ── iter_cudart_candidates (public discovery surface) ───────────────────
#
# Public API: out-of-process consumers (the tesseract_jax C++ shim) dlopen
# libcudart using these candidates, so its contract -- ordering, dedup, and that
# every item is a bare soname or absolute path -- is part of the interface. It is
# re-exported from cuda_ipc for those consumers, but defined in cuda.loader.


def test_iter_cudart_candidates_orders_wheel_then_system(monkeypatch):
    """Wheel paths come first, then find_library hits, then bare sonames.

    Wheel-first matches how JAX and PyTorch load libcudart (venv copy over a
    system one), so a codec handing device memory to them agrees on the runtime.
    """
    monkeypatch.setattr(
        loader.ctypes.util,
        "find_library",
        lambda name: "/usr/lib/libcudart.so.12" if name == "cudart" else None,
    )
    monkeypatch.setattr(
        loader,
        "_iter_wheel_cudart_paths",
        lambda: ["/wheel/nvidia/lib/libcudart.so.12"],
    )

    candidates = list(loader.iter_cudart_candidates())

    # Wheel (venv) path leads, ahead of the system loader's resolved path.
    assert candidates[0] == "/wheel/nvidia/lib/libcudart.so.12"
    assert candidates.index("/wheel/nvidia/lib/libcudart.so.12") < candidates.index(
        "/usr/lib/libcudart.so.12"
    )
    # Bare sonames trail the find_library hits.
    assert "libcudart.so" in candidates
    assert candidates.index("/usr/lib/libcudart.so.12") < candidates.index(
        "libcudart.so"
    )


def test_iter_cudart_candidates_dedups_preserving_order(monkeypatch):
    """A path surfaced by both find_library and the wheel search appears once."""
    dup = "/wheel/nvidia/lib/libcudart.so.12"
    monkeypatch.setattr(
        loader.ctypes.util,
        "find_library",
        lambda name: dup if name == "cudart" else None,
    )
    monkeypatch.setattr(loader, "_iter_wheel_cudart_paths", lambda: [dup])

    candidates = list(loader.iter_cudart_candidates())

    assert candidates.count(dup) == 1
    # The earlier (wheel) occurrence wins its position.
    assert candidates[0] == dup


def test_iter_cudart_candidates_are_dlopen_arguments(monkeypatch):
    """Every candidate is a bare soname or an absolute path (never a stem).

    C++ consumers pass these straight to dlopen/LoadLibrary, which need a real
    library name or path -- not a find_library stem like ``"cudart"``.
    """
    monkeypatch.setattr(loader.ctypes.util, "find_library", lambda name: None)
    monkeypatch.setattr(loader, "_iter_wheel_cudart_paths", list)

    for candidate in loader.iter_cudart_candidates():
        # A real soname (lib*.so*/lib*.dylib), a Windows DLL, or an absolute
        # path -- never a bare find_library stem like "cudart" / "cudart64_12".
        is_soname = candidate.startswith("lib") or candidate.endswith(".dll")
        is_path = candidate.startswith("/")
        assert is_soname or is_path, candidate


def test_iter_cudart_candidates_exported_from_cuda_package():
    """The discovery surface is importable from the cuda package for FFI consumers."""
    from tesseract_core.runtime import cuda

    assert cuda.iter_cudart_candidates is loader.iter_cudart_candidates


# ── cuda_ipc as a DeviceTransport backend ────────────────────────────────
#
# cuda_ipc is exposed through the shared DeviceTransport interface so further
# transports slot in behind one lookup. These check that the cuda_ipc backend
# is registered and routes to the same functions the direct API uses; the
# transport-agnostic registry machinery is covered in test_device_transport.py.


def test_cuda_ipc_registered_as_transport():
    """The cuda_ipc backend is discoverable by name and satisfies the interface."""
    from tesseract_core.runtime.device_transport import DeviceTransport, get_transport

    transport = get_transport("cuda_ipc")
    assert transport.name == "cuda_ipc"
    assert transport.reach == "same_host"
    assert isinstance(transport, DeviceTransport)


def test_cuda_ipc_transport_receive_materialises_wrapper(mocked_cuda):
    """The cuda_ipc transport's receive() decodes into an on-GPU wrapper.

    This is the decode seam array_encoding.decode_array drives for cuda_ipc; the
    end-to-end decode through decode_array is covered by the GPU suite (its
    return type is the framework-agnostic wrapper, outside decode_array's
    host-array return annotation).
    """
    from tesseract_core.runtime.device_transport import get_transport

    transport = get_transport("cuda_ipc")
    encoded = _encoded((2, 3), "float32", device=0, offset=0, storage_size=24)

    out = transport.receive(encoded)

    assert isinstance(out, cuda_ipc.IpcDeviceArray)
    assert out.shape == (2, 3)
    assert out.dtype == np.float32


def test_cuda_ipc_transport_delegates(mocked_cuda, monkeypatch):
    """register/descriptor/flush/receive/release drive the same cuda_ipc code.

    A pull transport's flush is a no-op and its bootstrap needs no shared state,
    so those return None; register+descriptor produce the same payload the direct
    dump does, and release drops the export pins.
    """
    from tesseract_core.runtime.device_transport import get_transport

    transport = get_transport("cuda_ipc")

    # bootstrap + flush are no-ops for a receiver-driven transport.
    assert transport.bootstrap("producer", None) is None
    assert transport.flush() is None

    arr = FakeCudaArray((4, 8), "<f4", data_ptr=0x5000)
    payload = transport.descriptor(transport.register(arr))
    # Same structure the direct dump produces (offset/size from the fake base).
    assert payload["data"]["encoding"] == "cuda_ipc"
    assert _unpack_cuda_ipc(payload["data"])["storage_offset"] == 256
    # register pinned the source array; release drops it.
    assert arr in cuda_ipc._DEFAULT_EXPORTS.pins
    transport.release()
    assert cuda_ipc._DEFAULT_EXPORTS.pins == []


# ── Empty arrays, dtypes without a NumPy equivalent, finalizers in the pool ──


def test_empty_array_crosses_without_a_handle(mocked_cuda):
    """An empty array, which frameworks give a null pointer, crosses without a handle."""
    from tesseract_core.runtime.array_encoding import CudaIpcArrayData

    out = cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((0, 3), "<f4", data_ptr=0))
    assert out["data"]["buffer"] == "0::0:0"
    assert mocked_cuda.calls["get_handle"] == []
    CudaIpcArrayData(**out["data"])  # the wire schema accepts it

    decoded = cuda_ipc.load_cuda_ipc_arraydict(out)
    assert decoded.shape == (0, 3)
    assert decoded.dtype == np.float32
    assert mocked_cuda.calls["open"] == []
    assert mocked_cuda.calls["malloc"] == []
    assert decoded.copy_to_host().shape == (0, 3)


def test_descriptor_without_a_handle_must_be_for_an_empty_array(mocked_cuda):
    encoded = _encoded((2,), "float32", device=0, offset=0, storage_size=0)
    encoded["data"]["buffer"] = "0::0:0"
    with pytest.raises(ValueError, match="no handle"):
        cuda_ipc.load_cuda_ipc_arraydict(encoded)


def test_dtype_without_numpy_equivalent_is_refused(mocked_cuda):
    # PyTorch describes bfloat16 as a 2-byte void dtype.
    with pytest.raises(TypeError, match="no NumPy equivalent"):
        cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((3,), "|V2"))


def test_strided_arrays_are_copied_on_the_device_by_their_framework():
    # cuda_ipc moves a flat byte range, so a strided array is first made
    # contiguous by the framework that made it, never through the host.
    contiguous = FakeCudaArray((3, 2), "<f4")
    assert cuda_ipc._contiguous_on_device(contiguous) is contiguous

    class TorchLike(FakeCudaArray):
        def contiguous(self):
            return contiguous

    class CuPyLike(FakeCudaArray):
        def copy(self, order="K"):
            assert order == "C"
            return contiguous

    for cls in (TorchLike, CuPyLike):
        strided = cls((3, 2), "<f4", strides=(4, 12))
        assert cuda_ipc._contiguous_on_device(strided) is contiguous


def test_release_while_the_pool_lock_is_held_does_not_wait(mocked_cuda):
    """A finalizer run by a GC pass inside take() or put() must not deadlock.

    The buffer it hands back joins the pool at the next take() or put().
    """
    import threading

    pool = cuda_ipc._BufferPool()

    def finalizer_inside_the_pool():
        with pool._lock:
            pool.release(0xA000, 0, 8)

    worker = threading.Thread(target=finalizer_inside_the_pool, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "release() waited for the pool's own lock"
    assert pool.take(0, 8) == (0xA000, None)


# ── Keeping a response's exports until its client is done with them ─────────

_GPU_OUTPUT_API = '''
from pydantic import BaseModel
from tesseract_core.runtime import Array, Float32


class _GpuArray:
    """A GPU array's metadata surface, as the mocked CUDA runtime expects."""

    def __init__(self, n):
        self.__cuda_array_interface__ = {
            "shape": (n,), "typestr": "<f4", "data": (0x1000, False),
            "strides": None, "version": 3,
        }


class InputSchema(BaseModel):
    n: int


class OutputSchema(BaseModel):
    y: Array[(None,), Float32]


def apply(inputs: InputSchema) -> OutputSchema:
    return {"y": _GpuArray(inputs.n)}
'''


def test_server_keeps_response_exports_until_their_client_is_done(
    mocked_cuda, tmp_path, monkeypatch
):
    """A response's exports outlive other clients' requests, until named or expired.

    Otherwise one client's request could free outputs another client has not
    read yet.
    """
    import collections

    from fastapi.testclient import TestClient

    from tesseract_core.runtime import serve
    from tesseract_core.runtime.config import override_config, update_config
    from tesseract_core.runtime.core import load_module_from_path

    monkeypatch.setattr(serve, "_PENDING_EXPORTS", collections.OrderedDict())
    api_path = tmp_path / "tesseract_api.py"
    api_path.write_text(_GPU_OUTPUT_API)
    done = serve.EXPORTS_DONE_HEADER

    with override_config():
        update_config(gpu_transport="cuda_ipc")
        client = TestClient(serve.create_rest_api(load_module_from_path(api_path)))

        def apply(**headers):
            response = client.post(
                "/apply",
                json={"inputs": {"n": 4}},
                headers={
                    "Accept": "application/json; gpu_transport=cuda_ipc",
                    **headers,
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["y"]["data"]["encoding"] == "cuda_ipc"
            export_id = response.headers[serve.EXPORTS_HEADER]
            return export_id, serve._PENDING_EXPORTS[export_id][0]

        first_id, first = apply(**{done: ""})
        assert len(first.pins) == 1
        # Another acknowledging client's request leaves them alone ...
        _, second = apply(**{done: ""})
        assert len(first.pins) == 1
        # ... and naming them, on any request, releases them.
        client.get("/health", headers={done: first_id})
        assert len(first.pins) == 0
        assert first_id not in serve._PENDING_EXPORTS

        # A client that does not acknowledge has its earlier exports released
        # by its next request, but not those of clients that do.
        _, legacy = apply()
        apply()
        assert len(legacy.pins) == 0
        assert len(second.pins) == 1

        # Exports nobody names are released once kept too long.
        monkeypatch.setattr(serve, "EXPORTS_TIMEOUT_S", 0.0)
        apply(**{done: ""})
        assert len(second.pins) == 0


def test_server_releases_exports_of_a_response_that_fails_to_encode(
    mocked_cuda, tmp_path, monkeypatch
):
    """Exports made before an encoding error are released, not leaked.

    Encoding fails after one array was exported. No client will read that
    response, so its export group must go back to the transport.
    """
    import collections

    from fastapi.testclient import TestClient

    from tesseract_core.runtime import serve
    from tesseract_core.runtime.config import override_config, update_config
    from tesseract_core.runtime.core import load_module_from_path
    from tesseract_core.runtime.device_transport import get_transport

    def export_then_fail(*args, device_exports=None, **kwargs):
        cuda_ipc.dump_cuda_ipc_arraydict(FakeCudaArray((4,), "<f4"), device_exports)
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(serve, "_PENDING_EXPORTS", collections.OrderedDict())
    monkeypatch.setattr(serve, "output_to_bytes", export_then_fail)
    api_path = tmp_path / "tesseract_api.py"
    api_path.write_text(_GPU_OUTPUT_API)

    with override_config():
        update_config(gpu_transport="cuda_ipc")
        transport = get_transport("cuda_ipc")
        released = []
        release = transport.release

        def record_release(session=None):
            released.append(None if session is None else len(session.pins))
            release(session)

        monkeypatch.setattr(transport, "release", record_release)
        client = TestClient(
            serve.create_rest_api(load_module_from_path(api_path)),
            raise_server_exceptions=False,
        )
        response = client.post(
            "/apply",
            json={"inputs": {"n": 4}},
            headers={
                "Accept": "application/json; gpu_transport=cuda_ipc",
                serve.EXPORTS_DONE_HEADER: "",
            },
        )

    assert response.status_code == 500
    assert released == [1], "the failed response's exports were not released"
    assert not serve._PENDING_EXPORTS


def test_concurrent_decodes_of_one_allocation_share_its_mapping(mocked_cuda):
    """CUDA maps an allocation into a process only once.

    Two arrays from one allocation, decoded at the same time, must share a
    single mapping, which closes when the last of them is done. Opening it
    twice failed with "resource already mapped".
    """
    handle = b"\x01" * 64
    with cuda_ipc._mapped(handle, 0) as first:
        with cuda_ipc._mapped(handle, 0) as second:
            assert first == second
        assert mocked_cuda.calls["close"] == []
    assert mocked_cuda.calls["open"] == [(handle, 0)]
    assert mocked_cuda.calls["close"] == [first]
    assert not cuda_ipc._MAPPINGS

    # A later decode maps it afresh.
    with cuda_ipc._mapped(handle, 0):
        pass
    assert len(mocked_cuda.calls["open"]) == 2
