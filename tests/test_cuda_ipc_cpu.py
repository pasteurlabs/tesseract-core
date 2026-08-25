# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the cuda_ipc encoding logic.

These run on ordinary (GPU-less) CI runners. They cover the Python
*orchestration* around CUDA IPC -- payload assembly, base/offset arithmetic,
device-ordinal detection, shape/dtype validation, the export registry, the
serve-side release hook, the ``--ipc=host`` wiring, and the CLI guard -- by

  * feeding fake objects that expose ``__cuda_array_interface__`` (no device
    memory), and
  * monkeypatching the thin ctypes/CuPy wrappers that actually talk to CUDA.

They deliberately do NOT verify that IPC transfers the correct bytes, that the
by-value handle marshalling is right, or that offsets read the right data: those
are CUDA-runtime properties with no meaning against a mock. Those guarantees are
covered by the GPU tests in ``test_cuda_ipc.py`` (marked ``@pytest.mark.gpu``).
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from tesseract_core.runtime import array_encoding, cuda_ipc


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


@pytest.fixture
def patched_cuda(monkeypatch):
    """Patch the CUDA wrapper functions so encode/decode run without a GPU.

    Yields a record dict capturing the calls made, so tests can assert on the
    orchestration (which pointer the handle was taken on, that close was called,
    etc.) rather than on real CUDA behavior.
    """
    calls: dict[str, list] = {
        "get_handle": [],
        "open": [],
        "close": [],
        "alloc_base": [],
        "stage": [],
        "free": [],
    }

    # Report an allocation whose base sits 256 bytes below the data pointer, so
    # storage_offset arithmetic is exercised with a non-zero value.
    def fake_alloc_base(device_ptr: int) -> tuple[int, int]:
        calls["alloc_base"].append(device_ptr)
        base = device_ptr - 256
        size = 4096
        return base, size

    def fake_get_handle(base_ptr: int) -> bytes:
        calls["get_handle"].append(base_ptr)
        return b"\x01" * cuda_ipc._CUDA_IPC_HANDLE_SIZE

    def fake_open(handle_bytes: bytes, device: int) -> int:
        calls["open"].append((handle_bytes, device))
        return 0x2000  # pretend mapped base pointer

    def fake_close(device_ptr: int) -> None:
        calls["close"].append(device_ptr)

    def fake_stage(base_ptr: int, storage_size: int) -> int:
        calls["stage"].append((base_ptr, storage_size))
        return 0x9000  # pretend staging buffer pointer

    monkeypatch.setattr(cuda_ipc, "_cuda_get_allocation_base", fake_alloc_base)
    monkeypatch.setattr(cuda_ipc, "_cuda_ipc_get_mem_handle", fake_get_handle)
    monkeypatch.setattr(cuda_ipc, "_cuda_ipc_open_mem_handle", fake_open)
    monkeypatch.setattr(cuda_ipc, "_cuda_ipc_close_mem_handle", fake_close)
    monkeypatch.setattr(cuda_ipc, "_stage_for_legacy_ipc", fake_stage)

    # Staging buffers are freed via cudart.cudaFree in release_pinned_ipc_exports;
    # stub the cudart accessor so no real driver is touched and frees are logged.
    fake_cudart = types.SimpleNamespace(
        cudaFree=lambda ptr: calls["free"].append(getattr(ptr, "value", ptr))
    )
    monkeypatch.setattr(cuda_ipc, "_get_cudart", lambda: fake_cudart)

    # Each test starts with empty registries.
    cuda_ipc._CUDA_IPC_EXPORT_REGISTRY.clear()
    cuda_ipc._CUDA_IPC_STAGING_BUFFERS.clear()
    yield calls
    cuda_ipc._CUDA_IPC_EXPORT_REGISTRY.clear()
    cuda_ipc._CUDA_IPC_STAGING_BUFFERS.clear()


# ── Encode-side orchestration (mocked CUDA) ─────────────────────────────


def test_dump_assembles_payload_and_offset(patched_cuda):
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
    assert patched_cuda["get_handle"] == [0x5000 - 256]
    # Handle is base64 of the 64 raw bytes.
    import pybase64

    assert len(pybase64.b64decode(unpacked["handle"])) == cuda_ipc._CUDA_IPC_HANDLE_SIZE


def test_dump_device_detection_cupy(patched_cuda):
    arr = FakeCudaArray((3,), "<f4", device=_CuPyDevice(id=2))
    out = cuda_ipc.dump_cuda_ipc_arraydict(arr)
    assert _unpack_cuda_ipc(out["data"])["device"] == 2


def test_dump_device_detection_torch(patched_cuda):
    arr = FakeCudaArray((3,), "<f4", device=_TorchDevice(index=3))
    out = cuda_ipc.dump_cuda_ipc_arraydict(arr)
    assert _unpack_cuda_ipc(out["data"])["device"] == 3


def test_dump_device_defaults_to_zero(patched_cuda):
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


def test_dump_rejects_non_cuda_array(patched_cuda):
    with pytest.raises(ValueError, match="cuda_ipc encoding requires a CUDA array"):
        cuda_ipc.dump_cuda_ipc_arraydict(np.zeros((2, 2), dtype=np.float32))


@pytest.mark.parametrize(
    "shape, strides",
    [
        ((50,), (8,)),  # every-other-element view of <f4 (itemsize 4)
        ((4, 3), (4, 16)),  # transposed 3x4 float32
    ],
)
def test_dump_rejects_non_contiguous(patched_cuda, shape, strides):
    arr = FakeCudaArray(shape, "<f4", strides=strides)
    with pytest.raises(ValueError, match="C-contiguous"):
        cuda_ipc.dump_cuda_ipc_arraydict(arr)


def test_dump_accepts_explicit_contiguous_strides(patched_cuda):
    # strides given but equal to the row-major strides -> still contiguous.
    arr = FakeCudaArray((3, 4), "<f4", strides=(16, 4))
    out = cuda_ipc.dump_cuda_ipc_arraydict(arr)
    assert out["data"]["encoding"] == "cuda_ipc"


# ── VMM staging fallback (legacy IPC reject) ────────────────────────────


def test_dump_falls_back_to_staging_on_ipc_reject(patched_cuda, monkeypatch):
    """When the base pointer is rejected, encode stages into a fresh buffer.

    The staged handle uses offset 0 / size == the array's own nbytes, and the
    staging buffer is registered for later free.
    """
    # Reject the base pointer (VMM-backed) but let the staging buffer succeed,
    # matching real behavior where the fresh cudaMalloc buffer is IPC-exportable.
    staging_ptr = 0x9000

    def get_handle(ptr):
        if ptr != staging_ptr:
            raise RuntimeError("cudaIpcGetMemHandle failed: simulated VMM reject")
        return b"\x01" * cuda_ipc._CUDA_IPC_HANDLE_SIZE

    monkeypatch.setattr(cuda_ipc, "_cuda_ipc_get_mem_handle", get_handle)

    arr = FakeCudaArray((4, 8), "<f4", data_ptr=0x5000)  # nbytes = 4*8*4 = 128
    out = cuda_ipc.dump_cuda_ipc_arraydict(arr)

    # Staging was invoked on the array's own data pointer and byte count.
    assert patched_cuda["stage"] == [(0x5000, 128)]
    # Payload reflects the staging buffer: offset 0, size == nbytes.
    unpacked = _unpack_cuda_ipc(out["data"])
    assert unpacked["storage_offset"] == 0
    assert unpacked["storage_size"] == 128
    # Staging pointer registered for cudaFree.
    assert cuda_ipc._CUDA_IPC_STAGING_BUFFERS == [0x9000]


# ── Export registry / ring-1 lifetime ───────────────────────────────────


def test_export_registry_pins_and_releases(patched_cuda):
    assert cuda_ipc._CUDA_IPC_EXPORT_REGISTRY == []
    arr = FakeCudaArray((3,), "<f4")
    cuda_ipc.dump_cuda_ipc_arraydict(arr)
    # The source array is retained so its (would-be) GPU memory stays valid.
    assert arr in cuda_ipc._CUDA_IPC_EXPORT_REGISTRY
    cuda_ipc.release_pinned_ipc_exports()
    assert cuda_ipc._CUDA_IPC_EXPORT_REGISTRY == []


def test_release_frees_staging_buffers(patched_cuda, monkeypatch):
    """Releasing exports cudaFree's every registered staging buffer."""
    cuda_ipc._pin_cuda_ipc_staging_buffer(0xAAAA)
    cuda_ipc._pin_cuda_ipc_staging_buffer(0xBBBB)
    cuda_ipc.release_pinned_ipc_exports()
    assert patched_cuda["free"] == [0xAAAA, 0xBBBB]
    assert cuda_ipc._CUDA_IPC_STAGING_BUFFERS == []


def test_client_request_releases_input_exports(patched_cuda):
    """HTTPClient._request must release the GPU inputs it pinned while encoding.

    Regression test: the client shares the process-global export registry with
    the server, and encoding a GPU input pins it there. If _request does not
    release afterward the registry grows without bound across calls (each call's
    inputs leaked). The pin must survive long enough for the server to decode --
    i.e. until the response body is buffered -- so we assert it is still present
    when the (fake) request is dispatched, and gone once _request returns.
    """
    from tesseract_core.sdk.tesseract import HTTPClient

    seen_during_request = {}

    response = Mock(status_code=200, ok=True, content=b"{}")

    class FakeSession:
        def __init__(self) -> None:
            self.headers = {}

        def request(self, **kwargs):
            # The input must still be pinned here: a real server has not yet
            # decoded and copied it out.
            seen_during_request["pinned"] = list(cuda_ipc._CUDA_IPC_EXPORT_REGISTRY)
            return response

    client = HTTPClient.__new__(HTTPClient)
    client._url = "http://localhost:8000"
    client._output_path = None
    client._output_format = "json+cuda_ipc"
    client._timeout = None
    client._session = FakeSession()

    arr = FakeCudaArray((3,), "<f4")
    assert cuda_ipc._CUDA_IPC_EXPORT_REGISTRY == []
    client._request("apply", method="POST", payload={"a": arr})

    # Pinned during the request (so the server can copy it out) ...
    assert arr in seen_during_request["pinned"]
    # ... and released once the request returned (no leak across calls).
    assert cuda_ipc._CUDA_IPC_EXPORT_REGISTRY == []


def test_client_request_cpu_only_payload_skips_release(monkeypatch):
    """A cuda_ipc request with no GPU inputs must not touch the release path.

    Nothing gets pinned, so _request must not import/call the cuda_ipc runtime
    for cleanup -- otherwise a base install (no runtime extra) would spuriously
    fail on an all-CPU payload. Guard by making the release helper explode if
    called.
    """
    from tesseract_core.sdk import tesseract as sdk
    from tesseract_core.sdk.tesseract import HTTPClient

    def _boom():
        raise AssertionError("release must not be called for a CPU-only payload")

    monkeypatch.setattr(
        sdk,
        "_import_cuda_ipc",
        lambda: types.SimpleNamespace(release_pinned_ipc_exports=_boom),
    )

    response = Mock(status_code=200, ok=True, content=b"{}")

    class FakeSession:
        def __init__(self) -> None:
            self.headers = {}

        def request(self, **kwargs):
            return response

    client = HTTPClient.__new__(HTTPClient)
    client._url = "http://localhost:8000"
    client._output_path = None
    client._output_format = "json+cuda_ipc"
    client._timeout = None
    client._session = FakeSession()

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
        # dependencies, e.g. fsspec, are absent). Covers both `import
        # tesseract_core.runtime.cuda_ipc` and `from tesseract_core.runtime
        # import cuda_ipc`.
        if name == "tesseract_core.runtime.cuda_ipc" or (
            name == "tesseract_core.runtime" and "cuda_ipc" in (fromlist or ())
        ):
            raise ImportError("No module named 'fsspec'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"tesseract-core\[runtime\]"):
        sdk._import_cuda_ipc()


# ── Decode-side orchestration (mocked CUDA, no CuPy) ─────────────────────
#
# Decoding no longer depends on CuPy: it uses only the ctypes cudart primitives
# (cudaSetDevice/cudaMalloc/cudaMemcpy/cudaDeviceSynchronize/cudaFree) plus the
# IPC open/close helpers. These tests mock those primitives so the *Python
# orchestration* (offset arithmetic, own-nbytes copy, synchronize-before-close,
# mapping close, buffer free, DLPack ownership) is exercised on a GPU-less box;
# the real device-copy correctness lives in the GPU tests in test_cuda_ipc.py.


@pytest.fixture
def patched_decode(monkeypatch):
    """Mock the ctypes cudart decode primitives and the IPC open/close helpers.

    A single fake device buffer is simulated with a Python ``bytearray`` so that
    ``copy_to_host`` returns real bytes. Records the sequence of primitive calls
    for ordering/argument assertions.
    """
    import ctypes

    calls: dict[str, list] = {
        "set_device": [],
        "malloc": [],
        "memcpy": [],
        "sync": [],
        "free": [],
        "open": [],
        "close": [],
    }
    state: dict[str, Any] = {"owned_ptr": 0xD000, "buffer": None}

    class FakeCudart:
        def cudaSetDevice(self, device):
            calls["set_device"].append(device)
            return 0

        def cudaMalloc(self, pptr, size):
            size_v = getattr(size, "value", size)
            calls["malloc"].append(int(size_v))
            state["buffer"] = bytearray(int(size_v))
            # Write the owned pointer into the c_void_p the caller passed by ref.
            ctypes.cast(pptr, ctypes.POINTER(ctypes.c_void_p)).contents.value = state[
                "owned_ptr"
            ]
            return 0

        def cudaMemcpy(self, dst, src, size, kind):
            dst_v = getattr(dst, "value", dst)
            src_v = getattr(src, "value", src)
            size_v = int(getattr(size, "value", size))
            kind_v = int(getattr(kind, "value", kind))
            calls["memcpy"].append((dst_v, src_v, size_v, kind_v))
            # For device->host copies, fill the host buffer with the recorded
            # device bytes so copy_to_host yields deterministic data.
            if kind_v == cuda_ipc._cudaMemcpyDeviceToHost:
                src_bytes = state.get("device_bytes")
                if src_bytes is not None:
                    ctypes.memmove(dst_v, src_bytes, size_v)
            return 0

        def cudaDeviceSynchronize(self):
            calls["sync"].append(True)
            return 0

        def cudaFree(self, ptr):
            calls["free"].append(getattr(ptr, "value", ptr))
            return 0

        def cudaGetErrorString(self, code):
            return b"fake error"

    fake = FakeCudart()
    monkeypatch.setattr(cuda_ipc, "_get_cudart", lambda: fake)

    def fake_open(handle_bytes, device):
        calls["open"].append((handle_bytes, device))
        return 0x2000  # pretend mapped base pointer

    def fake_close(device_ptr):
        calls["close"].append(device_ptr)

    monkeypatch.setattr(cuda_ipc, "_cuda_ipc_open_mem_handle", fake_open)
    monkeypatch.setattr(cuda_ipc, "_cuda_ipc_close_mem_handle", fake_close)
    return calls, state


def _encoded(shape, dtype, device, offset, storage_size, fill=b"\x02"):
    import pybase64

    handle = pybase64.b64encode_as_string(fill * cuda_ipc._CUDA_IPC_HANDLE_SIZE)
    return {
        "object_type": "array",
        "shape": list(shape),
        "dtype": dtype,
        "data": {
            "buffer": f"{device}:{handle}:{offset}:{storage_size}",
            "encoding": "cuda_ipc",
        },
    }


def test_load_copies_own_bytes_at_offset_and_closes(patched_decode):
    """Decode allocates the array's own nbytes, copies from base+offset, closes."""
    calls, _state = patched_decode
    handle = b"\x02" * cuda_ipc._CUDA_IPC_HANDLE_SIZE
    encoded = _encoded((4, 8), "float32", device=1, offset=128, storage_size=4096)

    out = cuda_ipc.load_cuda_ipc_arraydict(encoded)

    nbytes = 4 * 8 * 4  # 128
    # Opened on the requested device with the decoded handle.
    assert calls["open"] == [(handle, 1)]
    # Allocated exactly the array's own byte size (not the whole storage_size).
    assert calls["malloc"] == [nbytes]
    # One device->device copy of nbytes, from base(0x2000)+offset(128) into the
    # owned buffer (0xD000).
    assert len(calls["memcpy"]) == 1
    dst, src, size, kind = calls["memcpy"][0]
    assert dst == 0xD000
    assert src == 0x2000 + 128
    assert size == nbytes
    assert kind == cuda_ipc._cudaMemcpyDeviceToDevice
    # Synchronised before the mapping was closed.
    assert calls["sync"] == [True]
    assert calls["close"] == [0x2000]
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


def test_load_frees_owned_buffer_on_del(patched_decode):
    """When no DLPack consumer adopts it, the wrapper frees its buffer on GC."""
    calls, _state = patched_decode
    out = cuda_ipc.load_cuda_ipc_arraydict(
        _encoded((2,), "float32", device=0, offset=0, storage_size=8)
    )
    assert calls["free"] == []
    del out
    import gc

    gc.collect()
    assert calls["free"] == [0xD000]


def test_load_closes_handle_even_on_copy_failure(patched_decode, monkeypatch):
    """The IPC mapping is released and the owned buffer freed if the copy fails."""
    calls, _state = patched_decode

    fake = cuda_ipc._get_cudart()

    def boom_memcpy(dst, src, size, kind):
        return 999  # non-zero -> error

    monkeypatch.setattr(fake, "cudaMemcpy", boom_memcpy)

    with pytest.raises(RuntimeError, match="cudaMemcpy"):
        cuda_ipc.load_cuda_ipc_arraydict(
            _encoded((2,), "float32", device=0, offset=0, storage_size=8)
        )
    # Owned buffer freed and the IPC mapping closed despite the failure.
    assert calls["free"] == [0xD000]
    assert calls["close"] == [0x2000]


def test_copy_to_host_reads_device_bytes(patched_decode):
    """copy_to_host performs a device->host memcpy + sync and returns the bytes."""
    calls, state = patched_decode
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    state["device_bytes"] = expected.tobytes()

    out = cuda_ipc.load_cuda_ipc_arraydict(
        _encoded((2, 3), "float32", device=0, offset=0, storage_size=24)
    )
    host = out.copy_to_host()
    np.testing.assert_array_equal(host, expected)
    # A device->host copy happened and was synchronised.
    d2h = [c for c in calls["memcpy"] if c[3] == cuda_ipc._cudaMemcpyDeviceToHost]
    assert len(d2h) == 1
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


def test_encode_array_cuda_ipc_requires_cuda_array():
    """cuda_ipc in JSON mode rejects a plain host array."""
    with pytest.raises(ValueError, match="cuda_ipc encoding requires a CUDA array"):
        array_encoding.encode_array(
            np.arange(3),
            _info(True, {"array_encoding": "cuda_ipc"}),
            (None,),
            "int64",
        )


def test_cuda_array_to_host_branches():
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


# ── experimental feature flag gating ────────────────────────────────────


def test_output_to_bytes_rejects_cuda_ipc_by_default():
    """Without the experimental flag, json+cuda_ipc is not an accepted format."""
    from tesseract_core.runtime import config, file_interactions

    config.update_config(enable_experimental_cuda_ipc=False)
    with pytest.raises(ValueError, match=r"Unsupported format json\+cuda_ipc"):
        file_interactions.output_to_bytes({"y": 1}, "json+cuda_ipc")


def test_available_formats_reflects_flag():
    from tesseract_core.runtime import config
    from tesseract_core.runtime.file_interactions import available_formats

    config.update_config(enable_experimental_cuda_ipc=False)
    assert "json+cuda_ipc" not in available_formats()

    config.update_config(enable_experimental_cuda_ipc=True)
    assert "json+cuda_ipc" in available_formats()


# ── format -> encoding-context mapping ──────────────────────────────────


def test_output_to_bytes_cuda_ipc_context(monkeypatch):
    """json+cuda_ipc maps to the cuda_ipc array-encoding context (flag enabled)."""
    from tesseract_core.runtime import config, file_interactions

    config.update_config(enable_experimental_cuda_ipc=True)
    captured = {}

    class FakeAdapter:
        def __init__(self, _type):
            pass

        def dump_python(self, obj, mode, context, exclude_unset):
            captured["context"] = context
            return {}

    monkeypatch.setattr(file_interactions, "TypeAdapter", FakeAdapter)
    monkeypatch.setattr(file_interactions.orjson, "dumps", lambda d: b"{}")

    file_interactions.output_to_bytes({"y": 1}, "json+cuda_ipc")
    assert captured["context"] == {"array_encoding": "cuda_ipc"}
