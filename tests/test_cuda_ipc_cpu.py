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

import sys
import types
from typing import Any

import numpy as np
import pytest

from tesseract_core.runtime import array_encoding

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
        return b"\x01" * array_encoding._CUDA_IPC_HANDLE_SIZE

    def fake_open(handle_bytes: bytes, device: int) -> int:
        calls["open"].append((handle_bytes, device))
        return 0x2000  # pretend mapped base pointer

    def fake_close(device_ptr: int) -> None:
        calls["close"].append(device_ptr)

    def fake_stage(base_ptr: int, storage_size: int) -> int:
        calls["stage"].append((base_ptr, storage_size))
        return 0x9000  # pretend staging buffer pointer

    monkeypatch.setattr(array_encoding, "_cuda_get_allocation_base", fake_alloc_base)
    monkeypatch.setattr(array_encoding, "_cuda_ipc_get_mem_handle", fake_get_handle)
    monkeypatch.setattr(array_encoding, "_cuda_ipc_open_mem_handle", fake_open)
    monkeypatch.setattr(array_encoding, "_cuda_ipc_close_mem_handle", fake_close)
    monkeypatch.setattr(array_encoding, "_stage_for_legacy_ipc", fake_stage)

    # Staging buffers are freed via cudart.cudaFree in _release_cuda_ipc_exports;
    # stub the cudart accessor so no real driver is touched and frees are logged.
    fake_cudart = types.SimpleNamespace(
        cudaFree=lambda ptr: calls["free"].append(getattr(ptr, "value", ptr))
    )
    monkeypatch.setattr(array_encoding, "_get_cudart", lambda: fake_cudart)

    # Each test starts with empty registries.
    array_encoding._CUDA_IPC_EXPORT_REGISTRY.clear()
    array_encoding._CUDA_IPC_STAGING_BUFFERS.clear()
    yield calls
    array_encoding._CUDA_IPC_EXPORT_REGISTRY.clear()
    array_encoding._CUDA_IPC_STAGING_BUFFERS.clear()


# ── Encode-side orchestration (mocked CUDA) ─────────────────────────────


def test_dump_assembles_payload_and_offset(patched_cuda):
    arr = FakeCudaArray((4, 8), "<f4", data_ptr=0x5000)
    out = array_encoding._dump_cuda_ipc_arraydict(arr)

    assert out["object_type"] == "array"
    assert out["shape"] == [4, 8]
    assert out["dtype"] == "float32"
    data = out["data"]
    assert data["encoding"] == "cuda_ipc"
    # base = 0x5000 - 256; offset = data_ptr - base = 256; size from fake = 4096
    assert data["storage_offset"] == 256
    assert data["storage_size"] == 4096
    # The handle must be taken on the allocation *base*, not the data pointer.
    assert patched_cuda["get_handle"] == [0x5000 - 256]
    # Handle is base64 of the 64 raw bytes.
    import pybase64

    assert (
        len(pybase64.b64decode(data["handle"])) == array_encoding._CUDA_IPC_HANDLE_SIZE
    )


def test_dump_device_detection_cupy(patched_cuda):
    arr = FakeCudaArray((3,), "<f4", device=_CuPyDevice(id=2))
    out = array_encoding._dump_cuda_ipc_arraydict(arr)
    assert out["data"]["device"] == 2


def test_dump_device_detection_torch(patched_cuda):
    arr = FakeCudaArray((3,), "<f4", device=_TorchDevice(index=3))
    out = array_encoding._dump_cuda_ipc_arraydict(arr)
    assert out["data"]["device"] == 3


def test_dump_device_defaults_to_zero(patched_cuda):
    # No .device attribute, and torch tensors with device.index == None.
    assert (
        array_encoding._dump_cuda_ipc_arraydict(FakeCudaArray((3,), "<f4"))["data"][
            "device"
        ]
        == 0
    )
    arr = FakeCudaArray((3,), "<f4", device=_TorchDevice(index=None))
    assert array_encoding._dump_cuda_ipc_arraydict(arr)["data"]["device"] == 0


def test_dump_rejects_non_cuda_array(patched_cuda):
    with pytest.raises(ValueError, match="cuda_ipc encoding requires a CUDA array"):
        array_encoding._dump_cuda_ipc_arraydict(np.zeros((2, 2), dtype=np.float32))


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
        array_encoding._dump_cuda_ipc_arraydict(arr)


def test_dump_accepts_explicit_contiguous_strides(patched_cuda):
    # strides given but equal to the row-major strides -> still contiguous.
    arr = FakeCudaArray((3, 4), "<f4", strides=(16, 4))
    out = array_encoding._dump_cuda_ipc_arraydict(arr)
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
        return b"\x01" * array_encoding._CUDA_IPC_HANDLE_SIZE

    monkeypatch.setattr(array_encoding, "_cuda_ipc_get_mem_handle", get_handle)

    arr = FakeCudaArray((4, 8), "<f4", data_ptr=0x5000)  # nbytes = 4*8*4 = 128
    out = array_encoding._dump_cuda_ipc_arraydict(arr)

    # Staging was invoked on the array's own data pointer and byte count.
    assert patched_cuda["stage"] == [(0x5000, 128)]
    # Payload reflects the staging buffer: offset 0, size == nbytes.
    assert out["data"]["storage_offset"] == 0
    assert out["data"]["storage_size"] == 128
    # Staging pointer registered for cudaFree.
    assert array_encoding._CUDA_IPC_STAGING_BUFFERS == [0x9000]


# ── Export registry / ring-1 lifetime ───────────────────────────────────


def test_export_registry_pins_and_releases(patched_cuda):
    assert array_encoding._CUDA_IPC_EXPORT_REGISTRY == []
    arr = FakeCudaArray((3,), "<f4")
    array_encoding._dump_cuda_ipc_arraydict(arr)
    # The source array is retained so its (would-be) GPU memory stays valid.
    assert arr in array_encoding._CUDA_IPC_EXPORT_REGISTRY
    array_encoding._release_cuda_ipc_exports()
    assert array_encoding._CUDA_IPC_EXPORT_REGISTRY == []


def test_release_frees_staging_buffers(patched_cuda, monkeypatch):
    """Releasing exports cudaFree's every registered staging buffer."""
    array_encoding._pin_cuda_ipc_staging_buffer(0xAAAA)
    array_encoding._pin_cuda_ipc_staging_buffer(0xBBBB)
    array_encoding._release_cuda_ipc_exports()
    assert patched_cuda["free"] == [0xAAAA, 0xBBBB]
    assert array_encoding._CUDA_IPC_STAGING_BUFFERS == []


# ── Decode-side orchestration (mocked CUDA + fake cupy) ──────────────────


def _install_fake_cupy(monkeypatch):
    """Install a minimal fake ``cupy`` module into sys.modules.

    Only the surface used by ``_load_cuda_ipc_arraydict`` is implemented. The
    returned ``ndarray.copy()`` yields a plain numpy array so tests can inspect
    shape/dtype; this exercises orchestration, not real device copies.
    """
    fake = types.ModuleType("cupy")
    fake.cuda = types.ModuleType("cupy.cuda")
    record: dict[str, Any] = {}

    class UnownedMemory:
        def __init__(self, ptr, size, owner=None):
            record["unowned"] = (ptr, size, owner)

    class MemoryPointer:
        def __init__(self, mem, offset):
            record["memptr_offset"] = offset

    class _NDArray:
        def __init__(self, shape, dtype, memptr):
            self._shape = tuple(shape)
            self._dtype = np.dtype(dtype)

        def copy(self):
            record["copied"] = True
            return np.zeros(self._shape, dtype=self._dtype)

    class Device:
        def __init__(self, device):
            record["device_ctx"] = device

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake.cuda.UnownedMemory = UnownedMemory
    fake.cuda.MemoryPointer = MemoryPointer
    fake.ndarray = _NDArray
    fake.cuda.Device = Device
    fake.cuda.runtime = types.SimpleNamespace(
        deviceSynchronize=lambda: record.setdefault("synced", True)
    )

    monkeypatch.setitem(sys.modules, "cupy", fake)
    return record


def test_load_applies_offset_and_closes_handle(patched_cuda, monkeypatch):
    record = _install_fake_cupy(monkeypatch)

    import pybase64

    encoded = {
        "object_type": "array",
        "shape": [4, 8],
        "dtype": "float32",
        "data": {
            "handle": pybase64.b64encode_as_string(
                b"\x02" * array_encoding._CUDA_IPC_HANDLE_SIZE
            ),
            "device": 1,
            "storage_offset": 128,
            "storage_size": 4096,
            "encoding": "cuda_ipc",
        },
    }

    out = array_encoding._load_cuda_ipc_arraydict(encoded)

    # Orchestration assertions (not CUDA correctness):
    assert patched_cuda["open"] == [(b"\x02" * array_encoding._CUDA_IPC_HANDLE_SIZE, 1)]
    # The offset from the payload must be applied to the MemoryPointer.
    assert record["memptr_offset"] == 128
    # A copy into caller-owned memory happened, and the mapping was closed.
    assert record.get("copied") is True
    assert patched_cuda["close"] == [0x2000]
    # Returned array has the requested shape/dtype.
    assert out.shape == (4, 8)
    assert out.dtype == np.float32


def test_load_closes_handle_even_on_copy_failure(patched_cuda, monkeypatch):
    """The IPC mapping is released even if the copy raises (finally block)."""
    _install_fake_cupy(monkeypatch)

    # Make ndarray.copy() blow up.
    def boom(self):
        raise RuntimeError("copy failed")

    sys.modules["cupy"].ndarray.copy = boom

    import pybase64

    encoded = {
        "object_type": "array",
        "shape": [2],
        "dtype": "float32",
        "data": {
            "handle": pybase64.b64encode_as_string(
                b"\x03" * array_encoding._CUDA_IPC_HANDLE_SIZE
            ),
            "device": 0,
            "storage_offset": 0,
            "storage_size": 8,
            "encoding": "cuda_ipc",
        },
    }
    with pytest.raises(RuntimeError, match="copy failed"):
        array_encoding._load_cuda_ipc_arraydict(encoded)
    # Even though the copy failed, the handle was closed.
    assert patched_cuda["close"] == [0x2000]


def test_load_requires_cupy(monkeypatch):
    """A helpful error is raised when CuPy is not importable."""
    # Ensure any import of cupy fails.
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(RuntimeError, match="cuda_ipc decoding requires CuPy"):
        array_encoding._load_cuda_ipc_arraydict(
            {
                "object_type": "array",
                "shape": [1],
                "dtype": "float32",
                "data": {
                    "handle": "AA==",
                    "device": 0,
                    "storage_offset": 0,
                    "storage_size": 4,
                    "encoding": "cuda_ipc",
                },
            }
        )


# ── GPU-array validation (no CUDA calls at all) ─────────────────────────


def test_validate_cuda_array_passthrough():
    """A matching GPU array validates and is returned unchanged (not copied)."""
    arr = FakeCudaArray((4, 8), "<f4")
    assert array_encoding._validate_cuda_array(arr, (None, 8), "float32") is arr
    # Ellipsis shape means "no shape check".
    assert array_encoding._validate_cuda_array(arr, ..., None) is arr


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
        array_encoding._validate_cuda_array(arr, expected_shape, expected_dtype)


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
    """_cuda_array_to_host handles CuPy-like, torch-like, and rejects others."""

    class CupyLike:
        def get(self):
            return np.array([1.0, 2.0])

    class TorchLike:
        def cpu(self):
            return self

        def numpy(self):
            return np.array([3.0, 4.0])

    assert array_encoding._cuda_array_to_host(CupyLike()).tolist() == [1.0, 2.0]
    assert array_encoding._cuda_array_to_host(TorchLike()).tolist() == [3.0, 4.0]
    with pytest.raises(TypeError, match="Cannot copy GPU array"):
        array_encoding._cuda_array_to_host(object())


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
