# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the cuda_ipc encoding logic.

These run on ordinary (GPU-less) CI runners. They cover the Python
*orchestration* around CUDA IPC -- payload assembly, base/offset arithmetic,
device-ordinal detection, shape/dtype validation, the export registry, the
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

from tesseract_core.runtime import array_encoding, cuda_ipc
from tesseract_core.runtime.cuda import api as cuda_api
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
    """When the base pointer is rejected, encode stages into a fresh buffer.

    The staged handle uses offset 0 / size == the array's own nbytes, and the
    staging buffer is registered for later free.
    """
    # Reject the base pointer (VMM-backed) but let the staging buffer succeed,
    # matching real behavior where the fresh cudaMalloc buffer is IPC-exportable.
    mocked_cuda.reject_non_staging_ipc = True

    arr = FakeCudaArray((4, 8), "<f4", data_ptr=0x5000)  # nbytes = 4*8*4 = 128
    out = cuda_ipc.dump_cuda_ipc_arraydict(arr)

    # Staging was invoked on the array's own data pointer and byte count.
    assert mocked_cuda.calls["stage"] == [(0x5000, 128)]
    # Payload reflects the staging buffer: offset 0, size == nbytes.
    unpacked = _unpack_cuda_ipc(out["data"])
    assert unpacked["storage_offset"] == 0
    assert unpacked["storage_size"] == 128
    # Staging pointer registered for a later free.
    assert cuda_ipc._CUDA_IPC_STAGING_BUFFERS == [0x9000]


# ── Export registry / ring-1 lifetime ───────────────────────────────────


def test_export_registry_pins_and_releases(mocked_cuda):
    assert cuda_ipc._CUDA_IPC_EXPORT_REGISTRY == []
    arr = FakeCudaArray((3,), "<f4")
    cuda_ipc.dump_cuda_ipc_arraydict(arr)
    # The source array is retained so its (would-be) GPU memory stays valid.
    assert arr in cuda_ipc._CUDA_IPC_EXPORT_REGISTRY
    cuda_ipc.release_pinned_ipc_exports()
    assert cuda_ipc._CUDA_IPC_EXPORT_REGISTRY == []


def test_release_frees_staging_buffers(mocked_cuda):
    """Releasing exports frees every registered staging buffer."""
    cuda_ipc._pin_cuda_ipc_staging_buffer(0xAAAA)
    cuda_ipc._pin_cuda_ipc_staging_buffer(0xBBBB)
    cuda_ipc.release_pinned_ipc_exports()
    assert mocked_cuda.calls["free"] == [0xAAAA, 0xBBBB]
    assert cuda_ipc._CUDA_IPC_STAGING_BUFFERS == []


def test_client_request_releases_input_exports(mocked_cuda):
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


def test_load_frees_owned_buffer_on_del(mocked_cuda):
    """When no DLPack consumer adopts it, the wrapper frees its buffer on GC."""
    out = cuda_ipc.load_cuda_ipc_arraydict(
        _encoded((2,), "float32", device=0, offset=0, storage_size=8)
    )
    assert mocked_cuda.calls["free"] == []
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

    def boom_open(handle_bytes, device):
        raise RuntimeError("cudaIpcOpenMemHandle failed: simulated")

    monkeypatch.setattr(cuda_api, "ipc_open_mem_handle", boom_open)

    with pytest.raises(RuntimeError, match="cudaIpcOpenMemHandle"):
        cuda_ipc.load_cuda_ipc_arraydict(
            _encoded((2,), "float32", device=0, offset=0, storage_size=8)
        )
    # Owned buffer freed; nothing to close since the mapping never opened.
    assert mocked_cuda.calls["free"] == [0xD000]
    assert mocked_cuda.calls["close"] == []


def test_copy_to_host_reads_device_bytes(mocked_cuda):
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


# ── DeviceTransport interface ───────────────────────────────────────────
#
# cuda_ipc is exposed through the shared DeviceTransport interface so further
# transports slot in behind one lookup. These check the registry wiring and that
# the cuda_ipc backend routes to the same functions the direct API uses.


def test_cuda_ipc_registered_as_transport():
    """The cuda_ipc backend is discoverable by name and satisfies the protocol."""
    from tesseract_core.runtime.device_transport import DeviceTransport, get_transport

    transport = get_transport("cuda_ipc")
    assert transport.name == "cuda_ipc"
    assert transport.reach == "same_host"
    assert isinstance(transport, DeviceTransport)


def test_get_transport_rejects_unknown():
    from tesseract_core.runtime.device_transport import get_transport

    with pytest.raises(KeyError, match="No device transport registered"):
        get_transport("does_not_exist")


def test_cuda_ipc_transport_delegates(patched_cuda, monkeypatch):
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
    assert arr in cuda_ipc._CUDA_IPC_EXPORT_REGISTRY
    transport.release()
    assert cuda_ipc._CUDA_IPC_EXPORT_REGISTRY == []
