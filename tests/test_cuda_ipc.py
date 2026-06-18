#!/usr/bin/env python3
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for CUDA IPC array encoding.

Run on a GPU machine with:
    python tests/test_cuda_ipc.py

Requires: cupy (for encode/decode) and optionally torch (for interop tests).
The CUDA IPC implementation is framework-agnostic — it works with any
object that implements __cuda_array_interface__ (CuPy, PyTorch, JAX, Numba).
"""

import multiprocessing
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import cupy

    _CUDA_AVAILABLE = cupy.cuda.runtime.getDeviceCount() > 0
except (ImportError, Exception):
    _CUDA_AVAILABLE = False

try:
    import torch as _torch

    _TORCH_AVAILABLE = _torch.cuda.is_available()
except ImportError:
    _TORCH_AVAILABLE = False

requires_cuda = pytest.mark.skipif(
    not _CUDA_AVAILABLE, reason="CUDA + CuPy not available"
)
requires_torch_cuda = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="PyTorch CUDA not available"
)


def check_cuda_available():
    """For standalone script mode."""
    if not _CUDA_AVAILABLE:
        print("SKIP: CUDA + CuPy not available")
        sys.exit(0)
    print(
        f"CUDA available: device 0 = {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
    )


# ── Test 1: Low-level encode/decode round-trip (CuPy) ───────────────────


@requires_cuda
def test_roundtrip_same_process():
    """Encode a CuPy CUDA array, decode it, verify data matches."""
    from tesseract_core.runtime.array_encoding import (
        _dump_cuda_ipc_arraydict,
        _load_cuda_ipc_arraydict,
    )

    print("\n=== Test 1: Same-process encode/decode round-trip ===")

    original = cupy.random.randn(64, 128, dtype=cupy.float32)
    encoded = _dump_cuda_ipc_arraydict(original)

    # Verify the encoded dict structure
    assert encoded["object_type"] == "array"
    assert encoded["shape"] == [64, 128]
    assert encoded["dtype"] == "float32"
    assert encoded["data"]["encoding"] == "cuda_ipc"
    assert "handle" in encoded["data"]
    assert "device" in encoded["data"]
    assert "storage_size" in encoded["data"]
    print(
        f"  Encoded: shape={encoded['shape']}, dtype={encoded['dtype']}, "
        f"device={encoded['data']['device']}, "
        f"handle_len={len(encoded['data']['handle'])}"
    )

    # Decode — returns a CuPy array
    decoded = _load_cuda_ipc_arraydict(encoded)
    assert isinstance(decoded, cupy.ndarray)
    assert decoded.shape == (64, 128)
    assert decoded.dtype == cupy.float32

    # Verify data matches
    cupy.testing.assert_array_equal(original, decoded)
    print("  PASSED: Data matches after same-process round-trip")


# ── Test 2: Cross-process IPC ───────────────────────────────────────────


def _producer(queue, shape, dtype_name):
    """Producer: create a CuPy array and send its IPC-encoded dict."""
    import cupy

    from tesseract_core.runtime.array_encoding import _dump_cuda_ipc_arraydict

    dtype = np.dtype(dtype_name)
    arr = cupy.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    encoded = _dump_cuda_ipc_arraydict(arr)

    # Send encoded dict (small JSON-safe dict) over the queue
    queue.put(encoded)
    # Also send the expected values for verification
    queue.put(cupy.asnumpy(arr).tolist())

    # Wait for consumer to signal it's done reading
    queue.get()  # blocks until consumer is done


def _consumer(queue):
    """Consumer: receive the IPC-encoded dict and reconstruct the array."""
    import cupy

    from tesseract_core.runtime.array_encoding import _load_cuda_ipc_arraydict

    encoded = queue.get()
    expected_values = queue.get()

    decoded = _load_cuda_ipc_arraydict(encoded)
    assert isinstance(decoded, cupy.ndarray)

    actual_values = cupy.asnumpy(decoded).tolist()
    assert actual_values == expected_values, (
        f"Data mismatch!\n  Expected: {expected_values[:5]}...\n  Got: {actual_values[:5]}..."
    )

    # Signal producer we're done
    queue.put("done")
    return True


@requires_cuda
def test_cross_process_ipc():
    """Test CUDA IPC handle transfer between two processes."""
    print("\n=== Test 2: Cross-process CUDA IPC ===")

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()

    shape = (4, 8)

    producer = ctx.Process(target=_producer, args=(queue, shape, "float32"))
    consumer = ctx.Process(target=_consumer, args=(queue,))

    producer.start()
    consumer.start()

    consumer.join(timeout=30)
    producer.join(timeout=30)

    if consumer.exitcode != 0:
        print("  FAILED: Consumer process exited with error")
        sys.exit(1)
    if producer.exitcode != 0:
        print("  FAILED: Producer process exited with error")
        sys.exit(1)

    print("  PASSED: Cross-process CUDA IPC round-trip successful")


# ── Test 3: SDK client-side encode/decode ───────────────────────────────


@requires_cuda
def test_sdk_encode_decode():
    """Test the SDK-level _encode_array_cuda_ipc / _decode_array functions."""
    from tesseract_core.sdk.tesseract import _decode_array, _encode_array_cuda_ipc

    print("\n=== Test 3: SDK client-side encode/decode ===")

    original = cupy.random.randn(32, 64).astype(cupy.float64)
    encoded = _encode_array_cuda_ipc(original)

    assert encoded["data"]["encoding"] == "cuda_ipc"
    print(f"  Encoded via SDK: shape={encoded['shape']}, dtype={encoded['dtype']}")

    decoded = _decode_array(encoded)
    assert isinstance(decoded, cupy.ndarray)
    assert decoded.shape == (32, 64)
    assert decoded.dtype == cupy.float64
    cupy.testing.assert_array_equal(original, decoded)
    print("  PASSED: SDK encode/decode round-trip matches")


# ── Test 4: Multiple dtypes ─────────────────────────────────────────────


@requires_cuda
def test_multiple_dtypes():
    """Test CUDA IPC with various dtypes."""
    from tesseract_core.runtime.array_encoding import (
        _dump_cuda_ipc_arraydict,
        _load_cuda_ipc_arraydict,
    )

    print("\n=== Test 4: Multiple dtype support ===")

    test_cases = [
        ("float16", (10, 20)),
        ("float32", (5, 5, 5)),
        ("float64", (100,)),
        ("int32", (8, 8)),
        ("int64", (3, 3, 3)),
        ("int8", (256,)),
        ("uint8", (16, 16)),
        ("bool", (4, 4)),
    ]

    for dtype_name, shape in test_cases:
        dtype = np.dtype(dtype_name)
        if dtype_name == "bool":
            original = cupy.random.randint(0, 2, shape).astype(dtype)
        elif dtype.kind == "i" or dtype.kind == "u":
            original = cupy.random.randint(0, 100, shape).astype(dtype)
        else:
            original = cupy.random.randn(*shape).astype(dtype)

        encoded = _dump_cuda_ipc_arraydict(original)
        decoded = _load_cuda_ipc_arraydict(encoded)

        cupy.testing.assert_array_equal(original, decoded)
        print(f"  {dtype_name:15s} shape={shape!s:15s} OK")

    print("  PASSED: All dtypes round-trip correctly")


# ── Test 5: Large tensor (benchmark) ────────────────────────────────────


@requires_cuda
def test_large_tensor_benchmark():
    """Benchmark CUDA IPC vs base64 for a large array."""
    import time

    from tesseract_core.runtime.array_encoding import (
        _dump_cuda_ipc_arraydict,
        _load_cuda_ipc_arraydict,
    )
    from tesseract_core.sdk.tesseract import _decode_array, _encode_array

    print("\n=== Test 5: Large tensor benchmark ===")

    # 256 MB array
    size = 64 * 1024 * 1024  # 64M float32 = 256 MB
    original_gpu = cupy.random.randn(size, dtype=cupy.float32)
    original_cpu = cupy.asnumpy(original_gpu)

    # Benchmark base64
    t0 = time.perf_counter()
    encoded_b64 = _encode_array(original_cpu, b64=True)
    t_encode_b64 = time.perf_counter() - t0

    t0 = time.perf_counter()
    _decode_array(encoded_b64)
    t_decode_b64 = time.perf_counter() - t0

    # Benchmark CUDA IPC
    t0 = time.perf_counter()
    encoded_ipc = _dump_cuda_ipc_arraydict(original_gpu)
    t_encode_ipc = time.perf_counter() - t0

    t0 = time.perf_counter()
    decoded_ipc = _load_cuda_ipc_arraydict(encoded_ipc)
    t_decode_ipc = time.perf_counter() - t0

    mb = size * 4 / (1024 * 1024)
    print(f"  Tensor size: {mb:.0f} MB")
    print(
        f"  base64 encode: {t_encode_b64 * 1000:8.2f} ms  "
        f"decode: {t_decode_b64 * 1000:8.2f} ms  "
        f"total: {(t_encode_b64 + t_decode_b64) * 1000:8.2f} ms"
    )
    print(
        f"  cuda_ipc encode: {t_encode_ipc * 1000:8.2f} ms  "
        f"decode: {t_decode_ipc * 1000:8.2f} ms  "
        f"total: {(t_encode_ipc + t_decode_ipc) * 1000:8.2f} ms"
    )
    speedup = (t_encode_b64 + t_decode_b64) / max(t_encode_ipc + t_decode_ipc, 1e-9)
    print(f"  Speedup: {speedup:.1f}x")

    # Verify correctness
    decoded_ipc_cpu = cupy.asnumpy(decoded_ipc)
    np.testing.assert_array_equal(original_cpu, decoded_ipc_cpu)
    print("  PASSED: Large tensor data verified correct")


# ── Test 6: PyTorch interop ─────────────────────────────────────────────


@requires_torch_cuda
def test_torch_encode_cupy_decode():
    """Encode a PyTorch CUDA tensor, decode as CuPy array."""
    import torch

    from tesseract_core.runtime.array_encoding import (
        _dump_cuda_ipc_arraydict,
        _load_cuda_ipc_arraydict,
    )

    print("\n=== Test 6: PyTorch encode → CuPy decode ===")

    original = torch.randn(32, 64, device="cuda:0", dtype=torch.float32)
    encoded = _dump_cuda_ipc_arraydict(original)

    # Decode returns CuPy array
    decoded = _load_cuda_ipc_arraydict(encoded)
    assert hasattr(decoded, "__cuda_array_interface__")

    # Compare via numpy
    expected = original.cpu().numpy()
    actual = (
        cupy.asnumpy(decoded)
        if isinstance(decoded, cupy.ndarray)
        else np.asarray(decoded)
    )
    np.testing.assert_array_equal(expected, actual)
    print("  PASSED: PyTorch → CuPy round-trip matches")


@requires_torch_cuda
def test_cupy_encode_torch_consume():
    """Encode a CuPy array, consume the decoded result in PyTorch via as_tensor."""
    import torch

    from tesseract_core.runtime.array_encoding import (
        _dump_cuda_ipc_arraydict,
        _load_cuda_ipc_arraydict,
    )

    print(
        "\n=== Test 7: CuPy encode → PyTorch consume via __cuda_array_interface__ ==="
    )

    original = cupy.random.randn(16, 32).astype(cupy.float32)
    encoded = _dump_cuda_ipc_arraydict(original)

    decoded_cupy = _load_cuda_ipc_arraydict(encoded)

    # Convert CuPy → PyTorch via DLPack (zero-copy)
    decoded_torch = torch.as_tensor(decoded_cupy, device="cuda:0")
    assert decoded_torch.is_cuda
    assert decoded_torch.shape == (16, 32)

    expected = cupy.asnumpy(original)
    actual = decoded_torch.cpu().numpy()
    np.testing.assert_array_equal(expected, actual)
    print("  PASSED: CuPy → PyTorch consume via as_tensor works")


# ── Test 8: Full Tesseract API with cuda_ipc output format ──────────────


@requires_cuda
def test_tesseract_api_cuda_ipc():
    """Test a real Tesseract with json+cuda_ipc output format via LocalClient."""
    print("\n=== Test 8: Tesseract API with json+cuda_ipc format ===")

    # Create a minimal tesseract_api module in a temp directory
    api_code = """
import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array

class InputSchema(BaseModel):
    x: Array[(None,), "float32"]

class OutputSchema(BaseModel):
    y: Array[(None,), "float32"]

def apply(inputs: InputSchema) -> OutputSchema:
    x_np = np.asarray(inputs.x)
    y_np = x_np * 2.0 + 1.0
    return {"y": y_np}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        api_path = Path(tmpdir) / "tesseract_api.py"
        api_path.write_text(api_code)

        from tesseract_core.sdk.tesseract import Tesseract

        t = Tesseract.from_tesseract_api(
            api_path,
            output_format="json+cuda_ipc",
        )

        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = t.apply({"x": x})

        y = np.asarray(result["y"])
        expected = x * 2.0 + 1.0
        np.testing.assert_allclose(y, expected, rtol=1e-6)
        print(f"  Input:    {x}")
        print(f"  Output:   {y}")
        print(f"  Expected: {expected}")
        print("  PASSED: Tesseract API produces correct results")


# ── Main ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    check_cuda_available()

    test_roundtrip_same_process()
    test_cross_process_ipc()
    test_sdk_encode_decode()
    test_multiple_dtypes()
    test_large_tensor_benchmark()
    if _TORCH_AVAILABLE:
        test_torch_encode_cupy_decode()
        test_cupy_encode_torch_consume()
    test_tesseract_api_cuda_ipc()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
