#!/usr/bin/env python3
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for CUDA IPC array encoding.

Run on a GPU machine with either::

    python tests/test_cuda_ipc.py         # standalone runner
    pytest tests/test_cuda_ipc.py         # via pytest

Requires: cupy (used here only to *produce* GPU inputs) and optionally torch
(for the DLPack/CUDA-array-interface interop tests). The CUDA IPC implementation
is framework-agnostic on *both* sides: encode works with any object that
implements ``__cuda_array_interface__`` (CuPy, PyTorch, JAX, Numba), and decode
returns a framework-agnostic ``_IpcDeviceArray`` (no CuPy dependency) that
exposes ``__cuda_array_interface__`` and ``__dlpack__`` plus a host-copy helper.

Note on process model
---------------------
CUDA does not allow a process to ``cudaIpcOpenMemHandle`` a handle it exported
itself (it fails with "invalid device context"). IPC is therefore *inherently
cross-process*: every decode test runs the consumer in a separate process. The
producer must also keep the source GPU array alive until the consumer has
opened the handle, otherwise a pooled allocator may recycle and overwrite the
memory -- the harness below enforces that with an explicit handshake.
"""

import multiprocessing
import queue as queue_mod
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pytest

try:
    import cupy

    _CUDA_AVAILABLE = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _CUDA_AVAILABLE = False

try:
    import torch as _torch

    _TORCH_AVAILABLE = _torch.cuda.is_available()
except Exception:
    _TORCH_AVAILABLE = False

try:
    import jax as _jax

    _JAX_AVAILABLE = any(d.platform == "gpu" for d in _jax.devices())
except Exception:
    _JAX_AVAILABLE = False

# Every test in this module drives real CUDA IPC and therefore needs a physical
# GPU + CuPy. The `gpu` marker lets CI select or skip them (e.g. `-m "not gpu"`)
# in addition to the runtime skip guards below.
pytestmark = pytest.mark.gpu

requires_cuda = pytest.mark.skipif(
    not _CUDA_AVAILABLE, reason="CUDA + CuPy not available"
)
requires_torch_cuda = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="PyTorch CUDA not available"
)
requires_jax_cuda = pytest.mark.skipif(
    not _JAX_AVAILABLE, reason="JAX CUDA not available"
)

_TIMEOUT = 60


# ── Cross-process harness ───────────────────────────────────────────────
#
# A tiny producer/consumer harness that:
#   * runs the consumer in a *separate* process (required for CUDA IPC),
#   * keeps the producer's GPU arrays alive until the consumer is done,
#   * never deadlocks -- every blocking get() has a timeout, and failures
#     in either process are surfaced rather than hanging the test.


def _producer_main(build_fn_name, args, to_consumer, from_consumer):
    """Build arrays in this process, send their encodings, keep them alive.

    Mirrors the ring-1 server contract: the exported arrays are pinned by
    ``dump_cuda_ipc_arraydict`` and kept alive here until the consumer signals
    it is done (which, for the real server, is the next request's release).
    """
    try:
        import cupy  # noqa: F401

        from tesseract_core.runtime.cuda_ipc import dump_cuda_ipc_arraydict

        build_fn = _BUILDERS[build_fn_name]
        arrays = build_fn(*args)
        payloads = [
            (dump_cuda_ipc_arraydict(arr), expected) for arr, expected in arrays
        ]

        to_consumer.put(payloads)
        from_consumer.get(timeout=_TIMEOUT)
        del arrays
    except Exception:
        traceback.print_exc()
        try:
            to_consumer.put(("PRODUCER_ERROR", traceback.format_exc()))
        except Exception:
            pass
        sys.exit(2)


def _consumer_main(to_consumer, from_consumer, result_q):
    """Receive encodings, decode them cross-process, verify, report.

    Decoding is framework-agnostic: it returns an ``_IpcDeviceArray`` that
    exposes ``__cuda_array_interface__`` and ``__dlpack__``. We read the values
    back via the wrapper's host-copy helper -- no CuPy needed for the assertions
    -- so this path proves the decoded result is framework-independent.
    """
    try:
        from tesseract_core.runtime.cuda_ipc import (
            _IpcDeviceArray,
            load_cuda_ipc_arraydict,
        )

        payloads = to_consumer.get(timeout=_TIMEOUT)
        if isinstance(payloads, tuple) and payloads[0] == "PRODUCER_ERROR":
            result_q.put(("PRODUCER_ERROR", payloads[1]))
            return

        results = []
        for encoded, expected in payloads:
            # load_cuda_ipc_arraydict copies into caller-owned memory and closes
            # the IPC mapping before returning, so `decoded` no longer aliases
            # the producer's buffer.
            decoded = load_cuda_ipc_arraydict(encoded)
            assert isinstance(decoded, _IpcDeviceArray)
            assert hasattr(decoded, "__cuda_array_interface__")
            assert hasattr(decoded, "__dlpack__")
            # Host-copy helper: read values back without any GPU framework.
            got = decoded.copy_to_host()
            results.append(
                {
                    "shape": tuple(encoded["shape"]),
                    "dtype": encoded["dtype"],
                    "offset": encoded["data"]["storage_offset"],
                    "match": bool(np.array_equal(got, np.asarray(expected))),
                }
            )

        result_q.put(("OK", results))
    except Exception:
        traceback.print_exc()
        result_q.put(("CONSUMER_ERROR", traceback.format_exc()))
    finally:
        try:
            from_consumer.put("done")
        except Exception:
            pass


def run_cross_process(build_fn_name, *args):
    """Run the producer/consumer harness and return the per-array results.

    Raises AssertionError with the remote traceback if either side errors.
    """
    ctx = multiprocessing.get_context("spawn")
    to_consumer = ctx.Queue()
    from_consumer = ctx.Queue()
    result_q = ctx.Queue()

    producer = ctx.Process(
        target=_producer_main,
        args=(build_fn_name, args, to_consumer, from_consumer),
    )
    consumer = ctx.Process(
        target=_consumer_main, args=(to_consumer, from_consumer, result_q)
    )
    producer.start()
    consumer.start()
    try:
        try:
            status, payload = result_q.get(timeout=_TIMEOUT)
        except queue_mod.Empty:
            raise AssertionError(
                "cross-process IPC test timed out (no result from consumer)"
            ) from None
        if status == "PRODUCER_ERROR":
            raise AssertionError(f"producer failed:\n{payload}")
        if status == "CONSUMER_ERROR":
            raise AssertionError(f"consumer failed:\n{payload}")
        return payload
    finally:
        consumer.join(timeout=_TIMEOUT)
        producer.join(timeout=_TIMEOUT)
        for proc in (consumer, producer):
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)


# ── Array builders (run inside the producer process) ────────────────────


def _build_basic():
    arr = cupy.arange(64 * 128, dtype=cupy.float32).reshape(64, 128)
    return [(arr, cupy.asnumpy(arr))]


def _build_dtypes():
    """Multiple dtypes, forced to have non-trivial pool offsets."""
    # A few pad allocations so subsequent arrays sit at nonzero offsets within
    # their pool block.
    pad = [cupy.arange(777, dtype=cupy.float32) for _ in range(3)]  # noqa: F841
    cases = [
        ("float16", (10, 20)),
        ("float32", (5, 5, 5)),
        ("float64", (100,)),
        ("int32", (8, 8)),
        ("int64", (3, 3, 3)),
        ("int8", (256,)),
        ("uint8", (16, 16)),
        ("bool", (4, 4)),
    ]
    out = []
    for dtype_name, shape in cases:
        dtype = np.dtype(dtype_name)
        if dtype_name == "bool":
            arr = cupy.random.randint(0, 2, shape).astype(dtype)
        elif dtype.kind in "iu":
            arr = cupy.random.randint(0, 100, shape).astype(dtype)
        else:
            arr = cupy.random.randn(*shape).astype(dtype)
        out.append((arr, cupy.asnumpy(arr)))
    return out


def _build_offset_view():
    """A sliced view living at a genuine nonzero offset in its allocation."""
    parent = cupy.arange(10_000, dtype=cupy.float32)
    view = parent[1234 : 1234 + 256]
    # Return both so the parent stays alive (keepalive in producer).
    return [(parent, cupy.asnumpy(parent)), (view, cupy.asnumpy(view))]


def _build_torch():
    import torch

    t = torch.arange(32 * 64, device="cuda:0", dtype=torch.float32).reshape(32, 64)
    return [(t, t.cpu().numpy())]


def _build_jax():
    """JAX arrays exercise the VMM staging fallback (see cuda_ipc._stage_for_legacy_ipc).

    JAX/XLA's default GPU allocator uses CUDA's Virtual Memory Management API
    (``cuMemCreate``/``cuMemAddressReserve``), which the legacy
    ``cudaIpcGetMemHandle`` used by the fast path rejects outright. Unlike
    CuPy/PyTorch (whose default pools use plain ``cudaMalloc`` and hit the fast
    path), every JAX array here is expected to take the staging-buffer fallback.
    """
    import jax.numpy as jnp

    arr = jnp.arange(32 * 64, dtype=jnp.float32).reshape(32, 64)
    return [(arr, np.asarray(arr))]


def _build_force_staging():
    """A CuPy array plus a global patch that forces the VMM staging fallback.

    Makes the first ``_cuda_ipc_get_mem_handle`` call (on the array's base
    pointer) raise, so encode falls back to staging; the second call (on the
    staging buffer) uses the real implementation. The patch runs in the producer
    process and persists through the subsequent ``dump_cuda_ipc_arraydict``.
    """
    from tesseract_core.runtime import cuda_ipc

    real = cuda_ipc._cuda_ipc_get_mem_handle
    state = {"rejected": False}

    def flaky(ptr):
        if not state["rejected"]:
            state["rejected"] = True
            raise RuntimeError("cudaIpcGetMemHandle failed: simulated VMM reject")
        return real(ptr)

    cuda_ipc._cuda_ipc_get_mem_handle = flaky

    arr = cupy.arange(1024, dtype=cupy.float32) + 7.0
    return [(arr, cupy.asnumpy(arr))]


_BUILDERS = {
    "basic": _build_basic,
    "dtypes": _build_dtypes,
    "offset_view": _build_offset_view,
    "torch": _build_torch,
    "jax": _build_jax,
    "force_staging": _build_force_staging,
}


# ── Test 1: encode structure (in-process; no IPC open needed) ───────────


@requires_cuda
def test_encode_structure():
    """The encoded dict has the expected structure and metadata."""
    from tesseract_core.runtime.cuda_ipc import dump_cuda_ipc_arraydict

    arr = cupy.random.randn(64, 128, dtype=cupy.float32)
    encoded = dump_cuda_ipc_arraydict(arr)

    assert encoded["object_type"] == "array"
    assert encoded["shape"] == [64, 128]
    assert encoded["dtype"] == "float32"
    data = encoded["data"]
    assert data["encoding"] == "cuda_ipc"
    # 64-byte handle, base64-encoded.
    import pybase64

    from tesseract_core.runtime.cuda_ipc import _CUDA_IPC_HANDLE_SIZE

    assert len(pybase64.b64decode(data["handle"])) == _CUDA_IPC_HANDLE_SIZE
    assert isinstance(data["device"], int)
    assert data["storage_size"] >= arr.nbytes
    assert data["storage_offset"] >= 0
    # offset + array bytes must fit inside the reported allocation.
    assert data["storage_offset"] + arr.nbytes <= data["storage_size"]


@requires_cuda
def test_encode_requires_cuda_array():
    """Encoding a host array raises a clear error."""
    from tesseract_core.runtime.cuda_ipc import dump_cuda_ipc_arraydict

    with pytest.raises(ValueError, match="cuda_ipc encoding requires a CUDA array"):
        dump_cuda_ipc_arraydict(np.zeros((4, 4), dtype=np.float32))


@requires_cuda
def test_encode_rejects_non_contiguous():
    """Non-contiguous (strided/transposed) arrays are rejected, not corrupted.

    cuda_ipc transfers a flat contiguous byte range; a strided source would be
    silently misread, so encoding must refuse it.
    """
    from tesseract_core.runtime.cuda_ipc import dump_cuda_ipc_arraydict

    strided = cupy.arange(100, dtype=cupy.float32)[::2]
    assert strided.__cuda_array_interface__["strides"] is not None
    with pytest.raises(ValueError, match="C-contiguous"):
        dump_cuda_ipc_arraydict(strided)

    transposed = cupy.arange(12, dtype=cupy.float32).reshape(3, 4).T
    with pytest.raises(ValueError, match="C-contiguous"):
        dump_cuda_ipc_arraydict(transposed)

    # A contiguous copy of the same data encodes fine.
    dump_cuda_ipc_arraydict(cupy.ascontiguousarray(strided))


@requires_cuda
def test_same_process_open_is_unsupported():
    """Sanity: CUDA refuses to open an IPC handle in the exporting process.

    This documents *why* every decode test must be cross-process.
    """
    from tesseract_core.runtime.cuda_ipc import (
        dump_cuda_ipc_arraydict,
        load_cuda_ipc_arraydict,
    )

    arr = cupy.arange(16, dtype=cupy.float32)
    encoded = dump_cuda_ipc_arraydict(arr)
    with pytest.raises(RuntimeError, match="cudaIpcOpenMemHandle failed"):
        load_cuda_ipc_arraydict(encoded)


# ── Test 2: cross-process round-trip ────────────────────────────────────


@requires_cuda
def test_cross_process_basic():
    results = run_cross_process("basic")
    assert len(results) == 1
    assert results[0]["match"], results[0]


@requires_cuda
def test_cross_process_dtypes():
    results = run_cross_process("dtypes")
    assert len(results) == 8
    for r in results:
        assert r["match"], f"dtype {r['dtype']} shape {r['shape']} mismatch"


@requires_cuda
def test_cross_process_nonzero_offset():
    """A sliced view (nonzero storage_offset) decodes to the correct data."""
    results = run_cross_process("offset_view")
    # second entry is the sliced view
    view_result = results[1]
    assert view_result["offset"] > 0, "expected a nonzero storage offset"
    assert view_result["match"], view_result


@requires_cuda
@requires_jax_cuda
def test_cross_process_jax_vmm_fallback():
    """JAX arrays hit the VMM staging fallback and still round-trip correctly.

    JAX/XLA's GPU allocator is VMM-backed, so the legacy ``cudaIpcGetMemHandle``
    fast path (which works for CuPy/PyTorch's default cudaMalloc-based pools)
    rejects it; ``dump_cuda_ipc_arraydict`` should transparently fall back to
    staging the array into a fresh ``cudaMalloc`` buffer (see
    cuda_ipc._stage_for_legacy_ipc) and export a handle to that instead.
    """
    results = run_cross_process("jax")
    assert len(results) == 1
    result = results[0]
    assert result["match"], result
    # The fallback stages just the array's own bytes at offset 0, not the
    # (possibly huge) VMM arena the pointer actually lives in.
    assert result["offset"] == 0


@requires_cuda
def test_cross_process_staging_fallback_forced():
    """Force the staging fallback (without JAX) and verify correctness + free.

    Simulates a VMM-backed pointer by making the first ``cudaIpcGetMemHandle``
    call fail, so ``dump_cuda_ipc_arraydict`` stages a CuPy array into a fresh
    ``cudaMalloc`` buffer and exports a handle to that. Exercises the real
    staging cudaMalloc/cudaMemcpy/cudaFree path on GPU.
    """
    results = run_cross_process("force_staging")
    assert len(results) == 1
    assert results[0]["match"], results[0]
    assert results[0]["offset"] == 0


def _ring1_server(req_q, resp_q):
    """Serial server emulating the ring-1 lifetime contract.

    At the START of each request it releases the previous request's exports and
    churns the allocator (to force reuse of any freed block), then produces and
    exports a fresh output. This is exactly what the serve wrapper does.
    """
    try:
        import cupy

        from tesseract_core.runtime.cuda_ipc import (
            dump_cuda_ipc_arraydict,
            release_cuda_ipc_exports,
        )

        while True:
            req = req_q.get(timeout=_TIMEOUT)
            if req is None:
                return
            i = req
            release_cuda_ipc_exports()  # release-at-request-start
            for _ in range(32):
                tmp = cupy.zeros(4096, dtype=cupy.float32)
                del tmp
            out = cupy.arange(1024, dtype=cupy.float32) + (i + 1) * 100.0
            resp_q.put((i, dump_cuda_ipc_arraydict(out)))
    except Exception:
        traceback.print_exc()
        resp_q.put(("SERVER_ERROR", traceback.format_exc()))


def _ring1_client(req_q, resp_q, result_q, n):
    """Serial client: one request at a time; copy each output and keep it.

    The client decodes without CuPy (the decode returns a framework-agnostic
    wrapper); values are read back via its host-copy helper.
    """
    try:
        from tesseract_core.runtime.cuda_ipc import load_cuda_ipc_arraydict

        kept = []
        for i in range(n):
            req_q.put(i)
            msg = resp_q.get(timeout=_TIMEOUT)
            if isinstance(msg, tuple) and msg[0] == "SERVER_ERROR":
                result_q.put(("SERVER_ERROR", msg[1]))
                return
            j, encoded = msg
            owned = load_cuda_ipc_arraydict(encoded)  # copies + closes mapping
            kept.append((j, owned))
        req_q.put(None)

        # Every earlier copy must still be intact, even though the server has
        # since released and reused those buffers. Read back via the wrapper's
        # host-copy helper (no CuPy needed on the consumer side).
        all_ok = all(
            bool(np.allclose(a.copy_to_host(), np.arange(1024) + (j + 1) * 100.0))
            for j, a in kept
        )
        result_q.put(("OK", all_ok))
    except Exception:
        traceback.print_exc()
        result_q.put(("CLIENT_ERROR", traceback.format_exc()))


@requires_cuda
def test_ring1_serial_reuse():
    """Serial requests under the ring-1 contract keep every client copy valid.

    The server releases the previous request's exports at the start of the next
    request and reuses the pool; because the (serial) client has already copied
    each output into its own buffer, none of its copies are corrupted.
    """
    ctx = multiprocessing.get_context("spawn")
    req_q, resp_q, result_q = (ctx.Queue() for _ in range(3))
    server = ctx.Process(target=_ring1_server, args=(req_q, resp_q))
    client = ctx.Process(target=_ring1_client, args=(req_q, resp_q, result_q, 8))
    server.start()
    client.start()
    try:
        status, payload = result_q.get(timeout=_TIMEOUT)
        assert status == "OK", f"{status}:\n{payload}"
        assert payload, "a client-owned copy was corrupted after server reuse"
    finally:
        client.join(timeout=_TIMEOUT)
        server.join(timeout=_TIMEOUT)
        for proc in (client, server):
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)


# ── Test 3: SDK client-side encode path ─────────────────────────────────


@requires_cuda
def test_sdk_encode_structure():
    """The SDK ``_encode_array_cuda_ipc`` yields the same dict shape."""
    from tesseract_core.sdk.tesseract import _encode_array_cuda_ipc

    arr = cupy.random.randn(32, 64).astype(cupy.float64)
    encoded = _encode_array_cuda_ipc(arr)
    assert encoded["data"]["encoding"] == "cuda_ipc"
    assert encoded["shape"] == [32, 64]
    assert encoded["dtype"] == "float64"


# ── Test 4: framework interop (encode a torch tensor; decode CuPy-free) ──


@requires_cuda
@requires_torch_cuda
def test_cross_process_torch_encode():
    results = run_cross_process("torch")
    assert len(results) == 1
    assert results[0]["match"], results[0]


@requires_cuda
@requires_torch_cuda
def test_decode_to_torch_via_dlpack():
    """A decoded array is adopted by torch zero-copy via DLPack -- no CuPy.

    This is the framework-agnostic proof: the decode returns an
    ``_IpcDeviceArray`` (no CuPy involved), ``torch.from_dlpack`` takes ownership
    of its device buffer, and the values match. ``cupy`` is imported in this
    process only to *produce* the input in the subprocess harness, never to
    inspect the decoded result.
    """
    import torch

    from tesseract_core.runtime.cuda_ipc import (
        _IpcDeviceArray,
        load_cuda_ipc_arraydict,
    )

    # Produce in a subprocess, hand the encoded dict back, decode here, then
    # adopt with torch via DLPack.
    ctx = multiprocessing.get_context("spawn")
    to_consumer = ctx.Queue()
    from_consumer = ctx.Queue()

    producer = ctx.Process(
        target=_producer_main,
        args=("basic", (), to_consumer, from_consumer),
    )
    producer.start()
    try:
        payloads = to_consumer.get(timeout=_TIMEOUT)
        encoded, expected = payloads[0]
        decoded = load_cuda_ipc_arraydict(encoded)
        assert isinstance(decoded, _IpcDeviceArray)
        decoded_torch = torch.from_dlpack(decoded)
        assert decoded_torch.is_cuda
        np.testing.assert_array_equal(decoded_torch.cpu().numpy(), np.asarray(expected))
    finally:
        from_consumer.put("done")
        producer.join(timeout=_TIMEOUT)
        if producer.is_alive():
            producer.terminate()


@requires_cuda
@requires_torch_cuda
def test_decode_to_torch_via_cuda_array_interface():
    """The decoded wrapper is also adoptable via ``__cuda_array_interface__``.

    Complements the DLPack path: ``torch.as_tensor`` consumes the wrapper's
    ``__cuda_array_interface__`` (leaving DLPack ownership untouched, so the
    wrapper still frees its own buffer). Again no CuPy touches the result.
    """
    import torch

    from tesseract_core.runtime.cuda_ipc import load_cuda_ipc_arraydict

    ctx = multiprocessing.get_context("spawn")
    to_consumer = ctx.Queue()
    from_consumer = ctx.Queue()

    producer = ctx.Process(
        target=_producer_main,
        args=("basic", (), to_consumer, from_consumer),
    )
    producer.start()
    try:
        payloads = to_consumer.get(timeout=_TIMEOUT)
        encoded, expected = payloads[0]
        decoded = load_cuda_ipc_arraydict(encoded)
        decoded_torch = torch.as_tensor(decoded, device="cuda:0")
        assert decoded_torch.is_cuda
        np.testing.assert_array_equal(decoded_torch.cpu().numpy(), np.asarray(expected))
    finally:
        from_consumer.put("done")
        producer.join(timeout=_TIMEOUT)
        if producer.is_alive():
            producer.terminate()


def _cupy_free_consumer_main(to_consumer, from_consumer, result_q):
    """Decode + adopt via torch with any import of ``cupy`` forced to fail.

    Proves the decode path carries no CuPy dependency: cupy is blocked before
    the decode runs, so if the decode (or torch.from_dlpack of its result) tried
    to import cupy the test would fail. The producer still uses cupy to build the
    input, but in its own separate process.
    """
    import builtins
    import sys

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("cupy is blocked in this process")
        return real_import(name, *args, **kwargs)

    try:
        import torch

        from tesseract_core.runtime.cuda_ipc import (
            _IpcDeviceArray,
            load_cuda_ipc_arraydict,
        )

        payloads = to_consumer.get(timeout=_TIMEOUT)
        encoded, expected = payloads[0]

        # Purge any cupy the test-module import pulled in, THEN block further
        # imports, so re-appearance in sys.modules can only be the decode path.
        for mod in [m for m in sys.modules if m == "cupy" or m.startswith("cupy.")]:
            del sys.modules[mod]
        builtins.__import__ = blocked_import

        decoded = load_cuda_ipc_arraydict(encoded)
        assert isinstance(decoded, _IpcDeviceArray)
        # Host-copy helper path (no framework).
        host = decoded.copy_to_host()
        host_ok = bool(np.array_equal(host, np.asarray(expected)))
        # Torch-via-DLPack path.
        decoded2 = load_cuda_ipc_arraydict(encoded)
        t = torch.from_dlpack(decoded2)
        torch_ok = bool(np.array_equal(t.cpu().numpy(), np.asarray(expected)))
        # The decode path must not have imported cupy.
        cupy_absent = "cupy" not in sys.modules

        result_q.put(("OK", (host_ok, torch_ok, cupy_absent)))
    except Exception:
        traceback.print_exc()
        result_q.put(("CONSUMER_ERROR", traceback.format_exc()))
    finally:
        try:
            from_consumer.put("done")
        except Exception:
            pass


@requires_cuda
@requires_torch_cuda
def test_decode_is_cupy_free():
    """Decode + torch.from_dlpack succeed with ``import cupy`` blocked.

    The consumer process forbids importing cupy entirely; the producer builds
    the input with cupy in its own process. This is the explicit proof that
    decoding has no CuPy runtime dependency.
    """
    ctx = multiprocessing.get_context("spawn")
    to_consumer = ctx.Queue()
    from_consumer = ctx.Queue()
    result_q = ctx.Queue()

    producer = ctx.Process(
        target=_producer_main,
        args=("basic", (), to_consumer, from_consumer),
    )
    consumer = ctx.Process(
        target=_cupy_free_consumer_main, args=(to_consumer, from_consumer, result_q)
    )
    producer.start()
    consumer.start()
    try:
        status, payload = result_q.get(timeout=_TIMEOUT)
        assert status == "OK", f"{status}:\n{payload}"
        host_ok, torch_ok, cupy_absent = payload
        assert host_ok, "host-copy values mismatch"
        assert torch_ok, "torch.from_dlpack values mismatch"
        assert cupy_absent, "the decode path imported cupy"
    finally:
        consumer.join(timeout=_TIMEOUT)
        producer.join(timeout=_TIMEOUT)
        for proc in (consumer, producer):
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)


# ── Test 5: full Tesseract API with json+cuda_ipc output format ─────────


@requires_cuda
def test_tesseract_api_cuda_ipc_local():
    """A Tesseract served with ``json+cuda_ipc`` returns correct results.

    The apply function returns a NumPy (host) array, which the runtime encodes
    via base64 fallback; this checks the format plumbs through end to end
    without breaking non-GPU outputs.
    """
    api_code = """
import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Float32

class InputSchema(BaseModel):
    x: Array[(None,), Float32]

class OutputSchema(BaseModel):
    y: Array[(None,), Float32]

def apply(inputs: InputSchema) -> OutputSchema:
    x_np = np.asarray(inputs.x)
    return OutputSchema(y=x_np * 2.0 + 1.0)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        api_path = Path(tmpdir) / "tesseract_api.py"
        api_path.write_text(api_code)

        from tesseract_core.sdk.tesseract import Tesseract

        # json+cuda_ipc is experimental and off by default; opt in explicitly.
        with Tesseract.from_tesseract_api(
            api_path,
            output_format="json+cuda_ipc",
            runtime_config={"enable_experimental_cuda_ipc": True},
        ) as t:
            x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            result = t.apply({"x": x})
            y = np.asarray(result["y"])
            np.testing.assert_allclose(y, x * 2.0 + 1.0, rtol=1e-6)


# ── Standalone runner ───────────────────────────────────────────────────


def _run_standalone():
    if not _CUDA_AVAILABLE:
        print("SKIP: CUDA + CuPy not available")
        return 0

    name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"CUDA available: device 0 = {name}\n")

    tests = [
        ("encode structure", test_encode_structure),
        ("encode requires cuda array", test_encode_requires_cuda_array),
        ("same-process open unsupported", test_same_process_open_is_unsupported),
        ("cross-process basic", test_cross_process_basic),
        ("cross-process dtypes", test_cross_process_dtypes),
        ("cross-process nonzero offset", test_cross_process_nonzero_offset),
        ("ring-1 serial reuse", test_ring1_serial_reuse),
        ("sdk encode structure", test_sdk_encode_structure),
    ]
    if _TORCH_AVAILABLE:
        tests += [
            ("cross-process torch encode", test_cross_process_torch_encode),
            ("decode to torch via dlpack", test_decode_to_torch_via_dlpack),
            (
                "decode to torch via cuda array interface",
                test_decode_to_torch_via_cuda_array_interface,
            ),
            ("decode is cupy-free", test_decode_is_cupy_free),
        ]
    if _JAX_AVAILABLE:
        tests.append(
            ("cross-process jax vmm fallback", test_cross_process_jax_vmm_fallback)
        )
    tests.append(("tesseract api cuda_ipc", test_tesseract_api_cuda_ipc_local))

    failures = 0
    for label, fn in tests:
        try:
            fn()
            print(f"  PASSED: {label}")
        except Exception as exc:
            failures += 1
            print(f"  FAILED: {label}: {exc}")

    print("\n" + "=" * 60)
    if failures:
        print(f"{failures} TEST(S) FAILED")
    else:
        print("ALL TESTS PASSED")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
