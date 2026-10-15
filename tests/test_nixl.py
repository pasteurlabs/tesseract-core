# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests for the nixl device transport: real cross-process NIXL transfers.

Run on a GPU machine with NIXL installed::

    pytest tests/test_nixl.py -m gpu

NIXL moves a GPU array between two *separate* processes (a process cannot READ
from a NIXL agent it registered itself in the same process), so these use either
a spawned producer/consumer pair sharing a rendezvous directory, mirroring the
cross-process cuda_ipc tests, or a Tesseract served in a subprocess. Marked
``gpu``, and skipped where CuPy or NIXL are unavailable.
"""

from __future__ import annotations

import multiprocessing
import os
import queue as queue_mod
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

try:
    import cupy  # noqa: F401

    _CUPY = True
except Exception:  # noqa: BLE001 -- capability probe; failure modes vary by host
    _CUPY = False

try:
    import nixl  # noqa: F401

    _NIXL = True
except Exception:  # noqa: BLE001 -- capability probe; failure modes vary by host
    _NIXL = False

pytestmark = pytest.mark.gpu

requires_nixl = pytest.mark.skipif(
    not (_CUPY and _NIXL), reason="requires CuPy and NIXL"
)

_TIMEOUT = 120


def _producer_main(rendezvous, ready_path, done_path, result_q):
    try:
        import cupy as cp
        import orjson

        from tesseract_core.runtime import nixl_transport as N

        cp.cuda.Device(0).use()
        arr = (cp.arange(4096, dtype=cp.float32) % 997.0).reshape(64, 64)
        group = N.NixlExportGroup()
        try:
            arraydict = N.dump_nixl_arraydict(arr, group)
            with open(rendezvous, "wb") as f:
                f.write(orjson.dumps(arraydict))
        finally:
            # Let the consumer stop waiting even if the export failed.
            open(ready_path, "w").close()
        # Keep the export alive until the consumer has read it.
        while not os.path.exists(done_path):
            pass
        group.release()
        result_q.put(("PRODUCER_OK", None))
    except Exception:  # noqa: BLE001 -- report any worker failure to the parent
        result_q.put(("PRODUCER_ERROR", traceback.format_exc()))


def _consumer_main(rendezvous, ready_path, done_path, result_q):
    try:
        import orjson

        from tesseract_core.runtime import nixl_transport as N
        from tesseract_core.runtime.cuda.ipc import IpcDeviceArray

        while not os.path.exists(ready_path):
            pass
        with open(rendezvous, "rb") as f:
            arraydict = orjson.loads(f.read())

        try:
            out = N.load_nixl_arraydict(arraydict)
            host = out.copy_to_host()
        finally:
            # Let the producer stop waiting even if the read failed.
            open(done_path, "w").close()

        result_q.put(
            (
                "CONSUMER_OK",
                {
                    "is_ipc_device_array": isinstance(out, IpcDeviceArray),
                    "has_cai": hasattr(out, "__cuda_array_interface__"),
                    "values": host,
                },
            )
        )
    except Exception:  # noqa: BLE001 -- report any worker failure to the parent
        result_q.put(("CONSUMER_ERROR", traceback.format_exc()))


def _run_cross_process():
    ctx = multiprocessing.get_context("spawn")
    result_q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        rendezvous = os.path.join(d, "arraydict.json")
        ready_path = os.path.join(d, "ready")
        done_path = os.path.join(d, "done")
        producer = ctx.Process(
            target=_producer_main,
            args=(rendezvous, ready_path, done_path, result_q),
        )
        consumer = ctx.Process(
            target=_consumer_main,
            args=(rendezvous, ready_path, done_path, result_q),
        )
        producer.start()
        consumer.start()
        results = {}
        try:
            for _ in range(2):
                try:
                    status, payload = result_q.get(timeout=_TIMEOUT)
                except queue_mod.Empty:
                    raise AssertionError("nixl cross-process test timed out") from None
                if status.endswith("ERROR"):
                    raise AssertionError(f"{status}:\n{payload}")
                results[status] = payload
        finally:
            consumer.join(timeout=_TIMEOUT)
            producer.join(timeout=_TIMEOUT)
            for proc in (consumer, producer):
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
        return results["CONSUMER_OK"]


@requires_nixl
def test_cross_process_nixl_read():
    """A NIXL READ moves a GPU array to a fresh, framework-agnostic wrapper."""
    result = _run_cross_process()

    # The consumer-facing object is the same wrapper cuda_ipc returns.
    assert result["is_ipc_device_array"]
    assert result["has_cai"]

    expected = (np.arange(4096, dtype=np.float32) % 997.0).reshape(64, 64)
    np.testing.assert_array_equal(result["values"], expected)


# Doubles x with CuPy, so both the input and the output stay on the GPU.
_DOUBLE_API_CODE = """
import cupy as cp
from pydantic import BaseModel
from tesseract_core.runtime import Array, Float32

class InputSchema(BaseModel):
    x: Array[(None,), Float32]

class OutputSchema(BaseModel):
    y: Array[(None,), Float32]

def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(y=cp.asarray(inputs.x) * 2.0)
"""


@requires_nixl
def test_resolve_gpu_transport_uses_nixl_when_it_works(free_port, serve_in_subprocess):
    """A client that did not ask for a transport uses nixl once it checked it works.

    Host copies are forbidden throughout (see ``forbid_device_host_copy``), so
    the input and the output both cross over NIXL.
    """
    import warnings

    import cupy

    from tesseract_core.runtime.cuda.ipc import IpcDeviceArray
    from tesseract_core.sdk.tesseract import Tesseract

    with tempfile.TemporaryDirectory() as tmpdir:
        api_path = Path(tmpdir) / "tesseract_api.py"
        api_path.write_text(_DOUBLE_API_CODE)

        server_env = {"TESSERACT_GPU_TRANSPORT": "nixl"}
        with serve_in_subprocess(api_path, free_port, env=server_env) as url:
            remote = Tesseract.from_url(url)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                transport = remote.resolve_gpu_transport()
            assert transport == "nixl"

            # Repeated calls exercise the client re-adding the server's agent,
            # whose metadata changes as it registers each response's exports.
            for n in (4, 1000, 0):
                x = cupy.arange(n, dtype=cupy.float32)
                result = remote.with_encoding(gpu_transport=transport).apply({"x": x})
                assert isinstance(result["y"], IpcDeviceArray)
                np.testing.assert_allclose(
                    result["y"].copy_to_host(), cupy.asnumpy(x) * 2.0, rtol=1e-6
                )


@requires_nixl
def test_concurrent_calls_over_nixl(free_port, serve_in_subprocess):
    """Concurrent calls through one client each get their own exports back."""
    import cupy

    from tesseract_core.sdk.tesseract import Tesseract

    with tempfile.TemporaryDirectory() as tmpdir:
        api_path = Path(tmpdir) / "tesseract_api.py"
        api_path.write_text(_DOUBLE_API_CODE)

        server_env = {"TESSERACT_GPU_TRANSPORT": "nixl"}
        with serve_in_subprocess(api_path, free_port, env=server_env) as url:
            remote = Tesseract.from_url(url).with_encoding(gpu_transport="nixl")

            def call(i):
                cupy.cuda.Device(0).use()
                x = cupy.full(256, i, dtype=cupy.float32)
                y = remote.apply({"x": x})["y"].copy_to_host()
                return i, y

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(call, range(32)))

            for i, y in results:
                np.testing.assert_array_equal(y, np.full(256, 2.0 * i, np.float32))
