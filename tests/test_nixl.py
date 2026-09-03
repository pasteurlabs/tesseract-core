# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests for the json+nixl encoding: a real cross-process NIXL transfer.

Run on a GPU machine with NIXL installed::

    pytest tests/test_nixl.py -m gpu

NIXL moves a GPU array between two *separate* processes (a process cannot READ
from a NIXL agent it registered itself in the same process), so these use a
spawned producer/consumer pair sharing a rendezvous directory, mirroring the
cross-process cuda_ipc tests. Marked ``gpu``; skipped where CuPy or NIXL are
unavailable.
"""

from __future__ import annotations

import multiprocessing
import os
import queue as queue_mod
import tempfile
import traceback

import numpy as np
import pytest

try:
    import cupy  # noqa: F401

    _CUPY = True
except Exception:
    _CUPY = False

try:
    import nixl  # noqa: F401

    _NIXL = True
except Exception:
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
        arraydict = N.dump_nixl_arraydict(arr)
        with open(rendezvous, "wb") as f:
            f.write(orjson.dumps(arraydict))
        open(ready_path, "w").close()
        # Keep the export alive until the consumer has read it.
        while not os.path.exists(done_path):
            pass
        N.release_nixl_exports()
        result_q.put(("PRODUCER_OK", None))
    except Exception:
        result_q.put(("PRODUCER_ERROR", traceback.format_exc()))


def _consumer_main(rendezvous, ready_path, done_path, result_q):
    try:
        import orjson

        from tesseract_core.runtime import nixl_transport as N
        from tesseract_core.runtime.cuda_ipc import IpcDeviceArray

        while not os.path.exists(ready_path):
            pass
        with open(rendezvous, "rb") as f:
            arraydict = orjson.loads(f.read())

        out = N.load_nixl_arraydict(arraydict)
        host = np.asarray(out)  # copy_to_host via __array__
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
    except Exception:
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
