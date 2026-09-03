# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests for the copy-free VMM path behind ``json+cuda_ipc``.

Run on a GPU machine::

    pytest tests/test_vmm.py -m gpu

The VMM path exports memory *by reference* over a POSIX fd (passed via
SCM_RIGHTS), so it needs two separate processes -- a producer that allocates
VMM-backed memory and encodes via ``dump_cuda_ipc_arraydict`` (which routes to
the VMM path automatically), and a consumer that decodes via
``load_cuda_ipc_arraydict``. VMM-backed memory is produced with PyTorch's
``expandable_segments`` allocator; the test skips without torch+CUDA.
"""

from __future__ import annotations

import multiprocessing
import os
import queue as queue_mod
import tempfile
import traceback

import numpy as np
import pytest


def _torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.gpu

requires_torch_cuda = pytest.mark.skipif(
    not _torch_cuda_available(), reason="requires torch with CUDA"
)

_TIMEOUT = 120


def _producer_main(rendezvous, ready, done, q):
    try:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        import orjson
        import torch

        from tesseract_core.runtime import cuda_ipc

        torch.cuda.set_device(0)
        arr = torch.arange(4096, dtype=torch.float32, device="cuda") % 997.0
        ad = cuda_ipc.dump_cuda_ipc_arraydict(arr)
        with open(rendezvous, "wb") as f:
            f.write(orjson.dumps(ad))
        open(ready, "w").close()
        while not os.path.exists(done):
            pass
        cuda_ipc.release_pinned_ipc_exports()
        q.put(("PRODUCER_OK", ad["data"]["buffer"][:4]))
    except Exception:
        q.put(("PRODUCER_ERROR", traceback.format_exc()))


def _consumer_main(rendezvous, ready, done, q):
    try:
        import orjson

        from tesseract_core.runtime import cuda_ipc
        from tesseract_core.runtime.cuda_ipc import IpcDeviceArray

        while not os.path.exists(ready):
            pass
        with open(rendezvous, "rb") as f:
            ad = orjson.loads(f.read())
        out = cuda_ipc.load_cuda_ipc_arraydict(ad)
        host = np.asarray(out)
        open(done, "w").close()
        q.put(
            (
                "CONSUMER_OK",
                {
                    "is_ipc": isinstance(out, IpcDeviceArray),
                    "values": host,
                },
            )
        )
    except Exception:
        q.put(("CONSUMER_ERROR", traceback.format_exc()))


def _run():
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        rendezvous = os.path.join(d, "ad.json")
        ready = os.path.join(d, "ready")
        done = os.path.join(d, "done")
        producer = ctx.Process(target=_producer_main, args=(rendezvous, ready, done, q))
        consumer = ctx.Process(target=_consumer_main, args=(rendezvous, ready, done, q))
        producer.start()
        consumer.start()
        results = {}
        try:
            for _ in range(2):
                try:
                    status, payload = q.get(timeout=_TIMEOUT)
                except queue_mod.Empty:
                    raise AssertionError("vmm cross-process test timed out") from None
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
        return results


@requires_torch_cuda
def test_vmm_export_routes_through_cuda_ipc():
    """VMM-backed memory takes the fd-passing path and round-trips correctly."""
    results = _run()

    # The producer's wire descriptor took the VMM variant, not the legacy handle.
    assert results["PRODUCER_OK"] == "vmm:"

    consumer = results["CONSUMER_OK"]
    assert consumer["is_ipc"]
    expected = np.arange(4096, dtype=np.float32) % 997.0
    np.testing.assert_array_equal(consumer["values"], expected)
