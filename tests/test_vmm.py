# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests for the copy-free ``cuda_vmm`` GPU transport.

Run on a GPU machine::

    pytest tests/test_vmm.py -m gpu

The VMM transport exports memory *by reference* over a POSIX fd (passed via
SCM_RIGHTS), so it needs two separate processes -- a producer that allocates
VMM-backed memory and exports it via the ``cuda_vmm`` transport, and a consumer
that decodes it via the same transport.

The producer allocates with JAX, whose XLA GPU allocator uses CUDA's Virtual
Memory Management API (``cuMemCreate``) -- the allocator class the transport
requires, and the only one available here that ``cuMemRetainAllocationHandle``
(and thus the fd export) accepts. JAX arrays do not expose
``__cuda_array_interface__``, so the buffer is adopted zero-copy by CuPy via
DLPack (``cupy.from_dlpack``), which exposes the interface the encoder reads --
the same bridge the ``_gpu_vmm`` example uses. The test skips without JAX +
CuPy + CUDA.
"""

from __future__ import annotations

import multiprocessing
import os
import queue as queue_mod
import tempfile
import traceback

import numpy as np
import pytest


def _jax_cupy_cuda_available() -> bool:
    try:
        import cupy
        import jax

        return cupy.cuda.runtime.getDeviceCount() > 0 and any(
            d.platform == "gpu" for d in jax.devices()
        )
    except Exception:
        return False


pytestmark = pytest.mark.gpu

requires_jax_cupy_cuda = pytest.mark.skipif(
    not _jax_cupy_cuda_available(), reason="requires JAX + CuPy with CUDA"
)

_TIMEOUT = 120


def _producer_main(rendezvous, ready, done, q):
    try:
        import cupy
        import jax
        import jax.numpy as jnp
        import orjson

        from tesseract_core.runtime.device_transport import get_transport

        cupy.cuda.Device(0).use()
        transport = get_transport("cuda_vmm")
        # JAX gives a VMM-backed (cuMemCreate) allocation; CuPy adopts the same
        # buffer zero-copy via DLPack so it exposes __cuda_array_interface__ for
        # the encoder. Same device pointer, no copy.
        jarr = jnp.arange(4096, dtype=jnp.float32) % 997.0
        jax.block_until_ready(jarr)
        arr = cupy.from_dlpack(jarr)
        with transport.session("producer") as server:
            ad = transport.descriptor(transport.register(arr, server))
            with open(rendezvous, "wb") as f:
                f.write(orjson.dumps(ad))
            open(ready, "w").close()
            while not os.path.exists(done):
                pass
        q.put(("PRODUCER_OK", ad["data"]["buffer"][:4], ad["data"]["encoding"]))
    except Exception:
        q.put(("PRODUCER_ERROR", traceback.format_exc(), None))


def _consumer_main(rendezvous, ready, done, q):
    try:
        import orjson

        from tesseract_core.runtime.cuda.ipc import IpcDeviceArray
        from tesseract_core.runtime.device_transport import get_transport

        while not os.path.exists(ready):
            pass
        with open(rendezvous, "rb") as f:
            ad = orjson.loads(f.read())
        out = get_transport("cuda_vmm").receive(ad)
        host = np.asarray(out)
        open(done, "w").close()
        q.put(
            (
                "CONSUMER_OK",
                {
                    "is_ipc": isinstance(out, IpcDeviceArray),
                    "values": host,
                },
                None,
            )
        )
    except Exception:
        q.put(("CONSUMER_ERROR", traceback.format_exc(), None))


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
                    status, payload, extra = q.get(timeout=_TIMEOUT)
                except queue_mod.Empty:
                    raise AssertionError("vmm cross-process test timed out") from None
                if status.endswith("ERROR"):
                    raise AssertionError(f"{status}:\n{payload}")
                results[status] = (payload, extra)
        finally:
            consumer.join(timeout=_TIMEOUT)
            producer.join(timeout=_TIMEOUT)
            for proc in (consumer, producer):
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
        return results


@requires_jax_cupy_cuda
def test_vmm_export_round_trip():
    """VMM-backed memory takes the fd-passing path and round-trips correctly."""
    results = _run()

    # The producer's wire descriptor took the VMM form under the cuda_vmm encoding.
    prefix, encoding = results["PRODUCER_OK"]
    assert prefix == "vmm:"
    assert encoding == "cuda_vmm"

    consumer, _ = results["CONSUMER_OK"]
    assert consumer["is_ipc"]
    expected = np.arange(4096, dtype=np.float32) % 997.0
    np.testing.assert_array_equal(consumer["values"], expected)
