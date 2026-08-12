# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GPU tests for serving Tesseracts with the CUDA IPC output format.

Unlike ``tests/test_cuda_ipc.py`` (which drives the encode/decode functions
in-process), these tests build a real GPU Tesseract image, serve it in a
container with ``--gpus all`` and ``--ipc=host``, and round-trip device memory
across the process/container boundary via a genuine ``cudaIpcMemHandle_t``.

Requires a physical CUDA GPU, Docker with the NVIDIA container runtime, and
CuPy on the host (to materialise the decoded device array). They are marked
``gpu`` so GPU-less CI runners skip them.
"""

from pathlib import Path

import numpy as np
import pytest
from common import build_tesseract, image_exists

from tesseract_core import Tesseract

pytestmark = pytest.mark.gpu

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"

try:
    import cupy

    _CUDA_AVAILABLE = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _CUDA_AVAILABLE = False

requires_cuda = pytest.mark.skipif(
    not _CUDA_AVAILABLE, reason="CUDA + CuPy not available on host"
)


@pytest.fixture(scope="module")
def gpu_image_name(docker_client, docker_cleanup_module, shared_dummy_image_name):
    """Build the GPU CUDA-IPC example image once for this module."""
    source = EXAMPLES_DIR / "_gpu_cuda_ipc"
    image_tag = build_tesseract(
        docker_client, source, shared_dummy_image_name, tag="sometag"
    )
    assert image_exists(docker_client, image_tag)
    docker_cleanup_module["images"].append(image_tag)
    return image_tag


@requires_cuda
def test_serve_cuda_ipc_roundtrip(gpu_image_name):
    """A GPU Tesseract served with json+cuda_ipc returns correct device memory.

    Exercises the full export path end-to-end: the served container computes on
    the GPU, exports the result as a CUDA IPC handle (rather than copying to
    host), and the host client opens the handle and materialises a CuPy array.
    """
    a = np.arange(8, dtype=np.float32)
    b = np.ones(8, dtype=np.float32)
    s = 3.0
    expected = s * a + b

    with Tesseract.from_image(
        gpu_image_name,
        gpus=["all"],
        output_format="json+cuda_ipc",
        runtime_config={"enable_experimental_cuda_ipc": True},
    ) as t:
        result = t.apply({"a": a, "b": b, "s": s})

    # The cuda_ipc decode path returns a client-owned CuPy array.
    got = result["result"]
    assert cupy.get_array_module(got) is cupy, (
        f"expected a device array from cuda_ipc, got {type(got)}"
    )
    np.testing.assert_allclose(cupy.asnumpy(got), expected, rtol=1e-5, atol=1e-5)


@requires_cuda
def test_serve_cuda_ipc_serial_reuse(gpu_image_name):
    """Serial requests each return correct data despite buffer reuse.

    The server releases the previously exported buffer at the start of each
    request, so back-to-back calls must not corrupt each other's results.
    """
    with Tesseract.from_image(
        gpu_image_name,
        gpus=["all"],
        output_format="json+cuda_ipc",
        runtime_config={"enable_experimental_cuda_ipc": True},
    ) as t:
        for i in range(3):
            a = np.full(4, float(i), dtype=np.float32)
            b = np.zeros(4, dtype=np.float32)
            result = t.apply({"a": a, "b": b, "s": 2.0})
            got = cupy.asnumpy(result["result"])
            np.testing.assert_allclose(got, 2.0 * a, rtol=1e-5, atol=1e-5)
