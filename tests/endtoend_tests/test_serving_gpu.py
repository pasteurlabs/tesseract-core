# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GPU tests for serving Tesseracts with the CUDA IPC output format.

Unlike ``tests/test_cuda_ipc.py`` (which drives the encode/decode functions
in-process), these tests build a real GPU Tesseract image, serve it in a
container with ``--gpus all`` and ``--ipc=host``, and round-trip device memory
across the process/container boundary via a genuine ``cudaIpcMemHandle_t``. One
image is built per GPU array framework (CuPy, JAX, PyTorch) so the export path
is covered against both metadata sources it reads: ``__cuda_array_interface__``
(CuPy, PyTorch) and DLPack (JAX).

Requires a physical CUDA GPU and Docker with the NVIDIA container runtime. CuPy
is used only as a convenient GPU-availability probe on the host; the decoded
result is inspected framework-agnostically (via its host-copy helper), so the
host does not need CuPy to read cuda_ipc outputs. They are marked ``gpu`` so
GPU-less CI runners skip them.
"""

import os
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


# The framework runs inside the container, so the host stays framework-agnostic
# and needs only a GPU to decode the handle (hence the plain requires_cuda skip).
GPU_EXAMPLES = ["_gpu_cupy", "_gpu_jax", "_gpu_torch"]

# Build the container against the same CUDA major the GPU CI matrix leg uses, so
# the runtime's in-container libcudart matches the host frameworks it exchanges
# handles with. The example ships CUDA 12 by default (so it builds standalone)
# plus a tesseract_requirements_cuda13.txt; the 13.x leg selects the latter and a
# 13.x base image. CI sets both env vars per matrix leg and pre-builds with the
# same overrides, so the fixture builds hit the warm layer cache. A local GPU run
# with neither set falls back to the CUDA 12 default the examples already carry.
CUDA_MAJOR = os.environ.get("TESSERACT_TEST_CUDA_MAJOR", "12")
CUDA_BASE_IMAGE = os.environ.get("TESSERACT_TEST_CUDA_BASE_IMAGE")


@pytest.fixture(scope="module", params=GPU_EXAMPLES)
def gpu_image_name(
    request, docker_client, docker_cleanup_module, shared_dummy_image_name
):
    """Build a GPU example image once per framework for this module."""
    source = EXAMPLES_DIR / request.param
    config_override = {}
    if CUDA_BASE_IMAGE:
        config_override["build_config.base_image"] = CUDA_BASE_IMAGE
    if CUDA_MAJOR != "12":
        config_override["build_config.requirements.requirements_file"] = (
            f"tesseract_requirements_cuda{CUDA_MAJOR}.txt"
        )
    image_tag = build_tesseract(
        docker_client,
        source,
        f"{shared_dummy_image_name}-{request.param.lstrip('_')}",
        config_override=config_override,
        tag="sometag",
    )
    assert image_exists(docker_client, image_tag)
    docker_cleanup_module["images"].append(image_tag)
    return image_tag


@requires_cuda
def test_serve_cuda_ipc_roundtrip(gpu_image_name):
    """A GPU Tesseract with gpu_transport='cuda_ipc' returns correct device memory.

    Exercises the full export path end-to-end: the served container computes on
    the GPU, exports the result as a CUDA IPC handle (rather than copying to
    host), and the host client opens the handle and materialises a client-owned
    device array. The decode is framework-agnostic -- the result is a device
    wrapper exposing ``__cuda_array_interface__`` and ``__dlpack__``, read back
    here via its host-copy helper (no CuPy needed to inspect it).
    """
    from tesseract_core.runtime.cuda.ipc import IpcDeviceArray

    a = np.arange(8, dtype=np.float32)
    b = np.ones(8, dtype=np.float32)
    s = 3.0
    expected = s * a + b

    with Tesseract.from_image(
        gpu_image_name,
        gpus=["all"],
        output_format="json+base64",
        runtime_config={"gpu_transport": "cuda_ipc"},
    ) as t:
        result = t.apply({"a": a, "b": b, "s": s})

    got = result["result"]
    assert isinstance(got, IpcDeviceArray), (
        f"expected a device array from cuda_ipc, got {type(got)}"
    )
    assert hasattr(got, "__cuda_array_interface__")
    assert hasattr(got, "__dlpack__")
    np.testing.assert_allclose(got.copy_to_host(), expected, rtol=1e-5, atol=1e-5)


@requires_cuda
def test_serve_cuda_ipc_serial_reuse(gpu_image_name):
    """Serial requests each return correct data despite buffer reuse.

    The server releases the previously exported buffer at the start of each
    request, so back-to-back calls must not corrupt each other's results.
    """
    with Tesseract.from_image(
        gpu_image_name,
        gpus=["all"],
        output_format="json+base64",
        runtime_config={"gpu_transport": "cuda_ipc"},
    ) as t:
        for i in range(3):
            a = np.full(4, float(i), dtype=np.float32)
            b = np.zeros(4, dtype=np.float32)
            result = t.apply({"a": a, "b": b, "s": 2.0})
            got = result["result"].copy_to_host()
            np.testing.assert_allclose(got, 2.0 * a, rtol=1e-5, atol=1e-5)
