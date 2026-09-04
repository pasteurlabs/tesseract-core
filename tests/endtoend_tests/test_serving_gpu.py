# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GPU tests for serving Tesseracts with the CUDA IPC output format.

Unlike ``tests/test_cuda_ipc.py`` (which drives the encode/decode functions
in-process), these tests build a real GPU Tesseract image, serve it in a
container with ``--gpus all`` and ``--ipc=host``, and round-trip device memory
across the process/container boundary via a genuine ``cudaIpcMemHandle_t``.

Requires a physical CUDA GPU and Docker with the NVIDIA container runtime. CuPy
is used only as a convenient GPU-availability probe on the host; the decoded
result is inspected framework-agnostically (via its host-copy helper), so the
host does not need CuPy to read cuda_ipc outputs. They are marked ``gpu`` so
GPU-less CI runners skip them.
"""

import tempfile
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


@pytest.fixture(scope="module")
def gpu_vmm_image_name(docker_client, docker_cleanup_module, shared_dummy_image_name):
    """Build the VMM (JAX-output) example image once for this module."""
    source = EXAMPLES_DIR / "_gpu_vmm"
    image_tag = build_tesseract(
        docker_client, source, shared_dummy_image_name, tag="vmmtag"
    )
    assert image_exists(docker_client, image_tag)
    docker_cleanup_module["images"].append(image_tag)
    return image_tag


@requires_cuda
def test_serve_cuda_ipc_roundtrip(gpu_image_name):
    """A GPU Tesseract served with json+cuda_ipc returns correct device memory.

    Exercises the full export path end-to-end: the served container computes on
    the GPU, exports the result as a CUDA IPC handle (rather than copying to
    host), and the host client opens the handle and materialises a client-owned
    device array. The decode is framework-agnostic -- the result is a device
    wrapper exposing ``__cuda_array_interface__`` and ``__dlpack__``, read back
    here via its host-copy helper (no CuPy needed to inspect it).
    """
    from tesseract_core.runtime.cuda_ipc import IpcDeviceArray

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
        output_format="json+cuda_ipc",
        runtime_config={"enable_experimental_cuda_ipc": True},
    ) as t:
        for i in range(3):
            a = np.full(4, float(i), dtype=np.float32)
            b = np.zeros(4, dtype=np.float32)
            result = t.apply({"a": a, "b": b, "s": 2.0})
            got = result["result"].copy_to_host()
            np.testing.assert_allclose(got, 2.0 * a, rtol=1e-5, atol=1e-5)


@requires_cuda
def test_serve_vmm_roundtrip(gpu_vmm_image_name):
    """A served JAX (VMM) output round-trips over json+cuda_ipc via the VMM path.

    The Tesseract computes on JAX, so its result is VMM-backed and the runtime
    exports it with ``cuMemExportToShareableHandle`` rather than legacy
    ``cudaIpcGetMemHandle``. The fd that names the export is passed out-of-band
    over a Unix socket, so host and container must share the directory the
    socket lives in: mount a shared dir and point ``TESSERACT_VMM_SOCKET_DIR``
    at it. ``--ipc=host`` is added automatically for the cuda_ipc format.
    """
    from tesseract_core.runtime.cuda_ipc import IpcDeviceArray

    a = np.arange(8, dtype=np.float32)
    b = np.ones(8, dtype=np.float32)
    s = 3.0
    expected = s * a + b

    with tempfile.TemporaryDirectory() as sock_dir:
        with Tesseract.from_image(
            gpu_vmm_image_name,
            gpus=["all"],
            output_format="json+cuda_ipc",
            # Mount read-write: the server binds its fd-passing socket here.
            volumes=[f"{sock_dir}:{sock_dir}:rw"],
            environment={"TESSERACT_VMM_SOCKET_DIR": sock_dir},
            runtime_config={"enable_experimental_cuda_ipc": True},
        ) as t:
            result = t.apply({"a": a, "b": b, "s": s})

        got = result["result"]
        assert isinstance(got, IpcDeviceArray), (
            f"expected a device array from the VMM path, got {type(got)}"
        )
        assert hasattr(got, "__cuda_array_interface__")
        np.testing.assert_allclose(got.copy_to_host(), expected, rtol=1e-5, atol=1e-5)
