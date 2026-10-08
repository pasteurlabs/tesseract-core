# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GPU tests for serving Tesseracts with the CUDA IPC output format.

Unlike ``tests/test_cuda_ipc.py`` (which drives the encode/decode functions
in-process), these tests build a real GPU Tesseract image, serve it in a
container with ``--gpus all`` and ``--ipc=host``, and round-trip device memory
across the process/container boundary via a genuine ``cudaIpcMemHandle_t``. One
image is built per GPU array framework (CuPy, JAX, PyTorch) so the export path
is covered against the device arrays each framework returns.

Requires a physical CUDA GPU and Docker with the NVIDIA container runtime. CuPy
is used only as a convenient GPU-availability probe on the host; the decoded
result is inspected framework-agnostically (via its host-copy helper), so the
host does not need CuPy to read cuda_ipc outputs. They are marked ``gpu`` so
GPU-less CI runners skip them.
"""

import os
from contextlib import contextmanager
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
except Exception:  # noqa: BLE001 -- CuPy/CUDA probe; failure modes vary by host
    _CUDA_AVAILABLE = False

requires_cuda = pytest.mark.skipif(
    not _CUDA_AVAILABLE, reason="CUDA + CuPy not available on host"
)


# The framework runs inside the container, so the host stays framework-agnostic
# and needs only a GPU to decode the handle (hence the plain requires_cuda skip).
GPU_EXAMPLES = ["_gpu_cupy", "_gpu_jax", "_gpu_torch"]

# The CUDA runtime comes from the framework wheels (the runtime loader prefers
# the pip-wheel libcudart), so the wheel's CUDA major is what libcudart resolves
# to. The example ships CUDA 12 by default plus a tesseract_requirements_cuda13.txt;
# CI sets TESSERACT_TEST_CUDA_MAJOR per matrix leg and pre-builds with the same
# override, so the fixture builds hit the warm layer cache. A local GPU run
# without it falls back to the CUDA 12 default the examples already carry.
CUDA_MAJOR = os.environ.get("TESSERACT_TEST_CUDA_MAJOR", "12")


@pytest.fixture(scope="module", params=GPU_EXAMPLES)
def gpu_image_name(
    request, docker_client, docker_cleanup_module, shared_dummy_image_name
):
    """Build a GPU example image once per framework for this module."""
    # jax only has a cuda13 build (its cuda12 plugin won't register under the CI's
    # CUDA 13 driver), so run it on the 13 leg only; cupy/torch cover .so.12.
    if request.param == "_gpu_jax" and CUDA_MAJOR == "12":
        pytest.skip("_gpu_jax runs on the CUDA 13 leg only")

    source = EXAMPLES_DIR / request.param
    config_override = {}
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


def _forbid_host_copy_env() -> dict[str, str]:
    """Container env that makes implicit device-to-host copies raise.

    Returns a fresh dict because serving mutates the ``environment`` it is given.
    """
    return {"TESSERACT_FORBID_DEVICE_HOST_COPY": "1"}


@contextmanager
def _serve_cuda_ipc(image_name: str):
    """Serve a GPU example with gpu_transport='cuda_ipc' and host copies forbidden."""
    with Tesseract.from_image(
        image_name,
        gpus=["all"],
        output_format="json+base64",
        runtime_config={"gpu_transport": "cuda_ipc"},
        environment=_forbid_host_copy_env(),
    ) as t:
        yield t


def _assert_device_result(got, expected: np.ndarray) -> None:
    """Assert a cuda_ipc result is a float32 device array holding ``expected``."""
    from tesseract_core.runtime.cuda.ipc import IpcDeviceArray

    assert isinstance(got, IpcDeviceArray), (
        f"expected a device array from cuda_ipc, got {type(got)}"
    )
    assert got.dtype == np.float32
    np.testing.assert_allclose(got.copy_to_host(), expected, rtol=1e-5, atol=1e-5)


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
    a = np.arange(8, dtype=np.float32)
    b = np.ones(8, dtype=np.float32)
    s = 3.0

    with _serve_cuda_ipc(gpu_image_name) as t:
        result = t.apply({"a": a, "b": b, "s": s})

    got = result["result"]
    assert hasattr(got, "__cuda_array_interface__")
    assert hasattr(got, "__dlpack__")
    _assert_device_result(got, s * a + b)


@requires_cuda
def test_serve_cuda_ipc_serial_reuse(gpu_image_name):
    """Serial requests each return correct data despite buffer reuse.

    The server releases the previously exported buffer at the start of each
    request, so back-to-back calls must not corrupt each other's results.
    """
    with _serve_cuda_ipc(gpu_image_name) as t:
        for i in range(3):
            a = np.full(4, float(i), dtype=np.float32)
            b = np.zeros(4, dtype=np.float32)
            result = t.apply({"a": a, "b": b, "s": 2.0})
            got = result["result"].copy_to_host()
            np.testing.assert_allclose(got, 2.0 * a, rtol=1e-5, atol=1e-5)


# The examples compute result = s * a + b, so d(result)/da = s * I and
# d(result)/db = I. Gradient endpoints return device memory just like apply.
_GRAD_INPUTS = {
    "a": np.arange(6, dtype=np.float32),
    "b": np.ones(6, dtype=np.float32),
    "s": 3.0,
}


@requires_cuda
def test_serve_cuda_ipc_abstract_eval(gpu_image_name):
    with _serve_cuda_ipc(gpu_image_name) as t:
        result = t.abstract_eval(
            {
                "a": {"shape": [6], "dtype": "float32"},
                "b": {"shape": [6], "dtype": "float32"},
                "s": 3.0,
            }
        )

    assert tuple(result["result"]["shape"]) == (6,)
    assert result["result"]["dtype"] == "float32"


@requires_cuda
def test_serve_cuda_ipc_jacobian(gpu_image_name):
    with _serve_cuda_ipc(gpu_image_name) as t:
        jac = t.jacobian(_GRAD_INPUTS, jac_inputs=["a", "b"], jac_outputs=["result"])

    eye = np.eye(6, dtype=np.float32)
    _assert_device_result(jac["result"]["a"], 3.0 * eye)
    _assert_device_result(jac["result"]["b"], eye)


@requires_cuda
def test_serve_cuda_ipc_jacobian_vector_product(gpu_image_name):
    tangent = {
        "a": np.linspace(0, 1, 6, dtype=np.float32),
        "b": np.full(6, 2.0, dtype=np.float32),
    }
    with _serve_cuda_ipc(gpu_image_name) as t:
        jvp = t.jacobian_vector_product(
            _GRAD_INPUTS,
            jvp_inputs=["a", "b"],
            jvp_outputs=["result"],
            tangent_vector=tangent,
        )

    _assert_device_result(jvp["result"], 3.0 * tangent["a"] + tangent["b"])


@requires_cuda
def test_serve_cuda_ipc_vector_jacobian_product(gpu_image_name):
    cotangent = np.linspace(0, 1, 6, dtype=np.float32)
    with _serve_cuda_ipc(gpu_image_name) as t:
        vjp = t.vector_jacobian_product(
            _GRAD_INPUTS,
            vjp_inputs=["a", "b"],
            vjp_outputs=["result"],
            cotangent_vector={"result": cotangent},
        )

    _assert_device_result(vjp["a"], 3.0 * cotangent)
    _assert_device_result(vjp["b"], cotangent)
