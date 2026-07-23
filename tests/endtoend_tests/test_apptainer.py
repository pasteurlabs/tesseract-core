# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the Apptainer backend.

Following AGENTS.md, these run against a real Apptainer install with real
converted Tesseract images and no mocks. They are skipped automatically when
Apptainer (or Docker, needed to build the images to convert) is unavailable. The
recommended-and-tested workflow mirrors what users do on a cluster: build the
image with Docker, convert it into the SIF store, then run/serve natively.
"""

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tesseract_core import Tesseract
from tesseract_core.sdk import engine

UNIT_TESSERACT = "helloworld"


def _apptainer_available() -> bool:
    if shutil.which("apptainer") is None:
        return False
    try:
        subprocess.run(["apptainer", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _apptainer_available(),
    reason="Apptainer is required for these tests",
)


@pytest.fixture(scope="module")
def apptainer_backend(tmp_path_factory, monkeypatch_module):
    """Configure the Apptainer backend with an isolated, temporary SIF store."""
    store = tmp_path_factory.mktemp("apptainer_store")
    monkeypatch_module.setenv("TESSERACT_CONTAINER_BACKEND", "apptainer")
    monkeypatch_module.setenv("TESSERACT_APPTAINER_IMAGE_DIR", str(store))
    from tesseract_core.sdk import config

    config.update_config(container_backend="apptainer", apptainer_image_dir=str(store))
    yield store
    # Best-effort cleanup of any instances left running.
    subprocess.run(["apptainer", "instance", "stop", "--all"], capture_output=True)
    config.update_config(container_backend="docker")


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch (pytest's built-in fixture is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def sif_image(apptainer_backend):
    """Build a helloworld Docker image and convert it into the SIF store."""
    src = Path(__file__).parents[2] / "examples" / UNIT_TESSERACT
    # Build with Docker (build is always Docker-based).
    result = subprocess.run(
        ["tesseract", "build", str(src)],
        capture_output=True,
        text=True,
        env=_docker_env(),
    )
    assert result.returncode == 0, result.stderr
    # Convert into the SIF store.
    return engine.build_sif(UNIT_TESSERACT, "latest")


def _docker_env() -> dict:
    """Environment for a Docker-backed build (build must use Docker)."""
    import os

    env = dict(os.environ)
    env["TESSERACT_CONTAINER_BACKEND"] = "docker"
    return env


def test_capabilities():
    """The Apptainer backend advertises host-only, no-build capabilities."""
    from tesseract_core.sdk.container_client.apptainer import ApptainerClient

    caps = ApptainerClient.capabilities
    assert caps.name == "apptainer"
    assert caps.supports_build is False
    assert caps.supports_networks is False
    assert caps.supports_port_mapping is False
    assert caps.needs_user_mapping is False


def test_build_sif_and_list(sif_image):
    """build_sif stores a SIF that list() can see."""
    images = engine.get_tesseract_images()
    tags = {tag for img in images for tag in (img.tags or [])}
    assert any(UNIT_TESSERACT in tag for tag in tags)


def test_run_one_shot(sif_image, tmp_path):
    """A one-shot `apply` runs and returns the expected result."""
    payload = tmp_path / "payload.json"
    payload.write_text('{"inputs": {"name": "Apptainer"}}')
    stdout, _ = engine.run_tesseract(
        f"{UNIT_TESSERACT}:latest",
        "apply",
        [f"@{payload}"],
    )
    assert "Hello Apptainer!" in stdout


def test_run_direct_sif_path(sif_image, apptainer_backend, tmp_path):
    """`run` accepts a direct path to a .sif file, not only a store reference."""
    sif_path = Path(apptainer_backend) / UNIT_TESSERACT / "latest.sif"
    assert sif_path.is_file()
    payload = tmp_path / "payload.json"
    payload.write_text('{"inputs": {"name": "Direct"}}')
    stdout, _ = engine.run_tesseract(str(sif_path), "apply", [f"@{payload}"])
    assert "Hello Direct!" in stdout


def test_serve_ps_teardown(sif_image):
    """Serve -> ps -> teardown cycle works via the instance API."""
    name, container = engine.serve(f"{UNIT_TESSERACT}:latest")
    try:
        assert container.host_port is not None
        running = {c.name for c in engine.get_tesseract_containers()}
        assert name in running
    finally:
        engine.teardown(name)
    time.sleep(1)
    running = {c.name for c in engine.get_tesseract_containers()}
    assert name not in running


def test_sdk_from_image(sif_image):
    """Tesseract.from_image().served() works end to end."""
    with Tesseract.from_image(f"{UNIT_TESSERACT}:latest") as t:
        assert set(t.available_endpoints) >= {"apply", "health"}
        result = t.apply({"name": "SDK"})
        assert result["greeting"] == "Hello SDK!"


def test_server_logs(sif_image):
    """server_logs returns the served Tesseract's captured output."""
    t = Tesseract.from_image(f"{UNIT_TESSERACT}:latest")
    t.serve()
    try:
        time.sleep(1)
        t.apply({"name": "Logs"})
        logs = t.server_logs()
        assert "Application startup" in logs or "Uvicorn" in logs
    finally:
        t.teardown()


def test_unsupported_network_errors(sif_image):
    """Serving with a user-defined network fails loudly (no silent degrade)."""
    from tesseract_core.sdk.exceptions import UserError

    with pytest.raises(UserError, match="host-network"):
        engine.serve(f"{UNIT_TESSERACT}:latest", network="mynet")


def test_pull_from_local_daemon(apptainer_backend):
    """Pull converts an image from the local Docker daemon into the store."""
    # Ensure a Docker image exists to pull.
    src = Path(__file__).parents[2] / "examples" / UNIT_TESSERACT
    build = subprocess.run(
        ["tesseract", "build", str(src)],
        capture_output=True,
        text=True,
        env=_docker_env(),
    )
    assert build.returncode == 0, build.stderr
    image = engine.pull_image(f"docker-daemon:{UNIT_TESSERACT}:latest")
    assert any(UNIT_TESSERACT in tag for tag in (image.tags or []))
