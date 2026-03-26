# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for multi-Tesseract workflows."""

import json
import os
import subprocess

from common import build_tesseract, image_exists

from tesseract_core.sdk.cli import app
from tesseract_core.sdk.config import get_config

NUM_IMAGE_REF_ITERATIONS = 3


def test_multi_helloworld_endtoend(
    cli_runner,
    docker_client,
    unit_tesseracts_parent_dir,
    dummy_image_name,
    dummy_network_name,
    docker_cleanup,
):
    """Test that multi-helloworld example can be built, served, and executed."""
    # Build Tesseract images
    img_names = []
    for tess_name in ("_multi-tesseract/multi-helloworld", "helloworld"):
        img_name = build_tesseract(
            docker_client,
            unit_tesseracts_parent_dir / tess_name,
            dummy_image_name + f"_{tess_name}",
            tag="sometag",
        )
        img_names.append(img_name)
        assert image_exists(docker_client, img_name)
        docker_cleanup["images"].append(img_name)

    config = get_config()
    docker = config.docker_executable

    result = subprocess.run(
        [*docker, "network", "create", dummy_network_name],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0, result.stderr
    docker_cleanup["networks"].append(dummy_network_name)

    # Serve target Tesseract
    multi_helloworld_tesseract_img_name, helloworld_tesseract_img_name = img_names
    result = cli_runner.invoke(
        app,
        [
            "serve",
            helloworld_tesseract_img_name,
            "--network",
            dummy_network_name,
            "--network-alias",
            "helloworld",
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0, result.output
    docker_cleanup["containers"].append(json.loads(result.stdout)["container_name"])
    api_port = json.loads(result.stdout)["containers"][0]["networks"][
        dummy_network_name
    ]["port"]
    payload = json.dumps(
        {
            "inputs": {
                "name": "you",
                "helloworld_tesseract_url": f"http://helloworld:{api_port}",
            }
        }
    )

    # Run multi-helloworld Tesseract
    result = cli_runner.invoke(
        app,
        [
            "run",
            multi_helloworld_tesseract_img_name,
            "apply",
            payload,
            "--network",
            dummy_network_name,
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0, result.output
    assert "The helloworld Tesseract says: Hello you!" in result.output


def test_tesseractreference_endtoend(
    cli_runner,
    docker_client,
    unit_tesseracts_parent_dir,
    dummy_image_name,
    dummy_network_name,
    docker_cleanup,
):
    """Test that tesseractreference example can be built and executed, calling helloworld tesseract."""
    # Build Tesseract images
    img_names = []
    for tess_name in ("tesseractreference", "helloworld"):
        img_name = build_tesseract(
            docker_client,
            unit_tesseracts_parent_dir / tess_name,
            dummy_image_name + f"_{tess_name}",
            tag="sometag",
        )
        img_names.append(img_name)
        assert image_exists(docker_client, img_name)
        docker_cleanup["images"].append(img_name)

    tesseractreference_img_name, helloworld_img_name = img_names

    # Create Docker network
    config = get_config()
    docker = config.docker_executable

    result = subprocess.run(
        [*docker, "network", "create", dummy_network_name],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0, result.stderr
    docker_cleanup["networks"].append(dummy_network_name)

    # Serve helloworld Tesseract on the shared network
    result = cli_runner.invoke(
        app,
        [
            "serve",
            helloworld_img_name,
            "--network",
            dummy_network_name,
            "--network-alias",
            "helloworld",
            "--port",
            "8000",
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0, result.output
    serve_meta = json.loads(result.stdout)
    docker_cleanup["containers"].append(serve_meta["container_name"])

    # Test url type
    url_payload = (
        '{"inputs": {"target": {"type": "url", "ref": "http://helloworld:8000"}}}'
    )
    result = cli_runner.invoke(
        app,
        [
            "run",
            tesseractreference_img_name,
            "apply",
            url_payload,
            "--network",
            dummy_network_name,
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0, result.output
    output_data = json.loads(result.stdout)
    expected_result = "Hello Alice! Hello Bob!"
    assert output_data["result"] == expected_result

    # Test image type
    image_payload = json.dumps(
        {
            "inputs": {
                "target": {
                    "type": "image",
                    "ref": helloworld_img_name,
                }
            }
        }
    )
    result = subprocess.run(
        [
            "tesseract-runtime",
            "apply",
            image_payload,
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TESSERACT_API_PATH": str(
                unit_tesseracts_parent_dir / "tesseractreference/tesseract_api.py"
            ),
        },
    )
    assert result.returncode == 0, result.stderr
    output_data = json.loads(result.stdout)
    assert output_data["result"] == expected_result

    # Test api_path type
    path_payload = json.dumps(
        {
            "inputs": {
                "target": {
                    "type": "api_path",
                    "ref": str(
                        unit_tesseracts_parent_dir / "helloworld/tesseract_api.py"
                    ),
                }
            }
        }
    )
    result = subprocess.run(
        [
            "tesseract-runtime",
            "apply",
            path_payload,
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TESSERACT_API_PATH": str(
                unit_tesseracts_parent_dir / "tesseractreference/tesseract_api.py"
            ),
        },
    )
    assert result.returncode == 0, result.stderr
    output_data = json.loads(result.stdout)
    assert output_data["result"] == expected_result


def test_tesseractreference_image_does_not_leak_containers(
    docker_client,
    unit_tesseracts_parent_dir,
    dummy_image_name,
    docker_cleanup,
):
    """Tesseract.serve() registers an atexit handler via atexit.register(self.teardown).

    This bound method holds a strong reference to the Tesseract object, preventing it
    from being garbage collected. In a loop, each iteration's Tesseract (and its
    container) stays alive until process exit, accumulating containers.

    This matters for TesseractReference type="image", which calls
    Tesseract.from_image().serve() during Pydantic validation. In an optimization loop,
    a new container is spawned per iteration and none are collected until the process ends.
    """
    from tesseract_core import Tesseract

    # Build helloworld image
    helloworld_img_name = build_tesseract(
        docker_client,
        unit_tesseracts_parent_dir / "helloworld",
        dummy_image_name + "_helloworld",
        tag="sometag",
    )
    docker_cleanup["images"].append(helloworld_img_name)

    containers_before = set(c.id for c in docker_client.containers.list())

    # Simulate what happens in an optimization loop: each iteration creates
    # a Tesseract from image, serves it, uses it, and should clean it up.
    leaked_containers = []
    for i in range(NUM_IMAGE_REF_ITERATIONS):
        tess = Tesseract.from_image(helloworld_img_name)
        tess.serve()

        # Use it
        result = tess.apply({"name": f"iteration_{i}"})
        assert "greeting" in result

        # A well-behaved API should clean up after itself, but currently
        # there's no automatic teardown. Track what was created.
        leaked_containers.append(tess._serve_context["container_name"])

    containers_after = set(c.id for c in docker_client.containers.list())
    leaked = containers_after - containers_before

    # Clean up for test hygiene
    for name in leaked_containers:
        try:
            docker_client.containers.get(name).remove(force=True)
        except Exception:
            pass

    assert len(leaked) == 0, (
        f"Leaked {len(leaked)} container(s) across {NUM_IMAGE_REF_ITERATIONS} "
        f"iterations. Each TesseractReference type='image' spawns a container "
        f"that is never cleaned up within the same process."
    )
