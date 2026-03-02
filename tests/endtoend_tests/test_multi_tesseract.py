# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for multi-Tesseract workflows."""

import json
import os
import subprocess

from common import build_tesseract, image_exists

from tesseract_core.sdk.cli import app
from tesseract_core.sdk.config import get_config


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
