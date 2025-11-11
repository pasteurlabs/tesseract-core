# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import traceback
from pathlib import Path
from uuid import uuid4

import docker.errors
import numpy as np
from typer.testing import CliRunner

from tesseract_core.sdk.cli import app
from tesseract_core.sdk.docker_client import (
    CLIDockerClient,
    ImageNotFound,
    NotFound,
)


def image_exists(client, image_name, tesseract_only: bool = True):
    """Checks if image name exists."""
    # Docker images may be prefixed with the registry URL
    kwargs = {}
    if isinstance(client, CLIDockerClient):
        kwargs["tesseract_only"] = tesseract_only

    try:
        client.images.get(image_name, **kwargs)
        return True
    except (ImageNotFound, docker.errors.ImageNotFound):
        return False


def container_exists(client, container_name_or_id, tesseract_only: bool = True):
    """Checks if containers exists."""
    try:
        client.containers.get(container_name_or_id, tesseract_only)
        return True
    except NotFound:
        return False


def print_debug_info(result):
    """Print debug info from result of a CLI command if it failed."""
    if result.exit_code == 0:
        return
    print(result.output)
    if result.exc_info:
        traceback.print_exception(*result.exc_info)


def build_tesseract(
    client, sourcedir, image_name, config_override=None, tag=None, build_retries=3
):
    cli_runner = CliRunner()
    build_args = [
        "--loglevel",
        "debug",
        "build",
        str(sourcedir),
        "--config-override",
        f"name={image_name}",
    ]

    if config_override is not None:
        for key, val in config_override.items():
            build_args.extend(["--config-override", f"{key}={val}"])

    if tag is not None:
        build_args.extend(["--tag", tag])
        image_name = f"{image_name}:{tag}"
    else:
        image_name = f"{image_name}:latest"

    for _ in range(build_retries):
        result = cli_runner.invoke(
            app,
            build_args,
            catch_exceptions=False,
        )
        # Retry if the build fails with EOF error (connectivity issue)
        # See https://github.com/docker/buildx/issues/2064
        is_expected_err = "error reading from server: EOF" in result.output
        if not is_expected_err:
            break

    print_debug_info(result)
    assert result.exit_code == 0, result.exception

    image_tag = json.loads(result.stdout.strip())[0]

    # This raise an error if the image does not exist
    client.images.get(image_tag)
    return image_tag


def encode_array(arr, as_json=False, encoding="base64", basedir=None):
    """Helper function to encode a numpy array into Tesseract-friendly format."""
    arr = np.asarray(arr)

    if encoding == "json":
        buffer = arr.tolist()
    elif encoding == "base64":
        buffer = base64.b64encode(arr.tobytes()).decode()
    elif encoding == "binref":
        filename = f"{uuid4()}.bin"
        if basedir is None:
            raise ValueError("basedir must be provided for binref encoding")
        filepath = Path(basedir) / filename
        with open(filepath, "wb") as f:
            f.write(arr.tobytes())
        buffer = str(filename)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    out = {
        "object_type": "array",
        "shape": arr.shape,
        "dtype": arr.dtype.name,
        "data": {
            "buffer": buffer,
            "encoding": encoding,
        },
    }
    if as_json:
        return json.dumps(out, separators=(",", ":"))

    return out
