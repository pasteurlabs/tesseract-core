# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import traceback

from typer.testing import CliRunner

from tesseract_core.sdk.cli import app


def image_exists(client, image_name):
    # Docker images may be prefixed with the registry URL
    return any(
        tag.split("/")[-1] == image_name
        for img in client.images.list()
        for tag in img.tags
    )


def print_debug_info(result):
    """Print debug info from result of a CLI command if it failed."""
    if result.exit_code == 0:
        return
    print(result.output)
    if result.exc_info:
        traceback.print_exception(*result.exc_info)


def build_tesseract(sourcedir, image_name, tag=None, build_retries=3):
    cli_runner = CliRunner()

    build_args = [
        "--loglevel",
        "debug",
        "build",
        str(sourcedir),
        "--config-override",
        f"name={image_name}",
    ]

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
    return image_name
