# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backwards-compatible entry point for the Docker container backend.

The Docker backend now lives in :mod:`tesseract_core.sdk.container_client.docker`
and shared exceptions in :mod:`tesseract_core.sdk.container_client.exceptions`, as
part of the container-backend abstraction that also supports Apptainer. This module
re-exports the public names so existing imports (including from downstream packages
such as tesseract-jax and tesseract-torch) keep working:

    from tesseract_core.sdk.docker_client import Container, Containers, NotFound
"""

from .container_client.docker import (
    CLIDockerClient,
    Container,
    Containers,
    Image,
    Images,
    Networks,
    Volume,
    Volumes,
    _get_docker_executable,
    _is_valid_docker_tag,
    build_docker_image,
    get_docker_metadata,
    is_podman,
)
from .container_client.exceptions import (
    APIError,
    BuildError,
    ContainerError,
    DockerException,
    ImageNotFound,
    NotFound,
)

__all__ = [
    "APIError",
    "BuildError",
    "CLIDockerClient",
    "Container",
    "ContainerError",
    "Containers",
    "DockerException",
    "Image",
    "ImageNotFound",
    "Images",
    "Networks",
    "NotFound",
    "Volume",
    "Volumes",
    "_get_docker_executable",
    "_is_valid_docker_tag",
    "build_docker_image",
    "get_docker_metadata",
    "is_podman",
]
