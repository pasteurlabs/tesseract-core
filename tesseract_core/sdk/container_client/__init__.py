# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container backends for Tesseract.

Exposes a :func:`get_client` factory that returns the configured backend
(:class:`~.docker.CLIDockerClient` or :class:`~.apptainer.ApptainerClient`),
selected via ``container_backend`` in the runtime config (env
``TESSERACT_CONTAINER_BACKEND``). All backends implement the
:class:`~.base.ContainerClient` protocol and expose a
:class:`~.base.Capabilities` object.
"""

import logging

from tesseract_core.sdk.config import get_config

from .base import Capabilities, ContainerClient
from .exceptions import (
    APIError,
    BuildError,
    ContainerError,
    ContainerException,
    DockerException,
    ImageNotFound,
    NotFound,
)

logger = logging.getLogger("tesseract")

# Emit the experimental-backend warning only once per process.
_apptainer_warned = False

__all__ = [
    "APIError",
    "BuildError",
    "Capabilities",
    "ContainerClient",
    "ContainerError",
    "ContainerException",
    "DockerException",
    "ImageNotFound",
    "NotFound",
    "get_client",
]


def get_client() -> ContainerClient:
    """Return a container client for the configured backend.

    The backend is chosen by ``container_backend`` in the runtime config
    (default ``"docker"``, override with ``TESSERACT_CONTAINER_BACKEND``).
    A fresh client is returned on each call; clients are cheap wrappers around
    CLI invocations and hold no persistent connection.
    """
    backend = get_config().container_backend
    if backend == "docker":
        from .docker import CLIDockerClient

        return CLIDockerClient()
    if backend == "apptainer":
        global _apptainer_warned
        if not _apptainer_warned:
            logger.warning(
                "The Apptainer backend is experimental. Please report issues and "
                "cluster-specific findings on the community forum: "
                "https://si-tesseract.discourse.group/"
            )
            _apptainer_warned = True
        from .apptainer import ApptainerClient

        return ApptainerClient()
    raise ValueError(
        f"Unknown container backend {backend!r}; expected 'docker' or 'apptainer'."
    )
