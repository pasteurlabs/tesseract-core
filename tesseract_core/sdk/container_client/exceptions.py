# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exceptions shared by all container backends.

These were historically named after Docker (``DockerException`` and friends) and
are still imported under those names from :mod:`tesseract_core.sdk.docker_client`
for backwards compatibility. They are backend-agnostic despite the names.
"""

from typing import List as list_  # noqa: UP035


class ContainerException(Exception):
    """Base class for container backend exceptions."""


# Backwards-compatible alias; the exception hierarchy predates the backend
# abstraction and is imported as ``DockerException`` in a few places.
DockerException = ContainerException


class BuildError(ContainerException):
    """Raised when an image build fails."""

    def __init__(self, build_log: list_[str]) -> None:  # noqa: UP006
        self.build_log = build_log

    def __str__(self) -> str:
        return (
            "Image build failed. Please check the build log for details:\n"
            + "\n".join(self.build_log)
        )


class ContainerError(ContainerException):
    """Raised when a container encounters an error."""

    def __init__(
        self,
        container: str | None,
        exit_status: int,
        command: str,
        image: str,
        stderr: bytes,
    ) -> None:
        self.container = container
        self.exit_status = exit_status
        self.command = command
        self.image = image
        self.stderr = stderr


class APIError(ContainerException):
    """Raised when a container backend API error occurs."""


class NotFound(ContainerException):
    """Raised when a container resource is not found."""


class ImageNotFound(NotFound):
    """Raised when an image is not found."""
