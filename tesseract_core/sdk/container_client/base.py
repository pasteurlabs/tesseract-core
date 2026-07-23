# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-agnostic container client interface.

The engine and CLI interact with container backends (Docker/Podman, Apptainer)
exclusively through the :class:`ContainerClient` protocol defined here and branch
on :class:`Capabilities` rather than on backend names. This keeps backend-specific
quirks (Podman user-namespace flags, Apptainer host-only networking, ...) out of
the engine.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class AbstractContainer:
    """Common base for backend Container classes.

    Exists so engine/SDK code can annotate parameters and return types with a
    single backend-neutral type (accepted by typeguard for both the Docker and
    Apptainer concrete classes). The real attribute/method surface is defined by
    each backend and documented on the Docker implementation.
    """


class AbstractImage:
    """Common base for backend Image classes (see :class:`AbstractContainer`)."""


@dataclass(frozen=True)
class Capabilities:
    """Describes what a container backend can and cannot do.

    The engine consults these flags instead of checking backend names, so that new
    backends slot in without touching call sites. Docker/Podman support everything;
    Apptainer is host-network only and cannot build.
    """

    #: Named user-defined networks (``docker network create``) and attaching
    #: containers to them.
    supports_networks: bool = True
    #: Publishing container ports on chosen host ports (``-p host:container``).
    #: Apptainer is host-network only: the runtime binds host ports directly.
    supports_port_mapping: bool = True
    #: Restart policies (``--restart unless-stopped``).
    supports_restart_policy: bool = True
    #: Building images from a Dockerfile. Apptainer cannot build; images are built
    #: with Docker/Podman and converted, or pulled from a registry.
    supports_build: bool = True
    #: Whether the backend needs an explicit uid:gid mapping. Docker runs images as
    #: their declared USER unless overridden; Apptainer always runs as the invoking
    #: user, so no mapping is needed.
    needs_user_mapping: bool = True
    #: Whether the backend applies Podman-specific ``--userns keep-id`` handling.
    is_podman: bool = False
    #: Human-readable backend name for error messages and diagnostics.
    name: str = "docker"


@runtime_checkable
class ContainerClient(Protocol):
    """Protocol implemented by every container backend.

    Only the surface actually consumed by the engine, CLI, and SDK is included.
    The ``images``, ``containers``, ``volumes``, and ``networks`` attributes are
    namespaces mirroring the docker-py facade; see the Docker backend for the
    canonical method signatures.
    """

    capabilities: Capabilities
    images: Any
    containers: Any
    volumes: Any
    networks: Any

    @staticmethod
    def info() -> tuple:
        """Return backend info, raising on an unreachable/absent backend."""
        ...
