# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Primitives shared by everything that serves a Tesseract over HTTP.

Choosing a port to serve on, waiting for the server to answer, and telling a lost
race for a port apart from a Tesseract that is genuinely broken. None of it is
particular to how the Tesseract is run, so none of it belongs in the module that
knows how to run one.
"""

import logging
import random
import socket
import time
from collections.abc import Sequence
from contextlib import closing

import requests

from .docker_client import APIError, Container, is_running

logger = logging.getLogger("tesseract")


class _PortInUseError(RuntimeError):
    """Container failed to start because its port was already bound.

    Signals that a fresh port should be picked and startup retried. Only raised
    when we chose the port ourselves; a user-supplied port is never retried.

    A port collision surfaces in one of two ways depending on network mode:
    - port-mapping mode: the Docker daemon fails to publish the host port and
      ``containers.run`` raises ``ContainerError`` ("port is already allocated").
    - host networking: the container binds the host port directly, so the
      failure appears in the container logs as uvicorn's "address already in
      use" and is detected in ``_wait_for_health``.
    """


# Substrings container runtimes use to report a host port already being taken.
_PORT_CONFLICT_MARKERS = ("address already in use", "port is already allocated")


def get_free_port(
    within_range: tuple[int, int] = (49152, 65535),
    exclude: Sequence[int] = (),
) -> int:
    """Find a random free port to use for HTTP."""
    start, end = within_range
    if start < 0 or end > 65535 or start > end:
        raise ValueError("Invalid port range, must be between 0 and 65535")

    # Try random ports in the given range
    portlist = list(range(start, end))
    random.shuffle(portlist)
    for port in portlist:
        if port in exclude:
            continue
        # Check if the port is free
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                # Port is already in use
                continue
            else:
                return port
    raise RuntimeError(f"No free ports found in range {start}-{end}")


def _is_port_conflict(stderr: str) -> bool:
    """Whether runtime stderr/logs indicate a host port collision."""
    lowered = stderr.lower()
    return any(marker in lowered for marker in _PORT_CONFLICT_MARKERS)


def _retry_or_raise_port_conflict(
    port: str, auto_port: bool, attempt: int, max_attempts: int
) -> None:
    """Decide whether a port collision should be retried.

    Returns normally if the caller should retry with a fresh port; raises
    otherwise. A user-supplied fixed port is never retried (we must not
    silently move the Tesseract elsewhere), and auto-selected ports raise once
    the attempt budget is exhausted.
    """
    if not auto_port:
        # User asked for this exact port; surface the collision as-is.
        raise _PortInUseError(f"Port {port} was already in use")
    if attempt + 1 >= max_attempts:
        raise RuntimeError(
            f"Failed to find a free port after {max_attempts} attempts"
        ) from None
    logger.info(f"Port {port} was taken, retrying with a new port...")


def _wait_for_health(
    container: Container, ping_ip: str, port: str, timeout: float = 30
) -> None:
    """Poll a container's /health endpoint until it responds 200 or timeout expires."""
    while True:
        try:
            response = requests.get(f"http://{ping_ip}:{port}/health")
        except requests.exceptions.ConnectionError:
            pass
        else:
            if response.status_code == 200:
                return

        time.sleep(0.1)
        timeout -= 0.1

        if timeout < 0 or not is_running(container):
            logs_text = ""
            try:
                logs_text = container.logs(stdout=True, stderr=True).decode()
                logger.error(
                    f"Tesseract container {container.name} failed to start:\n{logs_text}"
                )
            except APIError as ex:
                logger.warning(
                    f"Failed to get logs for container {container.name}: {ex}"
                )
            try:
                container.stop()
            except APIError as ex:
                logger.warning(f"Failed to stop container {container.name}: {ex}")

            # A port collision is racy and worth retrying with a fresh port;
            # distinguish it from genuine startup failures so those still fail
            # fast.
            if _is_port_conflict(logs_text):
                raise _PortInUseError(f"Port {port} was already in use")

            if timeout < 0:
                raise TimeoutError("Tesseract did not start in time")
            else:
                raise RuntimeError("Tesseract failed to start")
