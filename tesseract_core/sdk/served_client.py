# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The interface a running Tesseract presents, however it is being served.

Implemented by :mod:`tesseract_core.sdk.docker_client` for containers and by
:mod:`tesseract_core.sdk.local_client` for subprocesses. Kept apart from both so
that neither client has to import the other to get its base class, and so the
interface stays free of anything either transport needs.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Protocol


class ServedTesseract(Protocol):
    """A Tesseract that has been started and can be reached, inspected and stopped.

    Satisfied by :class:`~tesseract_core.sdk.docker_client.Container` and by
    :class:`~tesseract_core.sdk.local_client.TesseractProcess`, so callers that
    only need to talk to a Tesseract, read its output or shut it down need not
    know which of the two they hold.

    Structural, so neither has to declare allegiance to it: a `Container` is a
    mirror of docker-py's, and fits this by having the methods docker-py gave it
    rather than by inheriting anything from us.

    Deliberately narrow: it promises nothing about separating stdout from stderr,
    since a Tesseract served as a bare process writes both to one file.
    """

    host_ip: str | None
    host_port: str | None

    @property
    def url(self) -> str:
        """Base URL the Tesseract is serving on."""

    def reload(self) -> None:
        """Read the Tesseract's state again."""

    def remove(self, v: bool = False, link: bool = False, force: bool = False) -> None:
        """Dispose of the Tesseract, leaving nothing of it behind.

        docker-py's ``Container.remove`` signature exactly, since that is what a
        `Container` has and this is structural -- including refusing one that is
        still running unless ``force`` is set. ``v`` and ``link`` are Docker's
        and mean nothing to a Tesseract served any other way.
        """

    def wait(self, timeout: float | None = None) -> dict:
        """Wait for the Tesseract to stop, and report the status it stopped with.

        Shaped after docker-py's ``Container.wait``, down to returning a dict
        keyed by ``StatusCode``. Waits for as long as the Tesseract runs unless
        ``timeout`` says otherwise, so ask only about one you expect to have
        stopped -- and before disposing of it, since a Tesseract that is gone can
        no longer be asked.
        """

    def logs(self) -> bytes:
        """Everything the Tesseract has written so far."""


@singledispatch
def diagnose_exit(served: ServedTesseract, logs: str) -> str:
    """Anything this Tesseract can add about why it stopped running.

    The code it exited with and what it wrote are reported by whoever noticed.
    This is for what remains: a cause the transport can name and the logs cannot.
    Takes the logs as evidence, not to repeat them.

    Dispatched rather than a method, so a `Container` keeps to the shape of
    docker-py, which has nothing like this. Implementations are registered beside
    the class they are for, which is also why nothing above the clients has to
    know they exist.

    Raises:
        NotImplementedError: if nothing is registered for this kind of Tesseract.
    """
    del logs
    raise NotImplementedError(
        f"No diagnose_exit is registered for {type(served).__name__}. Register one "
        "with `@diagnose_exit.register` in the module that defines the class, "
        "returning an empty string if there is nothing to add."
    )


@singledispatch
def is_running(served: ServedTesseract) -> bool:
    """Whether this Tesseract is running now, asking again rather than recalling.

    Dispatched for the same reason as :func:`diagnose_exit`: docker-py has no
    such method, and asks you to compare `status` yourself after a `reload`.

    Raises:
        NotImplementedError: if nothing is registered for this kind of Tesseract.
    """
    raise NotImplementedError(
        f"No is_running is registered for {type(served).__name__}. Register one "
        "with `@is_running.register` in the module that defines the class."
    )
