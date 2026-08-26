# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The interface a running Tesseract presents, however it is being served.

Implemented by :mod:`tesseract_core.sdk.docker_client` for containers and by
:mod:`tesseract_core.sdk.local_client` for subprocesses. Kept apart from both so
that neither client has to import the other to get its base class, and so the
interface stays free of anything either transport needs.
"""

from __future__ import annotations

import abc


class ServedTesseract(abc.ABC):
    """A Tesseract that has been started and can be reached, inspected and stopped.

    Implemented by :class:`~tesseract_core.sdk.docker_client.Container` and by
    :class:`~tesseract_core.sdk.local_client.TesseractProcess`, so callers that
    only need to talk to a Tesseract, read its output or shut it down need not
    know which of the two they hold.

    Deliberately narrow: it promises nothing about separating stdout from stderr,
    since a Tesseract served as a bare process writes both to one file.
    """

    # Provided by subclasses, as a field or a property. Not abstract: a dataclass
    # field without a default does not satisfy an abstract property, and
    # requiring one would force every subclass to wrap its own attribute.
    host_ip: str | None
    host_port: str | None

    @property
    def url(self) -> str:
        """Base URL the Tesseract is serving on."""
        return f"http://{self.host_ip}:{self.host_port}"

    @abc.abstractmethod
    def reload(self) -> None:
        """Read the Tesseract's state again."""

    @abc.abstractmethod
    def is_running(self) -> bool:
        """Whether the Tesseract is running now."""

    @abc.abstractmethod
    def teardown(self) -> None:
        """Stop the Tesseract and dispose of it, leaving nothing behind.

        Neither ``stop`` nor ``remove`` would do: the first leaves a stopped
        container in place, and the second refuses to touch a running one unless
        forced. Named for what `tesseract teardown` already means.
        """

    @abc.abstractmethod
    def exit_code(self) -> int | None:
        """The status it exited with, or None if it has not exited.

        A snapshot, like ``is_running``: read after the Tesseract is disposed of,
        it still reports what it stopped with.
        """

    @abc.abstractmethod
    def logs(self) -> bytes:
        """Everything the Tesseract has written so far."""

    def diagnose_exit(self, logs: str) -> str:
        """Anything this Tesseract can add about why it stopped running.

        The code it exited with and what it wrote are reported by whoever noticed.
        This is for what remains: a cause the transport can name and the logs
        cannot. Empty by default, for a Tesseract with nothing to add. Takes the
        logs as evidence, not to repeat them.
        """
        del logs
        return ""
