# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable device-array transports.

A *device transport* moves a GPU array's bytes from a producer process to a
consumer process without a host round-trip. ``json+cuda_ipc`` is the first such
transport; this module defines the common interface they share so further
transports slot in behind one dispatch path instead of each bolting a new
encoder, wire format, and release hook onto the runtime.

The interface mirrors the lifecycle the ``cuda_ipc`` code follows:

* :meth:`DeviceTransport.register` -- encode side: pin the source array and
  return an opaque per-array handle.
* :meth:`DeviceTransport.descriptor` -- turn that handle into the array dict
  whose ``data.buffer`` field carries the wire string.
* :meth:`DeviceTransport.flush` -- post any pending transfers. A no-op for
  receiver-driven transports like ``cuda_ipc`` (the consumer pulls); the seam
  where a push transport posts its matched sends.
* :meth:`DeviceTransport.receive` -- decode side: materialise the array into a
  fresh, consumer-owned buffer.
* :meth:`DeviceTransport.bootstrap` -- establish any shared state a handshake
  transport needs before transferring. A no-op for ``cuda_ipc``.
* :meth:`DeviceTransport.release` -- drop the producer-side pins once the borrow
  is done.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:  # pragma: no cover
    from tesseract_core.runtime.array_encoding import ArrayDict

# Reach describes where a transport can move data, so negotiation can reject a
# cross-host request against a same-host-only transport (and vice versa) before
# any handle is minted.
Reach = Literal["same_host", "cross_host", "both"]


class DeviceTransport(abc.ABC):
    """The contract every device-array transport implements.

    A transport is a small, mostly-stateless object registered under a ``name``
    (the suffix of the ``json+<name>`` output format). The runtime looks one up
    by name and drives the lifecycle below; adding a transport means adding a
    backend, not editing the encode/decode dispatch.
    """

    name: ClassVar[str]
    reach: ClassVar[Reach]

    @abc.abstractmethod
    def bootstrap(self, role: Literal["producer", "consumer"], peer_offer: Any) -> Any:
        """Establish any shared state a transfer needs, once per pair.

        Returns a session object cached by the caller and passed back to the
        other methods. Receiver-driven transports whose handle is self-contained
        (``cuda_ipc``) return ``None`` and ignore the session everywhere.
        """

    @abc.abstractmethod
    def register(self, arr: Any, session: Any = None) -> Any:
        """Encode side: pin ``arr`` and return an opaque per-array handle.

        Keeps the source allocation alive until :meth:`release`, exactly as the
        cuda_ipc export registry does.
        """

    @abc.abstractmethod
    def descriptor(self, handle: Any) -> ArrayDict:
        """Turn a handle from :meth:`register` into the JSON array dict.

        The returned dict carries the transport's wire string in
        ``data.buffer`` and its name in ``data.encoding``.
        """

    @abc.abstractmethod
    def flush(self, session: Any = None) -> None:
        """Post any pending transfers. No-op for pull transports."""

    @abc.abstractmethod
    def receive(self, val: ArrayDict, session: Any = None) -> Any:
        """Decode side: materialise ``val`` into a fresh consumer-owned buffer.

        Returns the framework-agnostic on-GPU wrapper the consumer adopts
        (``IpcDeviceArray`` for the CUDA transports), unchanged across
        transports so the consumer-facing surface never forks.
        """

    @abc.abstractmethod
    def release(self, session: Any = None) -> None:
        """Drop producer-side pins once the borrow is provably complete."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Transports register here by name, keyed by the ``json+<name>`` format suffix,
# so the encode/decode dispatch goes through one table rather than a chain of
# ``if encoding == ...`` branches. Built-in transports are registered by the
# runtime package ``__init__``.

_TRANSPORTS: dict[str, DeviceTransport] = {}


def register_transport(transport: DeviceTransport) -> DeviceTransport:
    """Register a transport under its ``name``. Returns it, so it can decorate."""
    _TRANSPORTS[transport.name] = transport
    return transport


def get_transport(name: str) -> DeviceTransport:
    """Look up a registered transport by name."""
    if name not in _TRANSPORTS:
        raise KeyError(
            f"No device transport registered under {name!r} "
            f"(known: {sorted(_TRANSPORTS)})"
        )
    return _TRANSPORTS[name]


def available_transports() -> tuple[str, ...]:
    """Names of transports currently registered in this process.

    This reports what has been *registered*, not what a Tesseract will actually
    accept: whether a transport may be used is gated separately (e.g.
    ``json+cuda_ipc`` is only an accepted output format when
    ``enable_experimental_cuda_ipc`` is set; see
    :func:`tesseract_core.runtime.file_interactions.available_formats`). A caller
    deciding what to offer a client must apply that gating itself and not treat
    this list as the enabled set.
    """
    return tuple(sorted(_TRANSPORTS))
