# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable device-array transports.

A *device transport* moves a GPU array's bytes from a producer process to a
consumer process without a host round-trip. The legacy ``json+cuda_ipc``
encoding is the first such transport; this module defines the common interface
they share so further transports (VMM-fd map-and-read, and later cross-host
NCCL/NIXL) slot in behind one negotiation path instead of each bolting a new
encoder, wire format, and release hook onto the runtime.

The interface deliberately mirrors the lifecycle the ``cuda_ipc`` code already
follows, so wrapping it changes no behavior:

* :meth:`DeviceTransport.register` -- encode side: pin the source array and
  return an opaque per-array handle (the analog of ``dump_*_arraydict``).
* :meth:`DeviceTransport.descriptor` -- turn that handle into the wire string
  packed into the JSON ``data.buffer`` field.
* :meth:`DeviceTransport.flush` -- post any pending transfers. A no-op for
  receiver-driven transports like ``cuda_ipc`` (the consumer pulls); the seam
  where a push transport (NCCL) posts its matched sends.
* :meth:`DeviceTransport.receive` -- decode side: materialise the array into a
  fresh, consumer-owned buffer (the analog of ``load_*_arraydict``). Returns the
  framework-agnostic wrapper the consumer adopts.
* :meth:`DeviceTransport.release` -- drop the producer-side pins once the borrow
  is provably done.

:meth:`DeviceTransport.bootstrap` is the one axis genuinely new versus
``cuda_ipc``: transports that need a handshake (a shared communicator, a socket
for fd passing) establish it here; ``cuda_ipc``'s inert handle needs none, so its
bootstrap is a no-op.

This module itself imports no CUDA/driver libraries: a backend's native
machinery loads only when its module is imported (via :func:`get_transport` for a
name not yet registered). Note, though, that the CUDA libraries are still only
*touched* lazily -- importing ``cuda_ipc`` binds ctypes signatures but does not
dlopen libcudart until an encode/decode actually runs -- because the runtime
package's ``__init__`` eagerly imports every submodule (including ``cuda_ipc``),
so merely importing the ``tesseract_core.runtime`` package pulls this backend's
module in regardless of this lazy path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tesseract_core.runtime.array_encoding import ArrayDict

# Reach describes where a transport can move data, so negotiation can reject a
# cross-host request against a same-host-only transport (and vice versa) before
# any handle is minted.
Reach = Literal["same_host", "cross_host", "both"]


@runtime_checkable
class DeviceTransport(Protocol):
    """The contract every device-array transport implements.

    A transport is a small, mostly-stateless object registered under a ``name``
    (the suffix of the ``json+<name>`` output format). The runtime looks one up
    by name and drives the lifecycle below; adding a transport means adding a
    backend, not editing the encode/decode dispatch.
    """

    name: str
    reach: Reach

    def bootstrap(self, role: Literal["producer", "consumer"], peer_offer: Any) -> Any:
        """Establish any shared state a transfer needs, once per pair.

        Returns a session object cached by the caller and passed back to the
        other methods. Receiver-driven transports whose handle is self-contained
        (``cuda_ipc``) return ``None`` and ignore the session everywhere.
        """
        ...

    def register(self, arr: Any, session: Any = None) -> Any:
        """Encode side: pin ``arr`` and return an opaque per-array handle.

        Keeps the source allocation alive until :meth:`release`, exactly as the
        legacy export registry does.
        """
        ...

    def descriptor(self, handle: Any) -> ArrayDict:
        """Turn a handle from :meth:`register` into the JSON array dict.

        The returned dict carries the transport's wire string in
        ``data.buffer`` and its name in ``data.encoding``.
        """
        ...

    def flush(self, session: Any = None) -> None:
        """Post any pending transfers. No-op for pull transports."""
        ...

    def receive(self, val: ArrayDict, session: Any = None) -> Any:
        """Decode side: materialise ``val`` into a fresh consumer-owned buffer.

        Returns the framework-agnostic on-GPU wrapper the consumer adopts
        (``IpcDeviceArray`` for the CUDA transports), unchanged across
        transports so the consumer-facing surface never forks.
        """
        ...

    def release(self, session: Any = None) -> None:
        """Drop producer-side pins once the borrow is provably complete."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Transports register here by name. The lookup is by the ``json+<name>`` format
# suffix, so the encode/decode dispatch and the eventual negotiation endpoint go
# through one table rather than a chain of ``if encoding == ...`` branches.

_TRANSPORTS: dict[str, DeviceTransport] = {}


def register_transport(transport: DeviceTransport) -> DeviceTransport:
    """Register a transport under its ``name``. Returns it, so it can decorate."""
    _TRANSPORTS[transport.name] = transport
    return transport


# Built-in transports and the module whose import registers each one. Importing
# that module is what registers the backend; a name absent from _TRANSPORTS is
# resolved by importing its module here. (In the full runtime the package
# __init__ has usually imported these already; this makes get_transport work
# even when a backend module has not been imported yet.)
_BUILTIN_TRANSPORT_MODULES = {
    "cuda_ipc": "tesseract_core.runtime.cuda_ipc",
}


def get_transport(name: str) -> DeviceTransport:
    """Look up a registered transport by name, importing built-ins on demand."""
    if name not in _TRANSPORTS and name in _BUILTIN_TRANSPORT_MODULES:
        import importlib

        importlib.import_module(_BUILTIN_TRANSPORT_MODULES[name])
    if name not in _TRANSPORTS:
        raise KeyError(
            f"No device transport registered under {name!r} "
            f"(known: {sorted(_TRANSPORTS)})"
        )
    return _TRANSPORTS[name]


def available_transports() -> tuple[str, ...]:
    """Names of transports currently registered in this process.

    This reports what has been *registered*, not what a Tesseract will actually
    accept: registration says the code exists, whereas whether a transport may be
    used is gated separately (e.g. ``json+cuda_ipc`` is only an accepted output
    format when ``enable_experimental_cuda_ipc`` is set; see
    :func:`tesseract_core.runtime.file_interactions.available_formats`). A caller
    deciding what to offer a client -- a transport-negotiation endpoint, say --
    must apply that gating itself and not treat this list as the enabled set.

    Note also that a built-in transport registers on first
    :func:`get_transport`, so a name can be usable before it appears here.
    """
    return tuple(sorted(_TRANSPORTS))
