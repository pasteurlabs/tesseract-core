# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the transport-agnostic DeviceTransport interface and registry.

These exercise the shared machinery -- the name-keyed registry, lookup, and the
abstract base class -- with a stub backend, so they hold for any transport and
do not depend on CUDA. The cuda_ipc backend's own conformance lives in
``test_cuda_ipc_cpu.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from tesseract_core.runtime.array_encoding import ArrayDict
from tesseract_core.runtime.device_transport import (
    DeviceTransport,
    Reach,
)


class _StubTransport(DeviceTransport):
    """Minimal transport implementing the full lifecycle as no-ops."""

    name = "stub_test_transport"
    reach: Reach = "both"

    def bootstrap(self, role: Any, peer_offer: Any) -> None:
        return None

    def register(self, arr: Any, session: Any = None) -> Any:
        return arr

    def descriptor(self, handle: Any) -> ArrayDict:
        return handle

    def flush(self, session: Any = None) -> None:
        return None

    def receive(self, val: ArrayDict, session: Any = None) -> Any:
        return val

    def release(self, session: Any = None) -> None:
        return None


def test_get_transport_rejects_unknown():
    from tesseract_core.runtime.device_transport import get_transport

    with pytest.raises(KeyError, match="No device transport registered"):
        get_transport("does_not_exist")


def test_register_transport_returns_and_registers():
    """register_transport adds a backend by name and returns it (usable as a decorator)."""
    from tesseract_core.runtime import device_transport
    from tesseract_core.runtime.device_transport import (
        available_transports,
        get_transport,
        register_transport,
    )

    stub = _StubTransport()
    try:
        assert register_transport(stub) is stub
        assert get_transport("stub_test_transport") is stub
        assert "stub_test_transport" in available_transports()
    finally:
        # Keep the process-global registry clean for other tests.
        device_transport._TRANSPORTS.pop("stub_test_transport", None)


def test_available_transports_returns_sorted_registered_names():
    """available_transports reports registered backends in a stable sorted order."""
    from tesseract_core.runtime import device_transport
    from tesseract_core.runtime.device_transport import (
        available_transports,
        register_transport,
    )

    class _ZTransport(_StubTransport):
        name = "zzz_test_transport"

    class _ATransport(_StubTransport):
        name = "aaa_test_transport"

    try:
        register_transport(_ZTransport())
        register_transport(_ATransport())
        transports = available_transports()
        assert "aaa_test_transport" in transports
        assert "zzz_test_transport" in transports
        assert list(transports) == sorted(transports)
    finally:
        device_transport._TRANSPORTS.pop("zzz_test_transport", None)
        device_transport._TRANSPORTS.pop("aaa_test_transport", None)


def test_incomplete_transport_cannot_instantiate():
    """The ABC rejects a backend that leaves a lifecycle method unimplemented."""

    class _MissingRelease(DeviceTransport):
        name = "incomplete_test_transport"
        reach: Reach = "same_host"

        def bootstrap(self, role, peer_offer):
            return None

        def register(self, arr, session=None):
            return arr

        def descriptor(self, handle):
            return handle

        def flush(self, session=None):
            return None

        def receive(self, val, session=None):
            return val

        # release is intentionally missing.

    with pytest.raises(TypeError, match="abstract"):
        _MissingRelease()
