# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checking whether a GPU transport works between two processes.

Whether a transport works between two processes depends on things neither can
see from the other side, such as whether they share a host and a GPU, how a
container runtime isolates them, or whether one can reach the other over the
network. A client therefore finds out by trying, once per server. It exports
random bytes through the transport, the server reads and checks them, then
exports the bytes reversed for the client to read and check in turn. Comparing
contents also catches a handle that opens onto the wrong memory.

The check uses only the
:class:`~tesseract_core.runtime.device_transport.DeviceTransport` interface, so
it works for any registered transport whose ``receive`` returns an
:class:`~tesseract_core.runtime.cuda.ipc.IpcDeviceArray`, as the CUDA
transports do.

The check can run while other calls are in flight (Tesseract-JAX may lower one
function while a compiled one runs), so each side keeps its check exports in
sessions of their own. The server keeps the exports of its reply until the
client names them as done, as it does for any response (see
:mod:`tesseract_core.runtime.cuda.serving`).
"""

import functools
import os
from collections.abc import Callable
from typing import Any

import pybase64

from tesseract_core.runtime.array_encoding import ArrayDict
from tesseract_core.runtime.cuda import api as cuda_api
from tesseract_core.runtime.cuda.ipc import _device_array_from_bytes
from tesseract_core.runtime.device_transport import DeviceTransport, get_transport

__all__ = [
    "answer_transport_check",
    "finish_transport_check",
    "start_transport_check",
]

# Size of the array each side exports during a check.
_CHECK_NBYTES = 256


def _export(
    transport: DeviceTransport, data: bytes, device: int, session: Any
) -> ArrayDict:
    """Export ``data`` from ``device`` into ``session``, returning its descriptor."""
    arr = _device_array_from_bytes(data, device)
    return transport.descriptor(transport.register(arr, session))


def _read(transport: DeviceTransport, payload: Any) -> tuple[bytes, int]:
    """Read a check array the other side exported, returning its bytes and device."""
    if not isinstance(payload, dict) or payload.get("dtype") != "uint8":
        raise ValueError("expected a uint8 array")
    if list(payload.get("shape", ())) != [_CHECK_NBYTES]:
        raise ValueError(f"expected an array of {_CHECK_NBYTES} bytes")
    if payload.get("data", {}).get("encoding") != transport.name:
        raise ValueError(f"expected a {transport.name} array")
    arr = transport.receive(payload)
    return arr.copy_to_host().tobytes(), arr.device


def start_transport_check(
    gpu_transport: str,
) -> tuple[dict, bytes, Callable[[], None]]:
    """Client side: export random bytes from the active CUDA device.

    Returns the request body for the server, the bytes its reply must hold, and
    a function to call once the server has answered, which releases the export.

    Raises:
        KeyError: if no transport is registered under ``gpu_transport``.
    """
    transport = get_transport(gpu_transport)
    data = os.urandom(_CHECK_NBYTES)
    session = transport.new_exports()
    try:
        array = _export(transport, data, cuda_api.get_device(), session)
    except BaseException:
        transport.release(session)
        raise
    request = {
        "gpu_transport": gpu_transport,
        "array": array,
        "expected": pybase64.b64encode_as_string(data),
    }
    return request, data[::-1], functools.partial(transport.release, session)


def answer_transport_check(request: dict, session: Any) -> ArrayDict:
    """Server side: check the client's bytes, then export them reversed into ``session``.

    Raises if the client's array cannot be read or holds the wrong bytes.
    """
    transport = get_transport(request["gpu_transport"])
    data, device = _read(transport, request.get("array"))
    if data != pybase64.b64decode(request.get("expected", ""), validate=True):
        raise ValueError(
            "the bytes read from the client's array differ from the ones it wrote"
        )
    return _export(transport, data[::-1], device, session)


def finish_transport_check(gpu_transport: str, reply: Any, expected: bytes) -> None:
    """Client side: read the server's reply and check it holds ``expected``."""
    data, _ = _read(get_transport(gpu_transport), reply)
    if data != expected:
        raise ValueError(
            "the bytes read from the Tesseract's array differ from the ones it wrote"
        )
