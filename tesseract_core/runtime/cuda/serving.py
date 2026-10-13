# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the REST API needs to send GPU arrays over a device transport.

A response's device exports must stay alive until the client has copied them out
of the response, which only the client knows (see "Keeping exports alive" in
:mod:`tesseract_core.runtime.cuda.ipc`). The server names each response's exports
in the :data:`EXPORTS_HEADER` response header, and the client names the ones it
is done with in the :data:`EXPORTS_DONE_HEADER` of a later request. A client that
sends that header at all, even empty, acknowledges its responses this way.

Clients that do not (older SDKs, raw HTTP) give no sign of being done other than
their next request. Each of their requests therefore releases the exports of
all of them, which is only safe for a single such client making one request at a
time. The SDK mirrors the header names.
"""

import threading
import time
import uuid
from collections.abc import Callable
from typing import Annotated, Any, NamedTuple

from fastapi import Body, FastAPI, Response

from tesseract_core.runtime.device_transport import get_transport
from tesseract_core.runtime.file_interactions import available_gpu_transports

EXPORTS_HEADER = "Tesseract-Exports"
EXPORTS_DONE_HEADER = "Tesseract-Exports-Done"

# How long a response's exports are kept for an acknowledging client that never
# names them (because it crashed, say). Far longer than any live client takes
# to decode a response it has received.
EXPORTS_TIMEOUT_S = 300.0


class _PendingExports(NamedTuple):
    """A response's exports, kept until its client is done with them."""

    transport: Any
    """The DeviceTransport that exported them."""

    session: Any
    """The transport's session holding them (see DeviceTransport.new_exports)."""

    kept_at: float
    acknowledged: bool
    """Whether the client names the responses it is done with."""


_PENDING_EXPORTS: dict[str, _PendingExports] = {}
_PENDING_EXPORTS_LOCK = threading.Lock()


def _keep_exports(transport: Any, session: Any, acknowledged: bool) -> str | None:
    """Keep a response's exports and return the id to send with it, if it has any."""
    if not session:
        return None
    export_id = uuid.uuid4().hex
    with _PENDING_EXPORTS_LOCK:
        _PENDING_EXPORTS[export_id] = _PendingExports(
            transport, session, time.monotonic(), acknowledged
        )
    return export_id


def _release_exports(done: list[str], unacknowledged: bool) -> None:
    """Release the exports named in ``done`` and those kept too long.

    With ``unacknowledged``, also release those of every client that does not
    acknowledge its responses.
    """
    now = time.monotonic()
    released = []
    with _PENDING_EXPORTS_LOCK:
        for export_id in done:
            pending = _PENDING_EXPORTS.pop(export_id, None)
            if pending is not None:
                released.append(pending)
        for export_id, pending in list(_PENDING_EXPORTS.items()):
            if (unacknowledged and not pending.acknowledged) or (
                now - pending.kept_at > EXPORTS_TIMEOUT_S
            ):
                del _PENDING_EXPORTS[export_id]
                released.append(pending)
    for pending in released:
        pending.transport.release(pending.session)


class ResponseExports:
    """The device exports of one response, kept until its client is done with them.

    Creating one also releases earlier exports that are due (see
    :func:`_release_exports`).
    """

    def __init__(self, gpu_transport: str, acknowledged: bool) -> None:
        _release_exports([], unacknowledged=not acknowledged)
        self.acknowledged = acknowledged
        self.transport = None
        self.session = None
        if gpu_transport != "none":
            self.transport = get_transport(gpu_transport)
            self.session = self.transport.new_exports()

    def respond(self, make_response: Callable[[Any], Response]) -> Response:
        """Build the response with ``make_response(session)`` and keep its exports."""
        try:
            response = make_response(self.session)
        except BaseException:
            # No client will read the exports of a response that failed.
            if self.transport is not None:
                self.transport.release(self.session)
            raise
        export_id = _keep_exports(self.transport, self.session, self.acknowledged)
        if export_id is not None:
            response.headers[EXPORTS_HEADER] = export_id
        return response


class _ReleaseDoneExports:
    """ASGI middleware releasing the exports a request's EXPORTS_DONE_HEADER names.

    Runs for every route, so a client can name them on any request, e.g. a
    health check when it closes.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self._header = EXPORTS_DONE_HEADER.lower().encode()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            for name, value in scope["headers"]:
                if name == self._header:
                    if value:
                        _release_exports(
                            value.decode("latin-1").split(","),
                            unacknowledged=False,
                        )
                    break
        await self.app(scope, receive, send)


def check_gpu_transport(request: dict) -> dict:
    """Answer a client checking whether a GPU transport works between it and this server.

    Returns ``{"ok": True, "array": ...}`` with the transport's reply, or
    ``{"ok": False, "reason": ...}`` if this side of the check failed. See
    :mod:`tesseract_core.runtime.cuda.check`.
    """
    gpu_transport = request.get("gpu_transport")
    if gpu_transport == "none" or gpu_transport not in available_gpu_transports():
        return {
            "ok": False,
            "reason": f"gpu_transport={gpu_transport!r} is not enabled on this "
            f"Tesseract (available: {list(available_gpu_transports())})",
        }
    try:
        from tesseract_core.runtime.cuda.check import answer_transport_check

        return {"ok": True, "array": answer_transport_check(request)}
    except Exception as exc:  # noqa: BLE001 - reported to the client, which decides
        return {
            "ok": False,
            "reason": f"the Tesseract could not exchange GPU memory with the "
            f"client: {type(exc).__name__}: {exc}",
        }


def add_gpu_transport_handling(app: FastAPI) -> None:
    """Add the export-release middleware and the transport check route to ``app``."""
    app.add_middleware(_ReleaseDoneExports)

    # Not a Tesseract endpoint, so it stays out of the schema clients read
    # endpoints from. Async like the Tesseract endpoints, so it runs on the event
    # loop between requests rather than alongside one.
    async def check_gpu_transport_route(
        request: Annotated[dict[str, Any], Body()],
    ) -> dict:
        return check_gpu_transport(request)

    app.add_api_route(
        "/check_gpu_transport",
        check_gpu_transport_route,
        methods=["POST"],
        include_in_schema=False,
    )
