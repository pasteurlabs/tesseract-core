# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import inspect
import threading
import time
import uuid
from collections.abc import Callable
from functools import wraps
from types import ModuleType
from typing import Annotated, Any, NamedTuple

import uvicorn
from fastapi import Body, FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel

from .config import get_config
from .core import create_endpoints
from .file_interactions import (
    available_compressions,
    available_formats,
    available_gpu_transports,
    join_paths,
    output_to_bytes,
)
from .mpa import start_run
from .profiler import Profiler

# Endpoints that should use GET instead of POST
GET_ENDPOINTS = {"health"}


class NegotiatedEncoding(NamedTuple):
    """How a response encodes its arrays, as resolved from the Accept header."""

    output_format: str
    gpu_transport: str
    compression: str | None


def parse_media_range(media_range: str) -> tuple[str, dict[str, str]]:
    """Split one media range of an ``Accept`` header into its type and parameters.

    For example, ``application/json+base64; compression="lz4"`` parses to
    ``("application/json+base64", {"compression": "lz4"})``. The media type and
    parameter names are lowercased and quoted values are unwrapped. This does no
    validation of the values.
    """
    media_type, *param_strs = media_range.split(";")
    params = {}
    for param in param_strs:
        key, sep, value = param.partition("=")
        if sep:
            params[key.strip().lower()] = value.strip().strip('"')
    return media_type.strip().lower(), params


def negotiate_encoding(accept: str | None) -> NegotiatedEncoding:
    """Resolve the Accept header to an encoding the runtime currently offers.

    Media ranges are tried by descending q-value, and the first one whose format,
    ``gpu_transport`` and ``compression`` this runtime can all produce wins.
    Raises a 406 error if there is none.

    Values a range leaves out fall back to the configured ``output_format`` and
    ``compression``. The GPU transport falls back to ``none`` regardless of the
    config, which only controls which transports a request may ask for.
    """
    config = get_config()
    default_compression = config.compression or "none"
    if not accept:
        return NegotiatedEncoding(config.output_format, "none", config.compression)

    def quality(parsed: tuple[str, dict[str, str]]) -> float:
        try:
            return float(parsed[1].get("q", 1.0))
        except ValueError:
            return 0.0

    formats = available_formats()
    transports = available_gpu_transports()
    compressions = available_compressions()
    ranges = [parse_media_range(r) for r in accept.split(",") if r.strip()]
    # Sorting is stable, so equal-quality ranges keep the client's ordering.
    for media_type, params in sorted(ranges, key=quality, reverse=True):
        if media_type in ("*/*", "application/*"):
            output_format = config.output_format
        else:
            output_format = media_type.rpartition("/")[2]
        gpu_transport = params.get("gpu_transport", "none")
        compression = params.get("compression", default_compression)
        if (
            output_format in formats
            and gpu_transport in transports
            and compression in compressions
        ):
            return NegotiatedEncoding(
                output_format,
                gpu_transport,
                None if compression == "none" else compression,
            )

    raise HTTPException(
        status_code=406,
        detail={
            "message": f"Cannot produce any encoding accepted by {accept!r}",
            "available_formats": list(formats),
            "available_gpu_transports": list(transports),
            "available_compressions": list(compressions),
        },
    )


def create_response(
    model: BaseModel,
    encoding: NegotiatedEncoding,
    base_dir: str | None,
    binref_dir: str | None,
    device_exports: Any = None,
) -> Response:
    """Create a response in the given (already negotiated) encoding."""
    if base_dir is None:
        base_dir = get_config().output_path

    content = output_to_bytes(
        model,
        encoding.output_format,
        base_dir=base_dir,
        binref_dir=binref_dir,
        compression=encoding.compression,
        gpu_transport=encoding.gpu_transport,
        device_exports=device_exports,
    )
    # Name the format actually produced, which is not necessarily what the
    # client asked for: an Accept header may hold several media ranges.
    return Response(
        status_code=200,
        content=content,
        media_type=f"application/{encoding.output_format}",
    )


# A response's device exports must stay alive until the client has copied them
# out of the response, which only the client knows (see "Keeping exports alive"
# in tesseract_core.runtime.cuda.ipc). The server names each response's exports
# in the EXPORTS_HEADER response header, and the client names the ones it is done
# with in the EXPORTS_DONE_HEADER of a later request. A client that sends that
# header at all, even empty, acknowledges its responses this way. For clients
# that do not (older SDKs, raw HTTP), a request is the only sign that they are
# done, so each of their requests releases the exports of all of them, which is
# safe only for a single such client making one request at a time. Header names
# are mirrored in the SDK.
EXPORTS_HEADER = "Tesseract-Exports"
EXPORTS_DONE_HEADER = "Tesseract-Exports-Done"

# How long a response's exports are kept for an acknowledging client that never
# names them (because it crashed, say). Far longer than any live client takes
# to decode a response it has received.
EXPORTS_TIMEOUT_S = 300.0

# Kept exports by id: the transport's session holding them, when they were kept,
# and whether the client acknowledges.
_PENDING_EXPORTS: collections.OrderedDict[str, tuple[Any, float, bool]] = (
    collections.OrderedDict()
)
_PENDING_EXPORTS_LOCK = threading.Lock()


def _keep_exports(exports: Any, acknowledged: bool) -> str | None:
    """Keep a response's exports and return the id to send with it, if it has any."""
    if not exports:
        return None
    export_id = uuid.uuid4().hex
    with _PENDING_EXPORTS_LOCK:
        _PENDING_EXPORTS[export_id] = (exports, time.monotonic(), acknowledged)
    return export_id


def _release_exports(transport: Any, done: list[str], unacknowledged: bool) -> None:
    """Release the exports named in ``done`` and those kept too long.

    With ``unacknowledged``, also release those of every client that does not
    acknowledge its responses.
    """
    now = time.monotonic()
    released = []
    with _PENDING_EXPORTS_LOCK:
        for export_id in done:
            entry = _PENDING_EXPORTS.pop(export_id, None)
            if entry is not None:
                released.append(entry[0])
        for export_id, (exports, kept_at, acknowledged) in list(
            _PENDING_EXPORTS.items()
        ):
            if (unacknowledged and not acknowledged) or (
                now - kept_at > EXPORTS_TIMEOUT_S
            ):
                del _PENDING_EXPORTS[export_id]
                released.append(exports)
    for exports in released:
        transport.release(exports)


class _ReleaseDoneExports:
    """ASGI middleware releasing the exports a request's EXPORTS_DONE_HEADER names.

    Runs for every route, so a client can name them on any request, e.g. a
    health check when it closes.
    """

    def __init__(self, app: Any, transport_name: str) -> None:
        self.app = app
        self.transport_name = transport_name
        self._header = EXPORTS_DONE_HEADER.lower().encode()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            for name, value in scope["headers"]:
                if name == self._header:
                    if value:
                        from tesseract_core.runtime.device_transport import (
                            get_transport,
                        )

                        _release_exports(
                            get_transport(self.transport_name),
                            value.decode("latin-1").split(","),
                            unacknowledged=False,
                        )
                    break
        await self.app(scope, receive, send)


def check_gpu_transport(request: dict) -> dict:
    """Answer a client checking whether a GPU transport works between it and this server.

    Returns ``{"ok": True, "array": ...}`` with the transport's reply, or
    ``{"ok": False, "reason": ...}`` if this side of the check failed. See
    :func:`tesseract_core.runtime.cuda.ipc.answer_transport_check`.
    """
    gpu_transport = request.get("gpu_transport")
    if gpu_transport == "none" or gpu_transport not in available_gpu_transports():
        return {
            "ok": False,
            "reason": f"gpu_transport={gpu_transport!r} is not enabled on this "
            f"Tesseract (available: {list(available_gpu_transports())})",
        }
    try:
        from tesseract_core.runtime.cuda.ipc import answer_transport_check

        return {"ok": True, "array": answer_transport_check(request)}
    except Exception as exc:  # noqa: BLE001 - reported to the client, which decides
        return {
            "ok": False,
            "reason": f"the Tesseract could not open GPU memory exported by the "
            f"client: {type(exc).__name__}: {exc}",
        }


def create_rest_api(api_module: ModuleType) -> FastAPI:
    """Create the Tesseract REST API."""
    config = get_config()
    app = FastAPI(
        title=config.name,
        version=config.version,
        description=config.description.replace("\\n", "\n"),
        docs_url=None,
        redoc_url="/docs",
        debug=config.debug,
    )
    tesseract_endpoints = create_endpoints(api_module)

    def wrap_endpoint(endpoint_func: Callable):
        endpoints_to_wrap = [
            "apply",
            "jacobian",
            "jacobian_vector_product",
            "vector_jacobian_product",
        ]

        @wraps(endpoint_func)
        async def wrapper(
            *args: Any,
            accept: str,
            run_id: str | None,
            exports_done: str | None,
            **kwargs: Any,
        ):
            config = get_config()
            encoding = negotiate_encoding(accept)

            # Collect this response's device exports so they can be kept until
            # the client is done with them, and release earlier ones (see
            # _release_exports). Gated on a configured GPU transport so the
            # default path never imports the CUDA machinery.
            exports = None
            acknowledged = exports_done is not None
            if config.gpu_transport != "none":
                from tesseract_core.runtime.device_transport import get_transport

                transport = get_transport(config.gpu_transport)
                _release_exports(transport, [], unacknowledged=not acknowledged)
                exports = transport.new_exports()

            if run_id is None:
                run_id = str(uuid.uuid4())
            output_path = config.output_path
            rundir_name = f"run_{run_id}"
            rundir = join_paths(output_path, rundir_name)
            profiler = Profiler()
            with start_run(base_dir=rundir):
                with profiler:
                    result = endpoint_func(*args, **kwargs)

                # Print profiling stats inside start_run context
                # so they go through stdio redirection to the log file
                profiler.print_stats()
            try:
                response = create_response(
                    result,
                    encoding,
                    base_dir=output_path,
                    binref_dir=rundir_name,
                    device_exports=exports,
                )
            except BaseException:
                # No client will read the exports of a response that failed.
                if exports is not None:
                    transport.release(exports)
                raise
            export_id = _keep_exports(exports, acknowledged)
            if export_id is not None:
                response.headers[EXPORTS_HEADER] = export_id
            return response

        if endpoint_func.__name__ not in endpoints_to_wrap:
            return endpoint_func
        else:
            # wrapper's signature will be the same as endpoint
            # func's signature. We do however need to change this
            # in order to add a Header parameter that FastAPI
            # will understand.
            original_sig = inspect.signature(endpoint_func)
            accept = inspect.Parameter(
                "accept",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=Header(default=None),
                annotation=str | None,
            )
            run_id = inspect.Parameter(
                "run_id",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=Annotated[str | None, Query(include_in_schema=False)],
            )
            exports_done = inspect.Parameter(
                "exports_done",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=Header(
                    default=None, alias=EXPORTS_DONE_HEADER, include_in_schema=False
                ),
                annotation=str | None,
            )
            # Other header parameters common to computational endpoints
            # could be defined and appended here as well.
            new_params = original_sig.parameters.copy()
            new_params.update(
                {"accept": accept, "run_id": run_id, "exports_done": exports_done}
            )
            # Update the signature of the wrapper
            new_sig = original_sig.replace(parameters=list(new_params.values()))
            wrapper.__signature__ = new_sig
            return wrapper

    for endpoint_func in tesseract_endpoints:
        endpoint_name = endpoint_func.__name__

        # Skip test endpoint unless in debug mode
        if endpoint_name == "test" and not config.debug:
            continue

        wrapped_endpoint = wrap_endpoint(endpoint_func)
        http_methods = ["GET"] if endpoint_name in GET_ENDPOINTS else ["POST"]
        app.add_api_route(f"/{endpoint_name}", wrapped_endpoint, methods=http_methods)

    if config.gpu_transport != "none":
        app.add_middleware(_ReleaseDoneExports, transport_name=config.gpu_transport)

        # Not a Tesseract endpoint, so it stays out of the schema clients read
        # endpoints from. Async like the endpoints above, so it runs on the event
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

    generate_openapi = app.openapi

    def openapi_with_encodings() -> dict:
        # Advertise what the Accept header can negotiate, so clients can discover
        # it without knowing how this server was configured.
        schema = generate_openapi()
        schema["x-supported-output-formats"] = list(available_formats())
        schema["x-supported-gpu-transports"] = list(available_gpu_transports())
        schema["x-supported-compressions"] = list(available_compressions())
        return schema

    app.openapi = openapi_with_encodings
    return app


def serve(host: str, port: int, num_workers: int) -> None:
    """Start the REST API."""
    uvicorn.run(
        "tesseract_core.runtime.app_http:app",
        host=host,
        port=port,
        workers=num_workers,
        # Increase from uvicorn's default of 5s to account for the fact that Tesseract endpoints
        # tend to run longer than your average HTTP request. A higher value means that the server
        # will wait longer before closing idle connections, so we can avoid the overhead of reconnecting.
        timeout_keep_alive=60,
    )
