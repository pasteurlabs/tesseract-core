# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import wraps
from types import ModuleType
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel

from .config import get_config
from .core import create_endpoints
from .file_interactions import (
    available_formats,
    join_paths,
    output_to_bytes,
    parse_accept_header,
)
from .mpa import start_run
from .profiler import Profiler

# Endpoints that should use GET instead of POST
GET_ENDPOINTS = {"health"}


def negotiate_output_format(accept: str | None) -> str:
    """Resolve the Accept header to an output format the runtime currently offers.

    Media ranges are tried by descending q-value, so a client listing several
    types gets the first one this runtime can actually produce.
    """
    if not accept:
        return get_config().output_format

    def quality(media_range: str) -> float:
        _, _, params = media_range.partition(";")
        for param in params.split(";"):
            key, sep, value = param.partition("=")
            if sep and key.strip() == "q":
                try:
                    return float(value.strip())
                except ValueError:
                    return 0.0
        return 1.0

    allowed = available_formats()
    ranges = [r.strip() for r in accept.split(",") if r.strip()]
    # Sorting is stable, so equal-quality ranges keep the client's ordering.
    for media_range in sorted(ranges, key=quality, reverse=True):
        media_type = media_range.partition(";")[0].strip().lower()
        if media_type in ("*/*", "application/*"):
            return get_config().output_format
        output_format = media_type.rpartition("/")[2]
        if output_format in allowed:
            return output_format

    raise HTTPException(
        status_code=406,
        detail={
            "message": f"Cannot produce any format accepted by {accept!r}",
            "available_formats": list(allowed),
        },
    )


def create_response(
    model: BaseModel,
    output_format: str,
    accept: str | None,
    base_dir: str | None,
    binref_dir: str | None,
) -> Response:
    """Create a response in the given (already negotiated) output format.

    ``output_format`` is the host-array output format the caller already
    negotiated from the ``Accept`` header. How GPU arrays leave the process is a
    separate axis: it may ride the header as a ``gpu_transport`` media-type
    parameter (``application/json+base64; gpu_transport=cuda_ipc``), and when the
    header omits it the served Tesseract's ``gpu_transport`` config applies. So a
    raw HTTP client can opt in (or out) per request on top of the served default.
    """
    config = get_config()

    if not accept or accept == "*/*":
        gpu_transport = config.gpu_transport
    else:
        _, requested_transport = parse_accept_header(accept)
        # Header wins when it names a transport; otherwise fall back to config.
        gpu_transport = (
            requested_transport
            if requested_transport is not None
            else config.gpu_transport
        )

    if base_dir is None:
        base_dir = config.output_path

    content = output_to_bytes(
        model,
        output_format,
        base_dir=base_dir,
        binref_dir=binref_dir,
        gpu_transport=gpu_transport,
    )
    # Name the format actually produced, which is not necessarily what the
    # client asked for: an Accept header may hold several media ranges.
    return Response(
        status_code=200, content=content, media_type=f"application/{output_format}"
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """App-scoped setup/teardown for the served Tesseract.

    When the configured GPU transport is ``cuda_vmm``, bootstrap its fd-passing
    server here so the Unix socket is created at startup and closed at shutdown
    -- owned by the app rather than leaked as a lazily-started daemon thread.
    Bootstrapping also installs it as the process fallback the encode path uses,
    so exports and this server share one instance. No CUDA machinery is touched
    for the other transports (``cuda_ipc`` needs no handshake) or when GPU
    transport is off.
    """
    config = get_config()
    vmm = None
    if config.gpu_transport == "cuda_vmm":
        from tesseract_core.runtime.device_transport import get_transport

        vmm = get_transport("cuda_vmm")
        vmm.bootstrap("producer")
    try:
        yield
    finally:
        if vmm is not None:
            vmm.shutdown()


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
        lifespan=_lifespan,
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
        async def wrapper(*args: Any, accept: str, run_id: str | None, **kwargs: Any):
            config = get_config()
            output_format = negotiate_output_format(accept)

            # Release device buffers exported by the previous request's GPU
            # transport. Releasing at the start of each request keeps every
            # export alive long enough for a serial client to copy it out of the
            # response before it is reclaimed. See cuda_ipc for the assumptions
            # this relies on. Gated on a configured GPU transport so the default
            # path never imports the CUDA machinery.
            if config.gpu_transport != "none":
                from tesseract_core.runtime.device_transport import get_transport

                get_transport(config.gpu_transport).release()

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
            return create_response(
                result,
                output_format,
                accept=accept,
                base_dir=output_path,
                binref_dir=rundir_name,
            )

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
            # Other header parameters common to computational endpoints
            # could be defined and appended here as well.
            new_params = original_sig.parameters.copy()
            new_params.update({"accept": accept, "run_id": run_id})
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

    generate_openapi = app.openapi

    def openapi_with_output_formats() -> dict:
        from .file_interactions import available_formats

        schema = generate_openapi()
        schema["x-supported-output-formats"] = list(available_formats())
        return schema

    app.openapi = openapi_with_output_formats
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
