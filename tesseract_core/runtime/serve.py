# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import uuid
from collections.abc import Callable
from functools import wraps
from types import ModuleType
from typing import Annotated, Any, NamedTuple

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Response
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
    )
    # Name the format actually produced, which is not necessarily what the
    # client asked for: an Accept header may hold several media ranges.
    return Response(
        status_code=200,
        content=content,
        media_type=f"application/{encoding.output_format}",
    )


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
        async def wrapper(*args: Any, accept: str, run_id: str | None, **kwargs: Any):
            config = get_config()
            encoding = negotiate_encoding(accept)

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
                encoding,
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
