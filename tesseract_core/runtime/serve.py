# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import time
import uuid
from collections.abc import Callable
from functools import wraps
from types import ModuleType
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, Header, Query, Response
from pydantic import BaseModel

from .config import get_config
from .core import create_endpoints
from .file_interactions import SUPPORTED_FORMATS, join_paths, output_to_bytes
from .mpa import start_run
from .profiler import Profiler


class BodyTimingMiddleware:
    """ASGI middleware that measures body receive time vs processing time.

    Emits headers:
      - Server-Timing: total;dur=<ms>
      - X-Server-Body-Receive-Ms: time to read the full request body
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Wrap receive to measure how long body reads take
        body_receive_ns = 0
        body_done = False

        async def timed_receive():
            nonlocal body_receive_ns, body_done
            if body_done:
                return await receive()
            t0 = time.perf_counter_ns()
            msg = await receive()
            t1 = time.perf_counter_ns()
            body_receive_ns += t1 - t0
            if msg.get("type") == "http.request" and not msg.get("more_body", False):
                body_done = True
            return msg

        t_start = time.perf_counter_ns()

        # Wrap send to inject timing headers into the response
        async def timed_send(message):
            if message["type"] == "http.response.start":
                elapsed_ms = (time.perf_counter_ns() - t_start) / 1e6
                body_ms = body_receive_ns / 1e6
                headers = list(message.get("headers", []))
                headers.append(
                    (b"server-timing", f"total;dur={elapsed_ms:.3f}".encode())
                )
                headers.append((b"x-server-body-receive-ms", f"{body_ms:.3f}".encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, timed_receive, timed_send)


# Endpoints that should use GET instead of POST
GET_ENDPOINTS = {"health"}


def create_response(
    model: BaseModel, accept: str, base_dir: str | None, binref_dir: str | None
) -> Response:
    """Create a response of the format specified by the Accept header."""
    config = get_config()

    if accept is None or accept == "*/*":
        output_format = config.output_format
    else:
        output_format: SUPPORTED_FORMATS = accept.split("/")[-1]

    if base_dir is None:
        base_dir = config.output_path

    content = output_to_bytes(
        model, output_format, base_dir=base_dir, binref_dir=binref_dir
    )
    return Response(status_code=200, content=content, media_type=accept)


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
    app.add_middleware(BodyTimingMiddleware)
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
            if run_id is None:
                run_id = str(uuid.uuid4())
            output_path = get_config().output_path
            rundir_name = f"run_{run_id}"
            rundir = join_paths(output_path, rundir_name)
            profiler = Profiler()

            # Time endpoint execution (includes pydantic input validation
            # which already happened in FastAPI before this function is called)
            t_compute_start = time.perf_counter()
            with start_run(base_dir=rundir):
                with profiler:
                    result = endpoint_func(*args, **kwargs)

                # Print profiling stats inside start_run context
                # so they go through stdio redirection to the log file
                profiler.print_stats()
            t_compute_end = time.perf_counter()

            # Time response serialization
            t_serialize_start = time.perf_counter()
            resp = create_response(
                result, accept, base_dir=output_path, binref_dir=rundir_name
            )
            t_serialize_end = time.perf_counter()

            compute_ms = (t_compute_end - t_compute_start) * 1000
            serialize_ms = (t_serialize_end - t_serialize_start) * 1000
            resp.headers["X-Server-Compute-Ms"] = f"{compute_ms:.3f}"
            resp.headers["X-Server-Serialize-Ms"] = f"{serialize_ms:.3f}"
            return resp

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

    return app


def serve(host: str, port: int, num_workers: int) -> None:
    """Start the REST API."""
    config = get_config()
    if config.debug:
        import debugpy

        debugpy.listen(("0.0.0.0", 5678))

    uvicorn.run(
        "tesseract_core.runtime.app_http:app", host=host, port=port, workers=num_workers
    )
