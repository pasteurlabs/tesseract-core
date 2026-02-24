# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid
from collections.abc import Callable
from types import ModuleType
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Header, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .config import get_config
from .core import create_endpoints, get_input_schema
from .file_interactions import SUPPORTED_FORMATS, join_paths, output_to_bytes
from .mpa import start_run
from .profiler import Profiler

logger = logging.getLogger("tesseract")

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

    tesseract_endpoints = create_endpoints(api_module)

    def wrap_computational_endpoint(
        endpoint_func: Callable,
    ) -> tuple[Callable, type[BaseModel]]:
        """Wrap computational endpoints to add profiling of serde and computation.

        For computational endpoints (apply, jacobian, etc.), this wrapper:
        1. Accepts raw Request body to allow profiling of deserialization
        2. Uses openapi_extra to preserve OpenAPI schema documentation
        3. Profiles both deserialization, computation, and serialization

        Returns:
            A tuple of (wrapped_endpoint, InputSchema) for use with openapi_extra.
        """
        # Get the input schema for manual deserialization
        InputSchema = get_input_schema(endpoint_func)

        async def wrapper(
            request: Request,
            accept: Annotated[str | None, Header()] = None,
            run_id: Annotated[str | None, Query(include_in_schema=False)] = None,
        ) -> Response:
            if run_id is None:
                run_id = str(uuid.uuid4())
            output_path = get_config().output_path
            rundir_name = f"run_{run_id}"
            rundir = join_paths(output_path, rundir_name)
            profiler = Profiler(enabled=get_config().profiling)

            raw_body = await request.body()

            with start_run(base_dir=rundir):
                with profiler:
                    # Manual deserialization inside profiler context
                    try:
                        payload = InputSchema.model_validate_json(raw_body)
                    except ValidationError as e:
                        # Return 422 with validation errors (matches FastAPI's default behavior)
                        profiler.stop()
                        return JSONResponse(
                            status_code=422,
                            content={"detail": e.errors()},
                        )

                    # Run the actual endpoint
                    result = endpoint_func(payload)

                    # Serialization inside profiler context
                    resp = create_response(
                        result, accept, base_dir=output_path, binref_dir=rundir_name
                    )

                # Stop profiler and print stats inside start_run context
                # so they go through stdio redirection to the log file
                stats_text = profiler.get_stats()
                if stats_text:
                    print("\n--- Profiling Statistics ---")
                    print(stats_text)

            return resp

        # Copy over function metadata
        wrapper.__name__ = endpoint_func.__name__
        wrapper.__doc__ = endpoint_func.__doc__

        return wrapper, InputSchema

    # Endpoints that need openapi_extra for manual body parsing
    computational_endpoints = {
        "apply",
        "jacobian",
        "jacobian_vector_product",
        "vector_jacobian_product",
    }

    for endpoint_func in tesseract_endpoints:
        endpoint_name = endpoint_func.__name__

        # Skip test endpoint unless in debug mode
        if endpoint_name == "test" and not config.debug:
            continue

        http_methods = ["GET"] if endpoint_name in GET_ENDPOINTS else ["POST"]

        if endpoint_name in computational_endpoints:
            # Wrap with profiling and manual serde
            wrapped_endpoint, InputSchema = wrap_computational_endpoint(endpoint_func)
            # Use openapi_extra to document the request body schema
            # while accepting raw Request for manual deserialization
            openapi_extra = {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": InputSchema.model_json_schema(
                                ref_template="#/components/schemas/{model}"
                            )
                        }
                    },
                    "required": True,
                }
            }
            app.add_api_route(
                f"/{endpoint_name}",
                wrapped_endpoint,
                methods=http_methods,
                openapi_extra=openapi_extra,
            )
        else:
            app.add_api_route(f"/{endpoint_name}", endpoint_func, methods=http_methods)

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
