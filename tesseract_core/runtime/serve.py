# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
from collections.abc import Callable
from types import ModuleType
from typing import Annotated

import orjson
import uvicorn
from fastapi import FastAPI, Header, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .config import get_config
from .core import create_endpoints, get_input_schema
from .file_interactions import SUPPORTED_FORMATS, join_paths, output_to_bytes
from .mpa import start_run
from .profiler import Profiler

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

    # Computational endpoints get a raw request handler that bypasses FastAPI's
    # default model_validate_json (serde_json) in favour of from_json (jiter) +
    # model_validate, which is ~3x faster for large base64 strings.
    # See https://github.com/pydantic/pydantic/issues/12911
    computational_endpoints = {
        "apply",
        "jacobian",
        "jacobian_vector_product",
        "vector_jacobian_product",
    }

    # Track schemas that need to be added to components/schemas
    schemas_to_register: dict[str, dict] = {}

    def wrap_computational_endpoint(
        endpoint_func: Callable,
    ) -> tuple[Callable, type[BaseModel]]:
        """Wrap a computational endpoint with raw request handling.

        Instead of letting FastAPI parse the request body through
        model_validate_json (which uses serde_json), we:
        1. Accept the raw Request body
        2. Parse JSON with from_json (jiter, ~3x faster for large strings)
        3. Validate with model_validate (Python mode, no redundant string scanning)

        Returns:
            A tuple of (wrapped_endpoint, InputSchema) for use with openapi_extra.
        """
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
            profiler = Profiler()

            raw_body = await request.body()

            with start_run(base_dir=rundir):
                with profiler:
                    try:
                        json_data = orjson.loads(raw_body)
                    except json.JSONDecodeError as e:
                        return JSONResponse(
                            status_code=422,
                            content={
                                "detail": [{"type": "json_invalid", "msg": str(e)}]
                            },
                        )

                    try:
                        payload = InputSchema.model_validate(json_data)
                    except ValidationError as e:
                        return JSONResponse(
                            status_code=422,
                            content={"detail": jsonable_encoder(e.errors())},
                        )

                    result = endpoint_func(payload)

                # Print profiling stats inside start_run context
                # so they go through stdio redirection to the log file
                profiler.print_stats()

            return create_response(
                result, accept, base_dir=output_path, binref_dir=rundir_name
            )

        wrapper.__name__ = endpoint_func.__name__
        wrapper.__doc__ = endpoint_func.__doc__

        return wrapper, InputSchema

    for endpoint_func in tesseract_endpoints:
        endpoint_name = endpoint_func.__name__

        # Skip test endpoint unless in debug mode
        if endpoint_name == "test" and not config.debug:
            continue

        http_methods = ["GET"] if endpoint_name in GET_ENDPOINTS else ["POST"]

        if endpoint_name in computational_endpoints:
            wrapped_endpoint, InputSchema = wrap_computational_endpoint(endpoint_func)

            # Generate schema and extract $defs for registration
            full_schema = InputSchema.model_json_schema(
                ref_template="#/components/schemas/{model}"
            )
            if "$defs" in full_schema:
                schemas_to_register.update(full_schema.pop("$defs"))

            schema_name = full_schema.get(
                "title", f"{endpoint_name.title()}InputSchema"
            )
            schemas_to_register[schema_name] = full_schema

            # Use openapi_extra to document the request body schema
            # while accepting raw Request for manual deserialization
            openapi_extra = {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": f"#/components/schemas/{schema_name}"}
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

    # Override OpenAPI schema generation to include our custom schemas
    original_openapi = app.openapi

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = original_openapi()
        if "components" not in openapi_schema:
            openapi_schema["components"] = {}
        if "schemas" not in openapi_schema["components"]:
            openapi_schema["components"]["schemas"] = {}
        openapi_schema["components"]["schemas"].update(schemas_to_register)
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

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
