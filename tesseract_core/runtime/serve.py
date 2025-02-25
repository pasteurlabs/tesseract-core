# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Union

import uvicorn
from fastapi import FastAPI, Header, Response
from pydantic import BaseModel

from .config import get_config
from .core import create_endpoints
from .file_interactions import SUPPORTED_FORMATS, output_to_bytes

# Endpoints that should use GET instead of POST
GET_ENDPOINTS = {
    "input_schema",
    "output_schema",
    "health",
}

# TODO: make this configurable via environment variable
DEFAULT_ACCEPT = "application/json"


def create_response(model: BaseModel, accept: str) -> Response:
    """Create a response of the format specified by the Accept header."""
    if accept is None or accept == "*/*":
        accept = DEFAULT_ACCEPT

    output_format: SUPPORTED_FORMATS = accept.split("/")[-1]
    content = output_to_bytes(model, output_format)

    return Response(status_code=200, content=content, media_type=accept)


def create_rest_api(api_module: ModuleType) -> FastAPI:
    """Create the Tesseract REST API."""
    config = get_config()
    app = FastAPI(
        title=config.tesseract_name,
        version=config.tesseract_version,
        docs_url=None,
        redoc_url="/docs",
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
        async def wrapper(*args: Any, accept: str, **kwargs: Any):
            result = endpoint_func(*args, **kwargs)
            return create_response(result, accept)

        if endpoint_func.__name__ not in endpoints_to_wrap:
            return endpoint_func
        else:
            # wrapper's signarure will be the same as endpoint
            # func's signature. We do however need to change this
            # in order to add a Header parameter that FastAPI
            # will understand.
            original_sig = inspect.signature(endpoint_func)
            accept = inspect.Parameter(
                "accept",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=Header(default=None),
                annotation=Union[str, None],
            )
            # Other header parameters common to computational endpoints
            # could be defined and appended here as well.
            new_params = [*list(original_sig.parameters.values()), accept]
            new_sig = original_sig.replace(parameters=new_params)
            wrapper.__signature__ = new_sig
            return wrapper

    for endpoint_func in tesseract_endpoints:
        endpoint_name = endpoint_func.__name__
        wrapped_endpoint = wrap_endpoint(endpoint_func)
        http_methods = ["GET"] if endpoint_name in GET_ENDPOINTS else ["POST"]
        app.add_api_route(f"/{endpoint_name}", wrapped_endpoint, methods=http_methods)

    return app


def serve(host: str, port: int, num_workers: int) -> None:
    """Start the REST API."""
    uvicorn.run(
        "tesseract_core.runtime.app_http:app", host=host, port=port, workers=num_workers
    )
