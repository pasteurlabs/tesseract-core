# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Union
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Header, Response
from pydantic import BaseModel

from .config import get_config
from .core import create_endpoints
from .file_interactions import SUPPORTED_FORMATS, output_to_bytes

# Endpoints that should use GET instead of POST
GET_ENDPOINTS = {"input_schema", "output_schema", "health"}

# TODO: make this configurable via environment variable
DEFAULT_ACCEPT = "application/json"


# This function needs to be in global scope, so that it can be offloaded to a separate worker process.
def _run_api_function_within_worker(
    func_name: str, picklable_payload_dict: dict
) -> dict:
    # Need to load the user-supplied api module and retrieve API endpoints here,
    # since the worker doesn't inherit local (in the Python sense) definitions from the main process!
    from tesseract_core.runtime.core import get_tesseract_api

    api_module = get_tesseract_api()
    endpoints = create_endpoints(api_module)

    endpoint_func = next(
        (endpoint for endpoint in endpoints if endpoint.__name__ == func_name), None
    )
    assert endpoint_func is not None, (
        f"Function {func_name} not found in endpoints, but worker was tasked with its execution."
    )

    payload_model = endpoint_func.input_schema.model_validate(
        picklable_payload_dict
    )  # TODO: Maybe somehow skip the actual validation here, since it was already done in the main process?

    return endpoint_func(payload=payload_model).model_dump()


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
        title=config.name,
        version=config.version,
        docs_url=None,
        redoc_url="/docs",
        debug=config.debug,
    )
    tesseract_endpoints = create_endpoints(api_module)

    from concurrent.futures import ProcessPoolExecutor

    executor = ProcessPoolExecutor(
        max_workers=4
    )  # TODO: Adjust the number of workers according to user-configured num_workers!

    open_tasks = {}

    def wrap_endpoint(endpoint_func: Callable, async_endpoint: bool = False):
        endpoints_to_wrap = [
            "apply",
            "jacobian",
            "jacobian_vector_product",
            "vector_jacobian_product",
        ]

        @wraps(endpoint_func)
        async def wrapper(*args: Any, accept: str, **kwargs: Any):
            if async_endpoint:
                new_task_id = str(uuid4())
                # new_task_id = "1" # TODO: Remove this hardcoded task ID, since it is only for testing purposes!

                assert "payload" in kwargs, (
                    f"Defining async. endpoint {endpoint_func.__name__}. "
                    "For simplicity, we assume a single payload argument here, "
                    "but none was provided!"
                )
                picklable_payload_dict = kwargs["payload"].model_dump()

                open_tasks[new_task_id] = (
                    endpoint_func.__name__,
                    accept,
                    executor.submit(
                        _run_api_function_within_worker,
                        endpoint_func.__name__,
                        picklable_payload_dict=picklable_payload_dict,
                    ),
                )

                return Response(
                    status_code=202,
                    content=f'{{"task_id": "{new_task_id}"}}',
                    media_type="application/json",
                )
            else:
                result = endpoint_func(*args, **kwargs)
                return create_response(result, accept)

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
                annotation=Union[str, None],
            )
            # Other header parameters common to computational endpoints
            # could be defined and appended here as well.
            new_params = [*list(original_sig.parameters.values()), accept]
            new_sig = original_sig.replace(parameters=new_params)
            wrapper.__signature__ = new_sig
            return wrapper

    async_endpoints = [
        "apply",
        "jacobian",
        "jacobian_vector_product",
        "vector_jacobian_product",
    ]

    for endpoint_func in tesseract_endpoints:
        endpoint_name = endpoint_func.__name__

        # TODO: Remove sync endpoints for compute requests OR change them to internally use async mechanism, since
        # we only have 1 uvicorn worker now and need to offload the blocking calls to a separate process!
        wrapped_endpoint = wrap_endpoint(endpoint_func, async_endpoint=False)
        http_methods = ["GET"] if endpoint_name in GET_ENDPOINTS else ["POST"]
        app.add_api_route(f"/{endpoint_name}", wrapped_endpoint, methods=http_methods)

        if endpoint_name not in async_endpoints:
            continue

        wrapped_endpoint_async = wrap_endpoint(endpoint_func, async_endpoint=True)
        app.add_api_route(
            f"/{endpoint_name}/async_start", wrapped_endpoint_async, methods=["POST"]
        )

        class TaskRequest(BaseModel):
            task_id: str

        async def async_retrieve(
            data: TaskRequest,
            endpoint_name: str = endpoint_name,
            endpoint_func: Any = endpoint_func,
        ):
            task_id = data.task_id
            if task_id not in open_tasks:
                return Response(
                    status_code=404,
                    content=f'{{"task_id": "{task_id}", "status": "bad request", "message": "Task ID not found"}}',
                    media_type="application/json",
                )

            task_endpoint, accept, task = open_tasks[task_id]
            if (
                task_endpoint != endpoint_name
            ):  # Ensure the request is for the endpoint the task was created for
                return Response(
                    status_code=400,
                    content=(
                        f'{{"task_id": "{task_id}", "status": "bad request", '
                        f'"message": "Task ID does not match the endpoint"}}'
                    ),
                    media_type="application/json",
                )
            if not task.done():
                try:
                    await asyncio.wait_for(asyncio.wrap_future(task), timeout=10)
                except asyncio.TimeoutError:
                    return Response(
                        status_code=202,
                        content=f'{{"task_id": "{task_id}", "status": "in progress"}}',
                        media_type="application/json",
                    )

            output = task.result()

            output_model = endpoint_func.output_schema.model_validate(
                output
            )  # TODO: Avoid re-validation here?
            # # Use model_construct to avoid re-validation
            del open_tasks[task_id]
            return create_response(output_model, accept)

        app.add_api_route(
            f"/{endpoint_name}/async_retrieve",
            async_retrieve,
            methods=["POST"],
        )

    return app


def serve(host: str, port: int, num_workers: int) -> None:
    """Start the REST API."""
    uvicorn.run("tesseract_core.runtime.app_http:app", host=host, port=port, workers=1)
