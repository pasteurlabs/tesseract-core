# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, wraps
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
import requests
from pydantic import BaseModel, ValidationError
from pydantic_core import InitErrorDetails

from . import engine


@dataclass
class SpawnConfig:
    """Configuration for spawning a Tesseract.

    Attributes:
        image: The image to use.
        volumes: List of volumes to mount.
        gpus: List of GPUs to use.
        debug: Whether to run in debug mode.
    """

    image: str
    volumes: list[str] | None
    gpus: list[str] | None
    debug: bool


def requires_client(func):
    """Decorator to require a client for a Tesseract instance."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._client:
            raise RuntimeError(
                f"{self.__class__.__name__} must be used as a context manager when created via `from_image`."
            )
        return func(self, *args, **kwargs)

    return wrapper


class Tesseract:
    """A Tesseract.

    This class represents a single Tesseract instance, either remote or local,
    and provides methods to run commands on it and retrieve results.

    Communication between a Tesseract and this class is done either via
    HTTP requests or `docker exec` invocations (only possible for local
    instances spawned when instantiating the class).
    """

    def __init__(self, url: str) -> None:
        self._spawn_config = None
        self._serve_context = None
        self._lastlog = None
        self._client = HTTPClient(url)

    @classmethod
    def from_url(cls, url: str) -> Tesseract:
        obj = cls.__new__(cls)
        obj.__init__(url)
        return obj

    @classmethod
    def from_image(
        cls,
        image: str,
        *,
        volumes: list[str] | None = None,
        gpus: list[str] | None = None,
    ) -> Tesseract:
        obj = cls.__new__(cls)
        obj._spawn_config = SpawnConfig(
            image=image,
            volumes=volumes,
            gpus=gpus,
            debug=True,
        )
        obj._serve_context = None
        obj._lastlog = None
        obj._client = None
        return obj

    @classmethod
    def from_tesseract_api(
        cls,
        tesseract_api_path: str | Path,
    ) -> Tesseract:
        """Create a Tesseract instance from a tesseract API path.

        Warning: This does not use a containerized Tesseract, but rather
        imports the Tesseract API directly. This is useful for debugging,
        but requires a matching runtime environment + all dependencies to be
        installed locally.

        Args:
            tesseract_api_path: Path to the tesseract API.

        Returns:
            A Tesseract instance.
        """
        obj = cls.__new__(cls)
        obj._spawn_config = None
        obj._serve_context = None
        obj._lastlog = None
        obj._client = LocalClient(tesseract_api_path)
        return obj

    def __enter__(self):
        if self._serve_context is not None:
            raise RuntimeError(
                "Cannot nest the same `with Tesseract ...` context manager."
            )

        if self._client is not None:
            # Tesseract is already being served -> no-op
            return self

        self.serve()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._serve_context is None:
            # This can happen if __enter__ short-cirtuits
            return
        self.teardown()

    def server_logs(self):
        """Get the logs of the Tesseract server.

        Returns:
            logs of the Tesseract server.
        """
        if self._spawn_config is None:
            raise RuntimeError(
                "Can only retrieve logs for a Tesseract created via from_image."
            )
        if self._serve_context is None:
            return self._lastlog
        return engine.logs(self._serve_context["container_id"])

    def serve(self, port: str = "") -> None:
        """Serve the Tesseract.

        Args:
            port: Port to serve the Tesseract on.
        """
        if self._spawn_config is None:
            raise RuntimeError("Can only serve a Tesseract created via from_image.")
        if self._serve_context is not None:
            raise RuntimeError("Tesseract is already being served.")
        project_id, container_id, served_port = self._serve(
            self._spawn_config.image,
            port=port,
            volumes=self._spawn_config.volumes,
            gpus=self._spawn_config.gpus,
            debug=self._spawn_config.debug,
        )
        self._serve_context = dict(
            project_id=project_id,
            container_id=container_id,
            port=served_port,
        )
        self._lastlog = None
        self._client = HTTPClient(f"http://localhost:{served_port}")

    def teardown(self) -> None:
        """Teardown the Tesseract.

        This will stop and remove the Tesseract container.
        """
        if self._serve_context is None:
            raise RuntimeError("Tesseract is not being served.")
        self._lastlog = self.server_logs()
        engine.teardown(self._serve_context["project_id"])
        self._client = None
        self._serve_context = None

    def __del__(self):
        if self._serve_context is not None:
            self.teardown()

    @staticmethod
    def _serve(
        image: str,
        port: str = "",
        volumes: list[str] | None = None,
        gpus: list[str] | None = None,
        debug: bool = False,
    ) -> tuple[str, str, int]:
        project_id = engine.serve(
            [image], port=port, volumes=volumes, gpus=gpus, debug=debug
        )

        command = ["docker", "compose", "-p", project_id, "ps", "--format", "json"]
        result = subprocess.run(command, capture_output=True, text=True)

        # This relies on the fact that result.stdout from docker compose ps
        # contains multiple json dicts, one for each container, separated by newlines,
        # but json.loads will only parse the first one.
        # The first_container dict contains useful info like container id, ports, etc.
        first_container = json.loads(result.stdout)

        if first_container:
            first_container_id = first_container["ID"]
            first_container_port = first_container["Publishers"][0]["PublishedPort"]
        else:
            raise RuntimeError("No containers found.")

        return project_id, first_container_id, first_container_port

    @cached_property
    @requires_client
    def openapi_schema(self) -> dict:
        """Get the OpenAPI schema of this Tessseract.

        Returns:
            dictionary with the OpenAPI Schema.
        """
        return self._client.run_tesseract("openapi_schema")

    @cached_property
    @requires_client
    def input_schema(self) -> dict:
        """Get the input schema of this Tessseract.

        Returns:
             dictionary with the input schema.
        """
        return self._client.run_tesseract("input_schema")

    @cached_property
    @requires_client
    def output_schema(self) -> dict:
        """Get the output schema of this Tessseract.

        Returns:
             dictionary with the output schema.
        """
        return self._client.run_tesseract("output_schema")

    @property
    @requires_client
    def available_endpoints(self) -> list[str]:
        """Get the list of available endpoints.

        Returns:
            a list with all available endpoints for this Tesseract.
        """
        return [endpoint.lstrip("/") for endpoint in self.openapi_schema["paths"]]

    def run_tesseract(self, endpoint: str, payload: dict | None = None) -> dict:
        return self._client.run_tesseract(endpoint, payload)

    @requires_client
    def apply(self, inputs: dict) -> dict:
        """Run apply endpoint.

        Args:
            inputs: a dictionary with the inputs.

        Returns:
            dictionary with the results.
        """
        payload = {"inputs": inputs}
        return self.run_tesseract("apply", payload)

    @requires_client
    def abstract_eval(self, abstract_inputs: dict) -> dict:
        """Run abstract eval endpoint.

        Args:
            abstract_inputs: a dictionary with the (abstract) inputs.

        Returns:
            dictionary with the results.
        """
        payload = {"inputs": abstract_inputs}
        return self.run_tesseract("abstract_eval", payload)

    @requires_client
    def health(self) -> dict:
        """Check the health of the Tesseract.

        Returns:
            dictionary with the health status.
        """
        return self.run_tesseract("health")

    @requires_client
    def jacobian(
        self, inputs: dict, jac_inputs: list[str], jac_outputs: list[str]
    ) -> dict:
        """Calculate the Jacobian of (some of the) outputs w.r.t. (some of the) inputs.

        Args:
            inputs: a dictionary with the inputs.
            jac_inputs: Inputs with respect to which derivatives will be calculated.
            jac_outputs: Outputs which will be differentiated.

        Returns:
            dictionary with the results.
        """
        if "jacobian" not in self.available_endpoints:
            raise NotImplementedError("Jacobian not implemented for this Tesseract.")

        payload = {
            "inputs": inputs,
            "jac_inputs": jac_inputs,
            "jac_outputs": jac_outputs,
        }
        return self.run_tesseract("jacobian", payload)

    @requires_client
    def jacobian_vector_product(
        self,
        inputs: dict,
        jvp_inputs: list[str],
        jvp_outputs: list[str],
        tangent_vector: dict,
    ) -> dict:
        """Calculate the Jacobian Vector Product (JVP) of (some of the) outputs w.r.t. (some of the) inputs.

        Args:
            inputs: a dictionary with the inputs.
            jvp_inputs: Inputs with respect to which derivatives will be calculated.
            jvp_outputs: Outputs which will be differentiated.
            tangent_vector: Element of the tangent space to multiply with the Jacobian.

        Returns:
            dictionary with the results.
        """
        if "jacobian_vector_product" not in self.available_endpoints:
            raise NotImplementedError(
                "Jacobian Vector Product (JVP) not implemented for this Tesseract."
            )

        payload = {
            "inputs": inputs,
            "jvp_inputs": jvp_inputs,
            "jvp_outputs": jvp_outputs,
            "tangent_vector": tangent_vector,
        }
        return self.run_tesseract("jacobian_vector_product", payload)

    @requires_client
    def vector_jacobian_product(
        self,
        inputs: dict,
        vjp_inputs: list[str],
        vjp_outputs: list[str],
        cotangent_vector: dict,
    ) -> dict:
        """Calculate the Vector Jacobian Product (VJP) of (some of the) outputs w.r.t. (some of the) inputs.

        Args:
            inputs: a dictionary with the inputs.
            vjp_inputs: Inputs with respect to which derivatives will be calculated.
            vjp_outputs: Outputs which will be differentiated.
            cotangent_vector: Element of the cotangent space to multiply with the Jacobian.


        Returns:
            dictionary with the results.
        """
        if "vector_jacobian_product" not in self.available_endpoints:
            raise NotImplementedError(
                "Vector Jacobian Product (VJP) not implemented for this Tesseract."
            )

        payload = {
            "inputs": inputs,
            "vjp_inputs": vjp_inputs,
            "vjp_outputs": vjp_outputs,
            "cotangent_vector": cotangent_vector,
        }
        return self.run_tesseract("vector_jacobian_product", payload)


def _tree_map(func, tree, is_leaf=None):
    if is_leaf is not None and is_leaf(tree):
        return func(tree)
    if isinstance(tree, Mapping):  # Dictionary-like structure
        return {key: _tree_map(func, value, is_leaf) for key, value in tree.items()}

    if isinstance(tree, Sequence) and not isinstance(
        tree, (str, bytes)
    ):  # List, tuple, etc.
        return type(tree)(_tree_map(func, item, is_leaf) for item in tree)

    # If nothing above matched do nothing
    return tree


def _encode_array(arr: np.ndarray, b64: bool = True):
    if b64:
        data = {
            "buffer": base64.b64encode(arr.tobytes()).decode(),
            "encoding": "base64",
        }
    else:
        data = {
            "buffer": arr.tolist(),
            "encoding": "raw",
        }

    return {
        "shape": arr.shape,
        "dtype": arr.dtype.name,
        "data": data,
    }


def _decode_array(encoded_arr: dict):
    if "data" in encoded_arr:
        if encoded_arr["data"]["encoding"] == "base64":
            data = base64.b64decode(encoded_arr["data"]["buffer"])
            arr = np.frombuffer(data, dtype=encoded_arr["dtype"])
        else:
            arr = np.array(encoded_arr["data"]["buffer"], dtype=encoded_arr["dtype"])
    else:
        raise ValueError("Encoded array does not contain 'data' key. Cannot decode.")
    arr = arr.reshape(encoded_arr["shape"])
    return arr


class HTTPClient:
    """HTTP Client for Tesseracts."""

    def __init__(self, url: str) -> None:
        self._url = self._sanitize_url(url)

    @staticmethod
    def _sanitize_url(url: str) -> str:
        parsed = urlparse(url)

        if not parsed.scheme:
            url = f"http://{url}"
            parsed = urlparse(url)

        sanitized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
        return sanitized

    @property
    def url(self) -> str:
        """(Sanitized) URL to connect to."""
        return self._url

    def _request(
        self, endpoint: str, method: str = "GET", payload: dict | None = None
    ) -> dict:
        url = f"{self.url}/{endpoint.lstrip('/')}"

        if payload:
            encoded_payload = _tree_map(
                _encode_array, payload, is_leaf=lambda x: hasattr(x, "shape")
            )
        else:
            encoded_payload = None

        response = requests.request(method=method, url=url, json=encoded_payload)

        if response.status_code == requests.codes.unprocessable_entity:
            # Try and raise a more helpful error if the response is a Pydantic error
            try:
                data = response.json()
            except requests.JSONDecodeError:
                # Is not a Pydantic error
                data = {}
            if "detail" in data:
                errors = [
                    InitErrorDetails(
                        type=e["type"],
                        loc=tuple(e["loc"]),
                        input=e.get("input"),
                        ctx=e.get("ctx", {}),
                    )
                    for e in data["detail"]
                ]
                raise ValidationError.from_exception_data(
                    f"endpoint {endpoint}", errors
                )

        if not response.ok:
            raise RuntimeError(
                f"Error {response.status_code} from Tesseract: {response.text}"
            )

        data = response.json()

        if endpoint in [
            "apply",
            "jacobian",
            "jacobian_vector_product",
            "vector_jacobian_product",
        ]:
            data = _tree_map(
                _decode_array,
                data,
                is_leaf=lambda x: type(x) is dict and "shape" in x,
            )

        return data

    def run_tesseract(self, endpoint: str, payload: dict | None = None) -> dict:
        if endpoint in [
            "input_schema",
            "output_schema",
            "openapi_schema",
            "health",
        ]:
            method = "GET"
        else:
            method = "POST"

        if endpoint == "openapi_schema":
            endpoint = "openapi.json"

        return self._request(endpoint, method, payload)


class LocalClient:
    """Local Client for Tesseracts."""

    def __init__(self, tesseract_api_path: str | Path) -> None:
        from tesseract_core.runtime.core import create_endpoints, load_module_from_path
        from tesseract_core.runtime.serve import create_rest_api

        tesseract_api_path = Path(tesseract_api_path).resolve(strict=True)
        if not tesseract_api_path.is_file():
            raise RuntimeError(
                f"Tesseract API path {tesseract_api_path} is not a file."
            )

        try:
            tesseract_api = load_module_from_path(tesseract_api_path)
        except ImportError as ex:
            raise RuntimeError(
                f"Cannot load Tesseract API from {tesseract_api_path}"
            ) from ex

        self._endpoints = {
            func.__name__: func for func in create_endpoints(tesseract_api)
        }
        self._openapi_schema = create_rest_api(tesseract_api).openapi()

    def run_tesseract(self, endpoint: str, payload: dict | None = None) -> dict:
        if endpoint == "openapi_schema":
            return self._openapi_schema

        if endpoint not in self._endpoints:
            raise RuntimeError(f"Endpoint {endpoint} not found in Tesseract API.")

        func = self._endpoints[endpoint]
        InputSchema = func.__annotations__.get("payload", None)
        OutputSchema = func.__annotations__.get("return", None)

        if InputSchema is not None:
            parsed_payload = InputSchema.model_validate(payload)
        else:
            parsed_payload = None

        try:
            if parsed_payload is not None:
                result = self._endpoints[endpoint](parsed_payload)
            else:
                result = self._endpoints[endpoint]()
        except Exception as ex:
            raise RuntimeError(f"Error running Tesseract API {endpoint}.") from ex

        if OutputSchema is not None and issubclass(OutputSchema, BaseModel):
            # Validate via schema, then dump to stay consistent with other clients
            result = OutputSchema.model_validate(result).model_dump()

        return result
