# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import json
import subprocess
from collections.abc import Mapping, Sequence
from functools import cached_property
from urllib.parse import urlparse, urlunparse

import numpy as np
import requests

from . import engine


class Tesseract:
    """A Tesseract.

    This class represents a single Tesseract instance, either remote or local,
    and provides methods to run commands on it and retrieve results.

    Communication between a Tesseract and this class is done either via
    HTTP requests or `docker exec` invocations (only possible for local
    instances spawned when instantiating the class).
    """

    _client: HTTPClient | LocalClient
    image: str
    volumes: list[str] | None
    gpus: list[str] | None
    project_id: str

    def __init__(self, url: str) -> None:
        self._client = HTTPClient(url)

    @classmethod
    def from_image(
        cls,
        image: str,
        *,
        volumes: list[str] | None = None,
        gpus: list[str] | None = None,
    ):
        obj = cls.__new__(cls)

        obj.image = image
        obj.volumes = volumes
        obj.gpus = gpus

        return obj

    def __enter__(self):
        if not self.client_type == "local":
            raise ValueError(
                "Use Tesseract.from_image(...) to create a context-managed Tesseract instance."
            )
        self._serve(volumes=self.volumes, gpus=self.gpus)
        self._client = LocalClient(self.tesseract_container_id)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        engine.teardown(self.project_id)

    def _serve(
        self,
        port: str = "",
        volumes: list[str] | None = None,
        gpus: list[str] | None = None,
    ) -> None:
        if hasattr(self, "tesseract_container_id"):
            self.tesseract_container_id: str
            raise RuntimeError(
                "Client already attached to the Tesseract "
                f"container {self.tesseract_container_id}"
            )
        project_id = engine.serve([self.image], port=port, volumes=volumes, gpus=gpus)

        command = ["docker", "compose", "-p", project_id, "ps", "--format", "json"]
        result = subprocess.run(command, capture_output=True, text=True)

        containers = json.loads(result.stdout)

        if containers:
            first_container_id = containers["ID"]
        else:
            raise RuntimeError("No containers found.")

        self.tesseract_container_id = first_container_id
        self.project_id = project_id

    @cached_property
    def client_type(self) -> str:
        """Get the type of client being used ('http' or 'local')."""
        return (
            "http"
            if hasattr(self, "_client") and isinstance(self._client, HTTPClient)
            else "local"
        )

    @property
    def url(self) -> str | None:
        """Get the URL if using HTTP client, None for local client."""
        return getattr(self._client, "url", None)

    @cached_property
    def openapi_schema(self) -> dict:
        """Get the OpenAPI schema of this Tessseract.

        Returns:
            dictionary with the OpenAPI Schema.
        """
        return self._client._run_tesseract("openapi_schema")

    @cached_property
    def input_schema(self) -> dict:
        """Get the input schema of this Tessseract.

        Returns:
             dictionary with the input schema.
        """
        return self._client._run_tesseract("input_schema")

    @cached_property
    def output_schema(self) -> dict:
        """Get the output schema of this Tessseract.

        Returns:
             dictionary with the output schema.
        """
        return self._client._run_tesseract("output_schema")

    @property
    def available_endpoints(self) -> list[str]:
        """Get the list of available endpoints.

        Returns:
            a list with all available endpoints for this Tesseract.
        """
        return [endpoint.lstrip("/") for endpoint in self.openapi_schema["paths"]]

    def _run_tesseract(self, endpoint: str, payload: dict | None = None) -> dict:
        return self._client._run_tesseract(endpoint, payload)

    def apply(self, inputs: dict) -> dict:
        """Run apply endpoint.

        Args:
            inputs: a dictionary with the inputs.

        Returns:
            dictionary with the results.
        """
        payload = {"inputs": inputs}
        return self._run_tesseract("apply", payload)

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
        return self._run_tesseract("jacobian", payload)

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
        return self._run_tesseract("jacobian_vector_product", payload)

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
        return self._run_tesseract("vector_jacobian_product", payload)


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
            arr = np.array(encoded_arr["data"]["buffer"])
    else:
        arr = encoded_arr
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
        response.raise_for_status()

        data = response.json()

        if endpoint in [
            "apply",
            "jacobian",
            "jacobian_vector_product",
            "vector_jacobian_product",
        ]:
            data = _tree_map(
                _decode_array,
                response.json(),
                is_leaf=lambda x: type(x) is dict and "shape" in x,
            )

        return data

    def _run_tesseract(self, endpoint: str, payload: dict | None = None) -> dict:
        if endpoint in ["input_schema", "openapi_schema", "health"]:
            method = "GET"
        else:
            method = "POST"

        if endpoint == "openapi_schema":
            endpoint = "openapi.json"

        return self._request(endpoint, method, payload)


class LocalClient:
    """A client that connects to a local Tesseract."""

    def __init__(self, container_id: str) -> None:
        self.container_id = container_id

    def _run_tesseract(
        self,
        endpoint: str,
        payload: dict | None = None,
    ) -> dict:
        command = endpoint.replace("_", "-")
        if payload:
            encoded_payload = _tree_map(
                _encode_array, payload, is_leaf=lambda x: hasattr(x, "shape")
            )
        else:
            encoded_payload = None

        args = []
        if encoded_payload:
            args.append(json.dumps(encoded_payload))

        out, err = engine.exec_tesseract(self.container_id, command, args)

        if err:
            raise RuntimeError(err)

        data = json.loads(out)

        if command in [
            "apply",
            "jacobian",
            "jacobian-vector-product",
            "vector-jacobian-product",
        ]:
            data = _tree_map(
                _decode_array, data, is_leaf=lambda x: type(x) is dict and "shape" in x
            )

        return data
