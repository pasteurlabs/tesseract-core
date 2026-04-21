# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import tempfile
import traceback
import uuid
import warnings
import weakref
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property, wraps
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, TypeAlias
from urllib.parse import urlparse, urlunparse

import numpy as np
import orjson
import pybase64
import requests
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError, from_json

from . import engine
from .logs import LogStreamer

PathLike: TypeAlias = str | Path
BoolOrCallable: TypeAlias = bool | Callable[[str], Any]


def requires_client(func: Callable) -> Callable:
    """Decorator to require a client for a Tesseract instance."""

    @wraps(func)
    def wrapper(self: Tesseract, *args: Any, **kwargs: Any) -> Any:
        if not self._client:
            raise RuntimeError(
                f"When creating a {self.__class__.__name__} via `from_image`, "
                "you must either use it as a context manager or call .serve() before use."
            )
        return func(self, *args, **kwargs)

    return wrapper


class Tesseract:
    """A Tesseract.

    This class represents a single Tesseract instance, either remote or local,
    and provides methods to run commands on it and retrieve results.

    Communication between a Tesseract and this class is done either via
    HTTP requests or directly via Python calls to the Tesseract API.
    """

    _spawn_config: dict | None = None
    _serve_context: dict | None = None
    _lastlog: str | None = None
    _client: HTTPClient | LocalClient | None = None
    _stream_logs: BoolOrCallable = False

    def __init__(self, url: str, server_output_path: str | Path | None = None) -> None:
        warnings.warn(
            "Direct instantiation of Tesseract is deprecated. "
            "Use Tesseract.from_url(), Tesseract.from_image(), or Tesseract.from_tesseract_api() instead.",
            UserWarning,
            stacklevel=2,
        )
        self._client = HTTPClient(url, output_path=server_output_path)

    @classmethod
    def from_url(
        cls, url: str, server_output_path: str | Path | None = None
    ) -> Tesseract:
        """Create a Tesseract instance from a URL.

        This is useful for connecting to a remote Tesseract instance.

        Args:
            url: The URL of the Tesseract instance.
            server_output_path: Path where binary output files are stored when using json+binref.
                Required when the Tesseract is served with --output-format=json+binref.
                Must be a path accessible from the client machine (e.g., via a shared or
                mounted filesystem), since the server writes .bin files there and the
                client reads them from the same path.

        Returns:
            A Tesseract instance.
        """
        obj = cls.__new__(cls)
        obj._client = HTTPClient(url, output_path=server_output_path)
        return obj

    @classmethod
    def from_image(
        cls,
        image_name: str,
        *,
        host_ip: str = "127.0.0.1",
        port: str | None = None,
        network: str | None = None,
        network_alias: str | None = None,
        volumes: list[str] | None = None,
        environment: dict[str, str] | None = None,
        gpus: list[str] | None = None,
        num_workers: int = 1,
        user: str | None = None,
        memory: str | None = None,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        output_format: Literal["json", "json+base64", "json+binref"] = "json+base64",
        docker_args: list[str] | None = None,
        runtime_config: dict[str, Any] | None = None,
        stream_logs: BoolOrCallable = False,
    ) -> Tesseract:
        """Create a Tesseract instance from a Docker image.

        When using this method, the Tesseract will be spawned in a Docker
        container, serving the Tesseract API via HTTP. To use the Tesseract,
        you need to call the `serve` method or use it as a context manager.

        Example:
            >>> with Tesseract.from_image("my_tesseract") as t:
            ...    # Use tesseract here

        This will automatically teardown the Tesseract when exiting the
        context manager.

        Args:
            image_name: Tesseract image name to serve.
            host_ip: IP address to bind the Tesseracts to.
            port: port or port range to serve each Tesseract on.
            network: name of the network the Tesseract will be attached to.
            network_alias: alias to use for the Tesseract within the network.
            volumes: list of paths to mount in the Tesseract container.
            environment: dictionary of environment variables to pass to the Tesseract.
            gpus: IDs of host Nvidia GPUs to make available to the Tesseracts.
            num_workers: number of workers to use for serving the Tesseracts.
            user: user to run the Tesseracts as, e.g. '1000' or '1000:1000' (uid:gid).
                Defaults to the current user.
            memory: Memory limit for the container (e.g., "512m", "2g"). Minimum allowed is 6m.
            input_path: Input path to read input files from, such as local directory or S3 URI.
            output_path: Output path to write output files to, such as local directory or S3 URI.
                Required when using json+binref output format.
            output_format: Format to use for the output data. json+binref requires output_path to be set.
                This has no impact on what is returned to Python and only affects the format that is used internally.
            docker_args: Additional arguments to pass to the container runtime (e.g., Docker).
            runtime_config: Dictionary of runtime configuration options to pass to the Tesseract.
                These are converted to TESSERACT_* environment variables. For example,
                `{"profiling": True}` enables profiling via TESSERACT_PROFILING=true.
            stream_logs: If True, stream logs to stdout while endpoints run.
                If a callable, stream logs to that callable instead.

        Returns:
            A Tesseract instance.
        """
        obj = cls.__new__(cls)

        if environment is None:
            environment = {}

        if volumes is None:
            volumes = []
        if input_path is not None:
            input_path = Path(input_path).resolve()
        if output_path is not None:
            output_path = Path(output_path).resolve()
        else:
            # Auto-create temp directory for output (enables stream_logs without explicit output_path)
            output_path = Path(tempfile.mkdtemp(prefix="tesseract_output_"))

        obj._stream_logs = stream_logs
        obj._spawn_config = dict(
            image_name=image_name,
            volumes=volumes,
            environment=environment,
            gpus=gpus,
            num_workers=num_workers,
            network=network,
            network_alias=network_alias,
            user=user,
            memory=memory,
            input_path=input_path,
            output_path=output_path,
            output_format=output_format,
            runtime_config=runtime_config,
            port=port,
            host_ip=host_ip,
            debug=True,
            docker_args=docker_args,
        )
        return obj

    @classmethod
    def from_tesseract_api(
        cls,
        tesseract_api: str | Path | ModuleType,
        input_path: Path | None = None,
        output_path: Path | None = None,
        output_format: Literal["json", "json+base64", "json+binref"] = "json+base64",
        runtime_config: dict[str, Any] | None = None,
        stream_logs: BoolOrCallable = False,
    ) -> Tesseract:
        """Create a Tesseract instance from a Tesseract API module.

        Warning: This does not use a containerized Tesseract, but rather
        imports the Tesseract API directly. This is useful for debugging,
        but requires a matching runtime environment + all dependencies to be
        installed locally.

        Args:
            tesseract_api: Path to the `tesseract_api.py` file, or an
                already imported Tesseract API module.
            input_path: Path of input directory. All paths in the tesseract
                payload have to be relative to this path.
            output_path: Path of output directory. All paths in the tesseract
                result with be given relative to this path. Required when using json+binref.
            output_format: Format to use for the output data. json+binref requires output_path.
                This has no impact on what is returned to Python and only affects the format that is used internally.
            runtime_config: Dictionary of runtime configuration options to pass to the Tesseract.
                For example, `{"profiling": True}` enables profiling.
            stream_logs: If True, stream logs to stdout while endpoints run.
                If a callable, stream logs to that callable instead.

        Returns:
            A Tesseract instance.
        """
        from tesseract_core.runtime.config import update_config

        if isinstance(tesseract_api, str | Path):
            from tesseract_core.runtime.core import load_module_from_path

            tesseract_api_path = Path(tesseract_api).resolve(strict=True)
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

        if input_path is not None:
            update_config(input_path=str(input_path.resolve()))

        resolved_output_path = None
        if output_path is not None:
            resolved_output_path = engine._resolve_file_path(output_path, make_dir=True)
            update_config(output_path=str(resolved_output_path))

        # Apply runtime_config options
        config_kwargs: dict[str, Any] = {"output_format": output_format, "debug": True}
        if runtime_config is not None:
            config_kwargs.update(runtime_config)
        update_config(**config_kwargs)

        obj = cls.__new__(cls)
        obj._stream_logs = stream_logs
        obj._client = LocalClient(tesseract_api, output_path=resolved_output_path)
        return obj

    def __enter__(self) -> Tesseract:
        """Enter the Tesseract context.

        This will start the Tesseract server if it is not already running.
        """
        if self._serve_context is not None:
            raise RuntimeError("Cannot serve the same Tesseract multiple times.")

        if self._client is not None:
            # Tesseract is already being served -> no-op
            return self

        self.serve()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the Tesseract context.

        This will stop the Tesseract server if it is running.
        """
        if self._serve_context is None:
            # This can happen if __enter__ short-circuits (e.g., from_tesseract_api)
            return
        self.teardown()

    def server_logs(self) -> str:
        """Get the logs of the Tesseract server.

        Returns:
            logs of the Tesseract server.
        """
        if self._spawn_config is None:
            raise RuntimeError(
                "Can only retrieve logs for a Tesseract created via from_image."
            )
        if self._serve_context is None:
            return self._lastlog or ""
        return engine.logs(self._serve_context["container_name"])

    def serve(self) -> None:
        """Serve the Tesseract until it is stopped."""
        if self._spawn_config is None:
            raise RuntimeError("Can only serve a Tesseract created via from_image.")
        if self._serve_context is not None:
            raise RuntimeError("Tesseract is already being served.")
        container_name, container = engine.serve(**self._spawn_config)
        self._serve_context = dict(
            container_name=container_name,
            port=container.host_port,
            network=self._spawn_config["network"],
            network_alias=self._spawn_config["network_alias"],
        )
        host_ip = self._spawn_config["host_ip"]
        self._lastlog = None
        output_path = self._spawn_config.get("output_path")
        self._client = HTTPClient(
            f"http://{host_ip}:{container.host_port}",
            output_path=Path(output_path) if output_path else None,
        )

        # Ensure that the Tesseract is torn down once the object is garbage collected,
        # to avoid orphaned containers if the user forgets to call .teardown()
        def _silent_teardown(name: str) -> None:
            from tesseract_core.sdk.docker_client import NotFound

            try:
                engine.teardown(name)
            except NotFound:
                pass

        self._atexit_finalizer = weakref.finalize(
            self, _silent_teardown, container_name
        )

    def teardown(self) -> None:
        """Teardown the Tesseract.

        This will stop and remove the Tesseract container.
        """
        if self._serve_context is None:
            raise RuntimeError("Tesseract is not being served.")
        self._lastlog = self.server_logs()
        engine.teardown(self._serve_context["container_name"])
        self._client = None
        self._serve_context = None
        self._atexit_finalizer.detach()

    @cached_property
    @requires_client
    def openapi_schema(self) -> dict:
        """Get the OpenAPI schema of this Tesseract.

        Returns:
            dictionary with the OpenAPI Schema.
        """
        return self._client.run_tesseract("openapi_schema")

    @property
    @requires_client
    def available_endpoints(self) -> list[str]:
        """Get the list of available endpoints.

        Returns:
            a list with all available endpoints for this Tesseract.
        """
        return [endpoint.lstrip("/") for endpoint in self.openapi_schema["paths"]]

    @requires_client
    def apply(
        self,
        inputs: dict,
        run_id: str | None = None,
    ) -> dict:
        """Run apply endpoint.

        Args:
            inputs: a dictionary with the inputs.
            run_id: a string to identify the run. Run outputs will be located
                    in a directory suffixed with this id.

        Returns:
            dictionary with the results.
        """
        payload = {"inputs": inputs}
        return self._client.run_tesseract("apply", payload, run_id, self._stream_logs)

    @requires_client
    def abstract_eval(self, abstract_inputs: dict) -> dict:
        """Run abstract eval endpoint.

        Args:
            abstract_inputs: a dictionary with the (abstract) inputs.

        Returns:
            dictionary with the results.
        """
        payload = {"inputs": abstract_inputs}
        return self._client.run_tesseract("abstract_eval", payload)

    @requires_client
    def health(self) -> dict:
        """Check the health of the Tesseract.

        Returns:
            dictionary with the health status.
        """
        return self._client.run_tesseract("health")

    @requires_client
    def jacobian(
        self,
        inputs: dict,
        jac_inputs: list[str],
        jac_outputs: list[str],
        run_id: str | None = None,
    ) -> dict:
        """Calculate the Jacobian of (some of the) outputs w.r.t. (some of the) inputs.

        Args:
            inputs: a dictionary with the inputs.
            jac_inputs: Inputs with respect to which derivatives will be calculated.
            jac_outputs: Outputs which will be differentiated.
            run_id: a string to identify the run. Run outputs will be located
                    in a directory suffixed with this id.

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
        return self._client.run_tesseract(
            "jacobian", payload, run_id, self._stream_logs
        )

    @requires_client
    def jacobian_vector_product(
        self,
        inputs: dict,
        jvp_inputs: list[str],
        jvp_outputs: list[str],
        tangent_vector: dict,
        run_id: str | None = None,
    ) -> dict:
        """Calculate the Jacobian Vector Product (JVP) of (some of the) outputs w.r.t. (some of the) inputs.

        Args:
            inputs: a dictionary with the inputs.
            jvp_inputs: Inputs with respect to which derivatives will be calculated.
            jvp_outputs: Outputs which will be differentiated.
            tangent_vector: Element of the tangent space to multiply with the Jacobian.
            run_id: a string to identify the run. Run outputs will be located
                    in a directory suffixed with this id.

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
        return self._client.run_tesseract(
            "jacobian_vector_product", payload, run_id, self._stream_logs
        )

    @requires_client
    def vector_jacobian_product(
        self,
        inputs: dict,
        vjp_inputs: list[str],
        vjp_outputs: list[str],
        cotangent_vector: dict,
        run_id: str | None = None,
    ) -> dict:
        """Calculate the Vector Jacobian Product (VJP) of (some of the) outputs w.r.t. (some of the) inputs.

        Args:
            inputs: a dictionary with the inputs.
            vjp_inputs: Inputs with respect to which derivatives will be calculated.
            vjp_outputs: Outputs which will be differentiated.
            cotangent_vector: Element of the cotangent space to multiply with the Jacobian.
            run_id: a string to identify the run. Run outputs will be located
                    in a directory suffixed with this id.

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
        return self._client.run_tesseract(
            "vector_jacobian_product", payload, run_id, self._stream_logs
        )

    @requires_client
    def test(self, test_spec: dict) -> None:
        """Run a regression test, raising AssertionError on failure.

        Works in LocalClient, HTTPClient and remote if served in debug mode.

        Args:
            test_spec: Test specification dict with keys:
                - endpoint: Name of endpoint (e.g., "apply", "jacobian")
                - payload: Input data dict
                - expected_outputs: Expected output data dict (if no exception expected)
                - expected_exception: Optional exception type or name (e.g., ValueError or "ValueError")
                - expected_exception_regex: Optional regex pattern for exception message
                - atol: Optional absolute tolerance (default 1e-8)
                - rtol: Optional relative tolerance (default 1e-5)

            Must provide exactly one of expected_outputs or expected_exception.

        Raises:
            AssertionError: If test fails (outputs don't match or wrong exception)
            RuntimeError: If test encounters unexpected error

        Example:
            >>> tess = Tesseract.from_tesseract_api("path/to/tesseract_api.py")
            >>> tess.test(
            ...     {
            ...         "endpoint": "apply",
            ...         "payload": {"a": [1, 2], "b": [3, 4]},
            ...         "expected_outputs": {"result": [4, 6]},
            ...     }
            ... )
        """
        if "test" not in self.available_endpoints:
            raise NotImplementedError(
                "Test endpoint not available, to expose this Tesseracts must be served in debug mode."
            )

        result = self._client.run_tesseract("test", test_spec, run_id=None)

        # Re-raise errors for pytest compatibility
        if result["status"] == "failed":
            raise AssertionError(result["message"])
        elif result["status"] == "error":
            raise RuntimeError(result["message"])


def _tree_map(func: Callable, tree: Any, is_leaf: Callable | None = None) -> Any:
    """Recursively apply a function to all leaves of a tree-like structure."""
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


def _fast_tobytes(arr: np.ndarray) -> memoryview:
    """Convert a numpy array to bytes without copying if possible."""
    return np.ascontiguousarray(arr).data


def _encode_array(arr: np.ndarray, b64: bool = True) -> dict:
    if b64:
        data = {
            "buffer": pybase64.b64encode_as_string(_fast_tobytes(arr)),
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


def _decode_array(
    encoded_arr: dict, output_path: str | Path | None = None
) -> np.ndarray:
    import re

    if "data" not in encoded_arr:
        raise ValueError("Encoded array does not contain 'data' key. Cannot decode.")

    encoding = encoded_arr["data"]["encoding"]
    dtype = np.dtype(encoded_arr["dtype"])
    shape = tuple(encoded_arr["shape"])

    if encoding == "base64":
        data = pybase64.b64decode(encoded_arr["data"]["buffer"])
        arr = np.frombuffer(data, dtype=dtype)
    elif encoding in ["json", "raw"]:
        arr = np.array(encoded_arr["data"]["buffer"], dtype=dtype)
    elif encoding == "binref":
        buffer_spec = encoded_arr["data"]["buffer"]
        # Parse the buffer spec which has format: path[:offset]
        path_match = re.match(r"^(?P<path>.+?)(\:(?P<offset>\d+))?$", buffer_spec)
        if not path_match:
            raise ValueError(
                f"Invalid binref path format: {buffer_spec}. "
                "Expected format is '<path>[:<offset>]'."
            )
        bufferpath = path_match.group("path")
        offset = int(path_match.group("offset") or 0)

        # Calculate the number of bytes to read
        size = 1 if len(shape) == 0 else int(np.prod(shape))
        num_bytes = size * dtype.itemsize

        # Resolve the path
        if output_path is not None:
            full_path = Path(output_path) / bufferpath
        else:
            full_path = Path(bufferpath)

        if not full_path.exists():
            raise ValueError(
                f"Binary file not found: {full_path}. "
                "Make sure output_path is set when using json+binref encoding."
            )

        # Read the binary data
        with open(full_path, "rb") as f:
            f.seek(offset)
            data = f.read(num_bytes)

        arr = np.frombuffer(data, dtype=dtype)
    else:
        raise ValueError(f"Unexpected array encoding {encoding}. Cannot decode.")

    arr = arr.reshape(shape)
    return arr


class HTTPClient:
    """HTTP Client for Tesseracts."""

    def __init__(self, url: str, output_path: str | Path | None = None) -> None:
        self._url = self._sanitize_url(url)
        self._output_path = output_path
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

    @staticmethod
    def _sanitize_url(url: str) -> str:
        parsed = urlparse(url)

        if not parsed.scheme:
            url = f"http://{url}"
            parsed = urlparse(url)

        sanitized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
        sanitized = sanitized.rstrip("/")
        return sanitized

    @property
    def url(self) -> str:
        """(Sanitized) URL to connect to."""
        return self._url

    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        payload: dict | None = None,
        run_id: str | None = None,
    ) -> dict:
        url = f"{self.url}/{endpoint.lstrip('/')}"

        if payload:
            encoded_payload = _tree_map(
                _encode_array, payload, is_leaf=lambda x: hasattr(x, "shape")
            )
        else:
            encoded_payload = None

        params = {"run_id": run_id} if run_id is not None else {}
        data = orjson.dumps(encoded_payload)
        try:
            response = self._session.request(
                method=method, url=url, data=data, params=params
            )
        except requests.ConnectionError:
            # Retry once on stale keep-alive connections. There is a race
            # between urllib3's is_connection_dropped check and the server
            # closing idle connections (uvicorn timeout_keep_alive) that
            # can cause ConnectionError on an otherwise healthy server.
            response = self._session.request(
                method=method, url=url, data=data, params=params
            )

        if response.status_code == requests.codes.unprocessable_entity:
            # Try and raise a more helpful error if the response is a Pydantic error
            try:
                data = from_json(response.content)
            except requests.JSONDecodeError:
                # Is not a Pydantic error
                data = {}
            if "detail" in data:
                errors = []
                for e in data["detail"]:
                    error = InitErrorDetails(
                        type=PydanticCustomError(
                            e["type"],
                            e.get("msg", ""),
                            e.get("ctx"),
                        ),
                        loc=tuple(e["loc"]),
                        input=e.get("input"),
                    )
                    errors.append(error)

                raise ValidationError.from_exception_data(
                    f"endpoint {endpoint}", line_errors=errors
                )

        if not response.ok:
            raise RuntimeError(
                f"Error {response.status_code} from Tesseract: {response.text}"
            )

        data = from_json(response.content)

        if endpoint in [
            "apply",
            "jacobian",
            "jacobian_vector_product",
            "vector_jacobian_product",
        ]:
            # Create a decoder with the output_path bound
            def decode_with_path(arr: dict) -> np.ndarray:
                return _decode_array(arr, output_path=self._output_path)

            data = _tree_map(
                decode_with_path,
                data,
                is_leaf=lambda x: type(x) is dict and "shape" in x,
            )

        return data

    def run_tesseract(
        self,
        endpoint: str,
        payload: dict | None = None,
        run_id: str | None = None,
        stream_logs: BoolOrCallable = False,
    ) -> dict:
        """Run a Tesseract endpoint.

        Args:
            endpoint: The endpoint to run.
            payload: The payload to send to the endpoint.
            run_id: a string to identify the run. Run outputs will be located
                    in a directory suffixed with this id.
            stream_logs: If True, stream logs to stdout. If a callable, stream
                    logs to that callable.

        Returns:
            The loaded JSON response from the endpoint, with decoded arrays.
        """
        if endpoint in [
            "openapi_schema",
            "health",
        ]:
            method = "GET"
        else:
            method = "POST"

        if endpoint == "openapi_schema":
            endpoint = "openapi.json"

        # Set up log streaming if requested
        log_streamer = None
        if stream_logs:
            # Generate run_id if not provided so we know the log file path
            if run_id is None:
                run_id = str(uuid.uuid4())

            # output_path is always set by from_image (uses temp dir if not specified)
            assert self._output_path is not None
            log_path = self._output_path / f"run_{run_id}" / "logs" / "tesseract.log"

            # Determine log sink from stream_logs parameter
            if callable(stream_logs):
                log_sink = stream_logs
            elif stream_logs is True:
                log_sink = lambda msg: print(msg, file=sys.stderr, flush=True)
            else:
                raise ValueError(
                    f"Invalid value for stream_logs: {stream_logs}. Must be True, False, or a callable."
                )
            log_streamer = LogStreamer(log_path, log_sink)
            log_streamer.start()

        try:
            return self._request(endpoint, method, payload, run_id)
        finally:
            if log_streamer is not None:
                log_streamer.stop()


class LocalClient:
    """Local Client for Tesseracts."""

    def __init__(
        self, tesseract_api: ModuleType, output_path: Path | None = None
    ) -> None:
        # Import here to not depend on runtime dependencies globally
        from tesseract_core.runtime.core import create_endpoints
        from tesseract_core.runtime.serve import create_rest_api

        self._endpoints = {
            func.__name__: func for func in create_endpoints(tesseract_api)
        }
        self._openapi_schema = create_rest_api(tesseract_api).openapi()

        if output_path is None:
            output_path = Path(tempfile.mkdtemp(prefix="tesseract_output_"))
        self._output_path = output_path

    def run_tesseract(
        self,
        endpoint: str,
        payload: dict | None = None,
        run_id: str | None = None,
        stream_logs: BoolOrCallable = False,
    ) -> dict:
        """Run a Tesseract endpoint.

        Args:
            endpoint: The endpoint to run.
            payload: The payload to send to the endpoint.
            run_id: a string to identify the run.
            stream_logs: If True, stream logs to stdout. If a callable, stream logs to that callable.

        Returns:
            The loaded JSON response from the endpoint, with decoded arrays.
        """
        if endpoint == "openapi_schema":
            return self._openapi_schema

        if endpoint not in self._endpoints:
            raise RuntimeError(f"Endpoint {endpoint} not found in Tesseract API.")

        # Import here to not depend on runtime dependencies globally
        from tesseract_core.runtime.config import get_config
        from tesseract_core.runtime.file_interactions import join_paths
        from tesseract_core.runtime.mpa import start_run
        from tesseract_core.runtime.profiler import Profiler

        func = self._endpoints[endpoint]
        InputSchema = func.__annotations__.get("payload", None)
        OutputSchema = func.__annotations__.get("return", None)

        if InputSchema is not None:
            parsed_payload = InputSchema.model_validate(payload)
        else:
            parsed_payload = None

        # Set up run directory for logging
        if run_id is None:
            run_id = str(uuid.uuid4())
        rundir = join_paths(str(self._output_path), f"run_{run_id}")

        # Determine log sink from stream_logs parameter
        if stream_logs is False:
            log_sink = None
        elif stream_logs is True:
            log_sink = lambda msg: print(msg, file=sys.stderr, flush=True)
        elif callable(stream_logs):
            log_sink = stream_logs
        else:
            raise ValueError(
                f"Invalid value for stream_logs: {stream_logs}. Must be True, False, or a callable."
            )

        # Set up profiler
        profiler = Profiler(enabled=get_config().profiling)

        try:
            with start_run(base_dir=rundir, log_sink=log_sink):
                with profiler:
                    if parsed_payload is not None:
                        result = self._endpoints[endpoint](parsed_payload)
                    else:
                        result = self._endpoints[endpoint]()

                # Print profiling stats inside start_run context
                # so they go through stdio redirection to the configured sink
                profiler.print_stats()
        except Exception as ex:
            # Some clients like Tesseract-JAX swallow tracebacks from re-raised exceptions, so we explicitly
            # format the traceback here to include it in the error message.
            tb = traceback.format_exc()
            raise RuntimeError(
                f"{tb}\nError running Tesseract API {endpoint}: {ex} (see above for full traceback)"
            ) from None

        if OutputSchema is not None:
            # Validate via schema, then dump to stay consistent with other clients
            if isinstance(OutputSchema, type) and issubclass(OutputSchema, BaseModel):
                result = OutputSchema.model_validate(result).model_dump()
            else:
                result = TypeAdapter(OutputSchema).validate_python(result)

        return result
