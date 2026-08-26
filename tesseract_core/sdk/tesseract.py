# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
import uuid
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import cached_property, wraps
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias
from urllib.parse import urlparse, urlunparse

import numpy as np
import orjson
import pybase64
import requests
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError, from_json

from . import engine, local_engine, serving
from .binref import (
    CONTAINERS_SUPPORT_BINREF_POOL,
    BinrefSlot,
    BinrefWritePool,
    _fast_tobytes,
    encode_array_binref,
    encode_array_binref_pooled,
    mmap_binref_array,
    read_binref_array,
)
from .docker_client import Container
from .logs import LogStreamer
from .served_client import ServedTesseract

if TYPE_CHECKING:
    # Imported for type hints only. `from __future__ import annotations` makes
    # every annotation below a string, so these names are never needed at
    # runtime and the SDK does not eagerly pull in the runtime/CUDA machinery.
    from tesseract_core.runtime.cuda_ipc import IpcDeviceArray

# Output serialization formats; single SDK-side definition lives in engine.
OutputFormat: TypeAlias = engine.OutputFormat

PathLike: TypeAlias = str | Path
BoolOrCallable: TypeAlias = bool | Callable[[str], Any]


def _purge_tempdir(path: str) -> None:
    """Remove an auto-created output tempdir. Used as a weakref finalizer.

    Errors are ignored: the dir may already be gone, and a finalizer must never
    raise (it can run at interpreter shutdown).
    """
    shutil.rmtree(path, ignore_errors=True)


def requires_client(func: Callable) -> Callable:
    """Decorator to require a client for a Tesseract instance."""

    @wraps(func)
    def wrapper(self: Tesseract, *args: Any, **kwargs: Any) -> Any:
        if not self._client:
            if self._spawn_backend == "subprocess":
                constructor = "from_source"
            else:
                constructor = "from_image"
            raise RuntimeError(
                f"When creating a {self.__class__.__name__} via `{constructor}`, "
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
    # Which engine `serve()` should hand `_spawn_config` to. "container" rather
    # than "docker" because any docker-compatible CLI works here (see
    # `is_podman`), matching how `container_info` and `Container` are named.
    _spawn_backend: Literal["container", "subprocess"] | None = None
    _serve_context: ServedTesseract | None = None
    _lastlog: str | None = None
    _client: HTTPClient | LocalClient | None = None
    _stream_logs: BoolOrCallable = False
    _timeout: float | tuple[float, float] | None = None
    _binref_pool_enabled: bool = False

    def __init__(
        self,
        url: str,
        server_output_path: str | Path | None = None,
        timeout: float | tuple[float, float] | None = None,
    ) -> None:
        warnings.warn(
            "Direct instantiation of Tesseract is deprecated. "
            "Use Tesseract.from_url(), Tesseract.from_image(), or Tesseract.from_tesseract_api() instead.",
            UserWarning,
            stacklevel=2,
        )
        self._client = HTTPClient(url, output_path=server_output_path, timeout=timeout)

    @classmethod
    def from_url(
        cls,
        url: str,
        server_output_path: str | Path | None = None,
        timeout: float | tuple[float, float] | None = None,
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
            timeout: Request timeout in seconds. Can be a float for both connect and
                read timeouts, or a ``(connect, read)`` tuple for separate control.
                ``None`` (the default) disables timeouts. See the `requests documentation
                <https://requests.readthedocs.io/en/latest/user/advanced/#timeouts>`_
                for details.

        Returns:
            A Tesseract instance.
        """
        obj = cls.__new__(cls)
        obj._client = HTTPClient(url, output_path=server_output_path, timeout=timeout)
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
        output_format: OutputFormat = "json+base64",
        docker_args: list[str] | None = None,
        runtime_config: dict[str, Any] | None = None,
        stream_logs: BoolOrCallable = False,
        skip_health_check: bool = False,
        startup_timeout: float = serving.DEFAULT_STARTUP_TIMEOUT,
        timeout: float | tuple[float, float] | None = None,
        experimental_binref_pool: bool = False,
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
            skip_health_check: If True, skip the startup health check poll. Useful for
                Tesseracts with slow initialization (e.g., Julia runtime startup, large
                model loading). The caller is responsible for ensuring
                readiness, e.g. by calling :meth:`health`, before calling
                other endpoints.
            startup_timeout: How long to wait, in seconds, for the Tesseract to
                answer a health check. Raise it for one that is slow to
                initialize, in preference to skipping the check altogether.
            timeout: Request timeout in seconds for HTTP calls to the Tesseract.
                Can be a float for both connect and read timeouts, or a
                ``(connect, read)`` tuple for separate control. ``None`` (the default)
                disables timeouts. See the `requests documentation
                <https://requests.readthedocs.io/en/latest/user/advanced/#timeouts>`_
                for details.
            experimental_binref_pool: Opt-in fast path for ``json+binref`` that
                only makes sense when ``input_path`` and ``output_path`` point at
                a shared-memory tmpfs (``/dev/shm`` on Linux). Reuses a small pool
                of pre-faulted, memory-mapped input buffers instead of writing a
                fresh file per request, and decodes outputs as zero-copy
                memory-mapped views instead of eager copies. Linux only (raises on
                other platforms), since elsewhere the container runs in a VM and
                does not share a page cache with the client.

        Returns:
            A Tesseract instance.
        """
        obj = cls.__new__(cls)

        if environment is None:
            environment = {}

        if volumes is None:
            volumes = []
        auto_input_path = False
        if input_path is not None:
            input_path = Path(input_path).resolve()
        elif output_format == "json+binref":
            # Auto-create an input directory so binref-encoded inputs have a
            # mounted location to be written to and read from by the container.
            input_path = Path(tempfile.mkdtemp(prefix="tesseract_input_"))
            auto_input_path = True

        auto_output_path = output_path is None
        if output_path is not None:
            output_path = Path(output_path).resolve()
        else:
            # Auto-create temp directory for output (enables stream_logs without explicit output_path)
            output_path = Path(tempfile.mkdtemp(prefix="tesseract_output_"))

        obj._stream_logs = stream_logs
        obj._timeout = timeout
        if experimental_binref_pool and not CONTAINERS_SUPPORT_BINREF_POOL:
            raise RuntimeError(
                "experimental_binref_pool=True is only supported for containerized "
                "Tesseracts on Linux, since it relies on the client and the "
                "container sharing a page cache. Elsewhere the container runs "
                "inside a VM, so bind mounts cross the VM boundary and the premise "
                "does not hold. `from_source` has no VM to cross and is unrestricted."
            )
        obj._binref_pool_enabled = experimental_binref_pool
        # Purge auto-created tempdirs when the object is garbage collected.
        # User-supplied paths are left untouched.
        if auto_input_path:
            weakref.finalize(obj, _purge_tempdir, str(input_path))
        if auto_output_path:
            weakref.finalize(obj, _purge_tempdir, str(output_path))
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
            skip_health_check=skip_health_check,
            startup_timeout=startup_timeout,
        )
        obj._spawn_backend = "container"
        return obj

    @classmethod
    def from_tesseract_api(
        cls,
        tesseract_api: str | Path | ModuleType,
        input_path: Path | None = None,
        output_path: Path | None = None,
        output_format: OutputFormat = "json+base64",
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

    @classmethod
    def from_source(
        cls,
        tesseract_api: str | Path,
        input_path: Path | None = None,
        output_path: Path | None = None,
        output_format: Literal["json", "json+base64", "json+binref"] = "json+base64",
        runtime_config: dict[str, Any] | None = None,
        stream_logs: BoolOrCallable = False,
        python_executable: str | Path | None = None,
        startup_timeout: float = serving.DEFAULT_STARTUP_TIMEOUT,
        experimental_binref_pool: bool = False,
    ) -> Tesseract:
        """Create a Tesseract instance from a Tesseract API file, in its own process.

        The Tesseract is served by a dedicated ``tesseract-runtime serve``
        subprocess and reached over HTTP, so it does not share an interpreter,
        global state or signal handlers with the caller. That matters when
        sharing them is unsafe -- nesting JAX inside JAX can deadlock -- and it
        lets the Tesseract run in a different environment than the caller.

        Unlike :meth:`from_tesseract_api`, which imports the API into this
        process, this must be used as a context manager or served explicitly,
        since there is a process to clean up:

            >>> with Tesseract.from_source("tesseract_api.py") as tess:
            ...     tess.apply({"a": 1})

        This is not a substitute for a container: the Tesseract inherits this
        process's environment, working directory, filesystem access and user.

        Args:
            tesseract_api: Path to the `tesseract_api.py` file. Unlike
                :meth:`from_tesseract_api`, an already imported module cannot be
                used, since it cannot be shared with another process.
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
            python_executable: Interpreter used to run the Tesseract. Defaults to
                the one running this process; point it at another environment's
                ``python`` (for example one created with ``uv venv``) to give the
                Tesseract dependencies that conflict with the caller's. That
                environment must have ``tesseract-core[runtime]`` and the
                Tesseract's own requirements installed.
            startup_timeout: How long to wait, in seconds, for the Tesseract to
                become healthy before giving up.
            experimental_binref_pool: Opt-in fast path for ``json+binref`` that
                reuses warm memory-mapped buffers instead of allocating a file
                per call. Only pays off when the binref directory is
                memory-backed (a ``tmpfs``); on an ordinary disk-backed
                filesystem it is several times *slower* than plain
                ``json+binref``. See :doc:`/content/how-to/fast-local-runs`.

        Returns:
            A Tesseract instance.
        """
        obj = cls.__new__(cls)
        obj._stream_logs = stream_logs
        obj._spawn_backend = "subprocess"
        obj._binref_pool_enabled = experimental_binref_pool
        auto_dirs, obj._spawn_config = _subprocess_spawn_config(
            tesseract_api,
            input_path=input_path,
            output_path=output_path,
            output_format=output_format,
            runtime_config=runtime_config,
            python_executable=python_executable,
            startup_timeout=startup_timeout,
        )
        # Purge auto-created scratch dirs when the object is garbage collected.
        for scratch in auto_dirs:
            weakref.finalize(obj, _purge_tempdir, str(scratch))
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

    def __exit__(self, *args: object) -> None:
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
                "Can only retrieve logs for a Tesseract created via `from_image` "
                "or `from_source`."
            )
        if self._serve_context is None:
            return self._lastlog or ""
        return self._serve_context.logs().decode("utf-8", errors="replace")

    def serve(self) -> None:
        """Serve the Tesseract until it is stopped."""
        if self._spawn_config is None:
            raise RuntimeError(
                "Can only serve a Tesseract created via `from_image` or `from_source`."
            )
        if self._serve_context is not None:
            raise RuntimeError("Tesseract is already being served.")

        # The only part that has to know which backend it is: what to start.
        if self._spawn_backend == "subprocess":
            self._serve_context = local_engine.serve(**self._spawn_config)
        else:
            _, self._serve_context = engine.serve(**self._spawn_config)

        # Ensure that the Tesseract is torn down once the object is garbage
        # collected, to avoid orphaned containers or processes if the user
        # forgets to call .teardown()
        def _silent_teardown(handle: ServedTesseract) -> None:
            from tesseract_core.sdk.docker_client import NotFound

            try:
                handle.teardown()
            except NotFound:
                pass

        reap = (_silent_teardown, self._serve_context)
        url = self._serve_context.url

        self._lastlog = None
        output_path = self._spawn_config.get("output_path")
        input_path = self._spawn_config.get("input_path")
        output_format = self._spawn_config.get("output_format", "json+base64")
        self._client = HTTPClient(
            url,
            output_path=Path(output_path) if output_path else None,
            output_format=output_format,
            timeout=self._timeout,
            input_path=Path(input_path) if input_path else None,
            experimental_binref_pool=self._binref_pool_enabled,
        )

        self._atexit_finalizer = weakref.finalize(self, *reap)

    def teardown(self) -> None:
        """Teardown the Tesseract.

        This will stop and remove the Tesseract container, or stop the dedicated
        process serving it.
        """
        if self._serve_context is None:
            raise RuntimeError("Tesseract is not being served.")
        self._lastlog = self.server_logs()
        self._serve_context.teardown()
        if self._client is not None:
            self._client.close()
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

    def container_info(self) -> Container:
        """Retrieve information on the Docker container serving this Tesseract.

        Tesseract must be created via `from_image` and be actively served for
        this to be available.

        Raises:
            RuntimeError: if this Tesseract was not created via
                :meth:`from_image` (e.g. :meth:`from_url` or
                :meth:`from_tesseract_api`), or if it is not currently
                being served (call :meth:`serve` or use ``with tess:``
                first).
            tesseract_core.sdk.docker_client.NotFound: if the container
                disappeared between :meth:`serve` and this call.
        """
        if self._spawn_backend != "container":
            raise RuntimeError(
                "`container_info` is only available when using "
                "`Tesseract.from_image(...)`."
            )
        if self._serve_context is None:
            raise RuntimeError(
                "`container_info` is only available for served Tesseracts. "
                "Use `tess.serve()` or `with tess:` first."
            )
        return self._serve_context

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


def _subprocess_spawn_config(
    tesseract_api: str | Path | ModuleType,
    *,
    input_path: Path | None,
    output_path: Path | None,
    output_format: Literal["json", "json+base64", "json+binref"],
    runtime_config: dict[str, Any] | None,
    python_executable: str | Path | None,
    startup_timeout: float,
) -> tuple[list[Path], dict[str, Any]]:
    """Validate arguments for a dedicated-process Tesseract and build its config.

    Unlike the in-process path, nothing here touches this process's runtime
    config: it all reaches the child as environment variables, so several
    Tesseracts can be configured independently.
    """
    if not isinstance(tesseract_api, str | Path):
        raise ValueError(
            "`from_source` requires a path to a `tesseract_api.py` file, but an "
            f"already imported module was given "
            f"({getattr(tesseract_api, '__name__', tesseract_api)!r}). A module "
            "cannot be shared with another process; pass `module.__file__`, or "
            "use `from_tesseract_api` to run it in this one."
        )

    tesseract_api_path = Path(tesseract_api).resolve(strict=True)
    if not tesseract_api_path.is_file():
        raise RuntimeError(f"Tesseract API path {tesseract_api_path} is not a file.")

    # Scratch directories, as `from_image` does it: an output directory always
    # exists, which is what makes `stream_logs` work without the caller
    # specifying one, and binref additionally needs somewhere to put its inputs.
    # Auto-created ones are purged with the Tesseract; given ones are left alone.
    auto_dirs = []
    if output_path is not None:
        resolved_output_path = engine._resolve_file_path(output_path, make_dir=True)
    else:
        resolved_output_path = Path(tempfile.mkdtemp(prefix="tesseract_output_"))
        auto_dirs.append(resolved_output_path)

    if input_path is not None:
        resolved_input_path = Path(input_path).resolve()
    elif output_format == "json+binref":
        resolved_input_path = Path(tempfile.mkdtemp(prefix="tesseract_input_"))
        auto_dirs.append(resolved_input_path)
    else:
        resolved_input_path = None

    # Debug mode gives full tracebacks from the child and enables the `test`
    # endpoint, matching what the in-process path configures. The debugpy
    # listener it would normally imply is disabled separately, in
    # `local_engine.serve`.
    config_kwargs: dict[str, Any] = {"debug": True}
    if runtime_config is not None:
        config_kwargs.update(runtime_config)

    return auto_dirs, dict(
        api_path=tesseract_api_path,
        input_path=resolved_input_path,
        output_path=resolved_output_path,
        output_format=output_format,
        runtime_config=config_kwargs,
        python_executable=python_executable,
        startup_timeout=startup_timeout,
    )


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


def _import_cuda_ipc() -> ModuleType:
    """Import the cuda_ipc runtime module, or explain the missing extra.

    The ``json+cuda_ipc`` output format lives in ``tesseract_core.runtime``,
    which is an optional install (``tesseract-core[runtime]``). A base SDK
    install lacks its dependencies, so surface a clear message pointing at the
    extra instead of a bare ``ModuleNotFoundError`` from deep in the import chain.
    """
    try:
        from tesseract_core.runtime import cuda_ipc
    except ImportError as exc:
        raise ImportError(
            "The 'json+cuda_ipc' output format requires the Tesseract runtime, "
            "which is an optional dependency. Install it with "
            "'pip install tesseract-core[runtime]'."
        ) from exc
    return cuda_ipc


def _encode_array(
    arr: Any, encoding: Literal["base64", "raw", "cuda_ipc"] = "base64"
) -> dict:
    # With cuda_ipc encoding, GPU arrays are exported by reference via a CUDA IPC
    # handle, keeping the data on-device. Any other array (or any other encoding)
    # falls through to a host copy below, so a mixed payload (some GPU, some CPU
    # arrays) encodes correctly either way.
    if encoding == "cuda_ipc" and hasattr(arr, "__cuda_array_interface__"):
        return _import_cuda_ipc().dump_cuda_ipc_arraydict(arr)

    # Ensure arr is a numpy-compatible array so we guarantee it has a compatible dtype (not e.g. torch bfloat16)
    arr = np.asanyarray(arr, order="A")
    if encoding == "raw":
        data = {
            "buffer": arr.tolist(),
            "encoding": "raw",
        }
    else:
        # base64 (also the host-copy fallback for a CPU array under cuda_ipc)
        data = {
            "buffer": pybase64.b64encode_as_string(_fast_tobytes(arr)),
            "encoding": "base64",
        }

    return {
        "shape": arr.shape,
        "dtype": arr.dtype.name,
        "data": data,
    }


@contextmanager
def _encode_payload(payload: dict | None, output_format: str) -> Iterator[dict | None]:
    """Encode a request payload's arrays, managing CUDA IPC export lifetime.

    Yields the encoded payload (or None for an empty payload). For the
    ``json+cuda_ipc`` format, GPU arrays are exported by reference (base64 for
    CPU arrays), which pins each exported allocation in a process-global registry
    on the runtime side. Those pins are released on context exit -- by then the
    caller has read the full response, so the server has copied the inputs out
    and they are provably dead. The release is skipped (and cuda_ipc never
    imported) when no GPU array was actually exported.

    Releasing on exit rather than at the start of the next request keeps pinned
    GPU memory bounded to a single in-flight request.
    """
    if not payload:
        yield None
        return

    if output_format != "json+cuda_ipc":
        yield _tree_map(
            _encode_array, payload, is_leaf=lambda x: hasattr(x, "__array__")
        )
        return

    # cuda_ipc: a leaf is any array-like on either protocol; GPU leaves are
    # exported by handle and pin their allocation until we release below.
    exported = False

    def _encode_leaf(x: Any) -> dict:
        nonlocal exported
        if hasattr(x, "__cuda_array_interface__"):
            exported = True
        return _encode_array(x, encoding="cuda_ipc")

    def _is_leaf(x: Any) -> bool:
        return hasattr(x, "__array__") or hasattr(x, "__cuda_array_interface__")

    try:
        yield _tree_map(_encode_leaf, payload, is_leaf=_is_leaf)
    finally:
        if exported:
            _import_cuda_ipc().release_pinned_ipc_exports()


def _decode_array(
    encoded_arr: dict,
    output_path: str | Path | None = None,
    lazy: bool = False,
    mapped_paths: list[Path] | None = None,
) -> np.ndarray | IpcDeviceArray:
    """Decode an encoded array dict into a numpy array.

    When ``lazy`` is set and the array is decoded as a zero-copy mmap view, the
    backing file path is appended to ``mapped_paths`` (if given) so the caller
    can unlink it once the whole response is decoded. The mmap keeps the inode
    alive after unlink, so the returned view stays valid.

    Returns np.ndarray for every encoding except cuda_ipc, which yields a
    framework-agnostic on-GPU wrapper (IpcDeviceArray, exposing
    __cuda_array_interface__ and __dlpack__). That type is imported only under
    TYPE_CHECKING so naming it here adds no runtime import.
    """
    import re

    if "data" not in encoded_arr:
        raise ValueError("Encoded array does not contain 'data' key. Cannot decode.")

    encoding = encoded_arr["data"]["encoding"]
    dtype = np.dtype(encoded_arr["dtype"])
    shape = tuple(encoded_arr["shape"])

    if encoding == "base64":
        data = pybase64.b64decode(encoded_arr["data"]["buffer"])
        compression = encoded_arr["data"].get("compression")
        if compression == "lz4":
            import lz4.frame

            data = lz4.frame.decompress(data)
        elif compression is not None:
            raise ValueError(f"Unknown compression: {compression}")
        arr = np.frombuffer(data, dtype=dtype)
    elif encoding in ["json", "raw"]:
        arr = np.array(encoded_arr["data"]["buffer"], dtype=dtype)
    elif encoding == "binref":
        buffer_spec = encoded_arr["data"]["buffer"]
        # Parse the buffer spec which has format: path[:offset[:compressed_size]]
        path_match = re.match(
            r"^(?P<path>.+?)(\:(?P<offset>\d+)(\:(?P<compressed_size>\d+))?)?$",
            buffer_spec,
        )
        if not path_match:
            raise ValueError(
                f"Invalid binref path format: {buffer_spec}. "
                "Expected format is '<path>[:<offset>[:<compressed_size>]]'."
            )
        bufferpath = path_match.group("path")
        offset = int(path_match.group("offset") or 0)
        compressed_size_str = path_match.group("compressed_size")

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

        compression = encoded_arr["data"].get("compression")

        if compression is None:
            count = 1 if len(shape) == 0 else size
            if num_bytes == 0:
                arr = np.frombuffer(b"", dtype=dtype)
            elif lazy:
                # Zero-copy read-only view (POSIX only, see caller gating).
                arr = mmap_binref_array(full_path, offset, num_bytes, dtype, count)
                if mapped_paths is not None:
                    mapped_paths.append(full_path)
            else:
                # Eager copy into an owned, writable array (portable default).
                arr = read_binref_array(full_path, offset, num_bytes, dtype, count)
        else:
            if compressed_size_str is None:
                raise ValueError(
                    "compressed_size missing from buffer spec when compression is set "
                    "(expected format: '<path>:<offset>:<compressed_size>')"
                )
            with open(full_path, "rb") as f:
                f.seek(offset)
                data = f.read(int(compressed_size_str))

            if compression == "lz4":
                import lz4.frame

                data = lz4.frame.decompress(data)
            else:
                raise ValueError(f"Unknown compression: {compression}")

            arr = np.frombuffer(data, dtype=dtype)
    elif encoding == "cuda_ipc":
        # Returns a fresh, client-owned device-array wrapper: the decode opens
        # the IPC handle, copies device-to-device into our own memory, and
        # closes the mapping before returning. The result exposes
        # __cuda_array_interface__ and __dlpack__ so Torch/JAX/CuPy can adopt it
        # zero-copy. The server may reuse/free the exported buffer as soon as
        # this returns (it holds it until the next request).
        return _import_cuda_ipc().load_cuda_ipc_arraydict(encoded_arr)
    else:
        raise ValueError(f"Unexpected array encoding {encoding}. Cannot decode.")

    arr = arr.reshape(shape)
    return arr


class HTTPClient:
    """HTTP Client for Tesseracts."""

    # Class-level defaults so instances built via ``__new__`` (e.g. in tests)
    # still expose the binref attributes the request/decode paths read.
    _input_path: Path | None = None
    _binref_pool: BinrefWritePool | None = None

    def __init__(
        self,
        url: str,
        output_path: str | Path | None = None,
        output_format: OutputFormat = "json+base64",
        timeout: float | tuple[float, float] | None = None,
        input_path: str | Path | None = None,
        experimental_binref_pool: bool = False,
    ) -> None:
        self._url = self._sanitize_url(url)
        self._output_path = output_path
        self._output_format = output_format
        self._input_path = Path(input_path) if input_path is not None else None
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"
        # Opt-in warm-buffer pool for binref inputs. Only meaningful when passing
        # inputs as binref into a mounted (ideally shared-memory) input dir.
        self._binref_pool: BinrefWritePool | None = None
        # Whether the pool can work at all depends on how the Tesseract is
        # served, which is not something a client reached over HTTP can know.
        # Whoever served it decides; this honours the decision.
        if experimental_binref_pool and self._input_path is not None:
            self._binref_pool = BinrefWritePool(self._input_path)

    def close(self) -> None:
        """Release resources held by the client (e.g. the binref write pool)."""
        if self._binref_pool is not None:
            self._binref_pool.close()
            self._binref_pool = None

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

    def _send(
        self, url: str, method: str, data: bytes, params: dict
    ) -> requests.Response:
        # Only forward timeout when set; omitting it is equivalent to None for
        # requests.Session, and avoids passing a kwarg that some session
        # implementations (e.g. starlette's TestClient) don't accept.
        request_kwargs: dict[str, Any] = {
            "method": method,
            "url": url,
            "data": data,
            "params": params,
        }
        if self._timeout is not None:
            request_kwargs["timeout"] = self._timeout
        try:
            return self._session.request(**request_kwargs)
        except requests.ConnectionError:
            # Retry once on stale keep-alive connections. There is a race between
            # urllib3's is_connection_dropped check and the server closing idle
            # connections (uvicorn timeout_keep_alive) that can cause
            # ConnectionError on an otherwise healthy server.
            return self._session.request(**request_kwargs)

    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        payload: dict | None = None,
        run_id: str | None = None,
    ) -> dict:
        url = f"{self.url}/{endpoint.lstrip('/')}"
        params = {"run_id": run_id} if run_id is not None else {}

        if payload and self._output_format == "json+binref" and self._input_path:
            # Pass input arrays as binref files in the mounted input directory
            # instead of base64-in-body. The server reads them via its input
            # path, so no array data travels over HTTP. Files (and any pooled
            # slots) live only until the response returns, so clean them up in a
            # finally once the server has read them.
            binref_input_files: list[Path] = []
            checked_out_slots: list[BinrefSlot] = []
            if self._binref_pool is not None:
                encode_binref = lambda x: encode_array_binref_pooled(
                    x, self._binref_pool, checked_out_slots, binref_input_files
                )
            else:
                encode_binref = lambda x: encode_array_binref(
                    x, self._input_path, binref_input_files
                )
            encoded_payload = _tree_map(
                encode_binref, payload, is_leaf=lambda x: hasattr(x, "__array__")
            )
            try:
                response = self._send(
                    url, method, orjson.dumps(encoded_payload), params
                )
                return self._decode_response(response, endpoint)
            finally:
                for f in binref_input_files:
                    f.unlink(missing_ok=True)
                if self._binref_pool is not None:
                    for slot in checked_out_slots:
                        self._binref_pool.checkin(slot)

        # Non-binref path: _encode_payload handles base64 and cuda_ipc, holding
        # any exported GPU inputs alive until the response has been fully read.
        # `requests` buffers the whole body before `_send` returns, so exiting
        # the block afterwards releases them at the earliest safe point.
        with _encode_payload(payload, self._output_format) as encoded_payload:
            response = self._send(url, method, orjson.dumps(encoded_payload), params)
        return self._decode_response(response, endpoint)

    def _decode_response(self, response: requests.Response, endpoint: str) -> dict:
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
            # Use the zero-copy lazy decode only on the opt-in fast path
            # (binref pool enabled), which requires POSIX (enforced at client
            # construction); otherwise decode eagerly into an owned array.
            lazy = self._binref_pool is not None

            # Files mapped by the lazy decode, unlinked once the whole response
            # is decoded so the server's output files don't accumulate. Each
            # returned view keeps its own mmap (and thus the inode) alive after
            # unlink, so the arrays stay valid; the space is reclaimed when the
            # user drops them. Unlinking eagerly per-array would break responses
            # where several arrays share one file at different offsets.
            mapped_paths: list[Path] = []

            def decode_with_path(arr: dict) -> np.ndarray | IpcDeviceArray:
                return _decode_array(
                    arr,
                    output_path=self._output_path,
                    lazy=lazy,
                    mapped_paths=mapped_paths,
                )

            data = _tree_map(
                decode_with_path,
                data,
                is_leaf=lambda x: type(x) is dict and "shape" in x,
            )

            for path in set(mapped_paths):
                path.unlink(missing_ok=True)

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
            # Purge the auto-created tempdir when this client is garbage collected.
            weakref.finalize(self, _purge_tempdir, str(output_path))
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
