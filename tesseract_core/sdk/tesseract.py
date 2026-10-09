# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import builtins
import collections
import os
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
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

from . import engine, local_client, serving
from .binref import (
    CONTAINERS_SUPPORT_BINREF_POOL,
    SUPPORTS_BINREF_POOL,
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
from .serving import ServedTesseract

if TYPE_CHECKING:
    # Imported for type hints only. `from __future__ import annotations` makes
    # every annotation below a string, so these names are never needed at
    # runtime and the SDK does not eagerly pull in the runtime/CUDA machinery.
    from tesseract_core.runtime.config import ConfigSnapshot
    from tesseract_core.runtime.cuda.ipc import IpcDeviceArray

# Output serialization formats; single SDK-side definition lives in engine.
OutputFormat: TypeAlias = engine.OutputFormat

PathLike: TypeAlias = str | Path
BoolOrCallable: TypeAlias = bool | Callable[[str], Any]


def _scratch_dirs(
    input_path: str | Path | None,
    output_path: str | Path | None,
    output_format: str,
) -> tuple[Path | None, Path, list[Path]]:
    """Work out the directories a served Tesseract reads and writes through.

    An output directory always exists, which is what lets `stream_logs` work
    without the caller naming one, and `json+binref` additionally needs somewhere
    to put its inputs.

    Returns:
        The input directory (None unless binref needs one), the output directory,
        and whichever of them were created here -- the caller purges those and
        leaves any it was given alone.
    """
    created = []

    if input_path is not None:
        resolved_input = Path(input_path).resolve()
    elif output_format == "json+binref":
        resolved_input = Path(tempfile.mkdtemp(prefix="tesseract_input_"))
        created.append(resolved_input)
    else:
        resolved_input = None

    if output_path is not None:
        resolved_output = engine._resolve_file_path(output_path, make_dir=True)
    else:
        resolved_output = Path(tempfile.mkdtemp(prefix="tesseract_output_"))
        created.append(resolved_output)

    return resolved_input, resolved_output, created


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


@dataclass(frozen=True)
class ServerCapabilities:
    """The encodings a served Tesseract accepts, as advertised by the server.

    Each field lists the values a client may request for that part of the
    encoding, or is None if the server does not advertise it. Runtimes older
    than 1.13 advertise nothing, and 1.13 and 1.14 advertise only output
    formats.
    """

    output_formats: tuple[str, ...] | None
    """Formats for CPU arrays in responses (e.g. ``json+base64``)."""

    gpu_transports: tuple[str, ...] | None
    """How GPU arrays may cross the boundary in either direction. ``none`` copies
    them to the host, and anything else (e.g. ``cuda_ipc``) passes them by
    reference."""

    compressions: tuple[str, ...] | None
    """Compressions for array buffers in responses (``none`` disables it)."""

    @classmethod
    def from_openapi_schema(cls, schema: dict) -> ServerCapabilities:
        """Read the capabilities a server advertises in its OpenAPI schema."""

        def advertised(key: str) -> tuple[str, ...] | None:
            values = schema.get(key)
            return None if values is None else tuple(values)

        return cls(
            output_formats=advertised("x-supported-output-formats"),
            gpu_transports=advertised("x-supported-gpu-transports"),
            compressions=advertised("x-supported-compressions"),
        )


@dataclass(frozen=True)
class RequestedEncoding:
    """The encoding calls through a Tesseract request (see :attr:`Tesseract.current_encoding`).

    Each field is the value calls request, or None if they leave it to the
    server's default.
    """

    output_format: str | None = None
    gpu_transport: str | None = None
    compression: str | None = None

    def merge(self, overrides: RequestedEncoding | None) -> RequestedEncoding:
        """Return a copy with the fields ``overrides`` sets taking precedence."""
        if overrides is None:
            return self
        return RequestedEncoding(
            output_format=overrides.output_format or self.output_format,
            gpu_transport=overrides.gpu_transport or self.gpu_transport,
            compression=overrides.compression or self.compression,
        )

    @property
    def params(self) -> dict[str, str]:
        """The fields sent as media-type parameters, for those with a preference."""
        return {
            name: value
            for name in ("gpu_transport", "compression")
            if (value := getattr(self, name)) is not None
        }

    def accept_header(self) -> str | None:
        """The ``Accept`` value requesting this encoding, or None if there is no preference.

        Fields without a preference are left out, so the server picks them.
        """
        params = [f"{name}={value}" for name, value in self.params.items()]
        if self.output_format is None and not params:
            return None
        media_type = f"application/{self.output_format or '*'}"
        return "; ".join([media_type, *params])


def _fit_encoding_to_server(
    encoding: RequestedEncoding, capabilities: ServerCapabilities
) -> RequestedEncoding:
    """Check ``encoding`` against what the server advertises, before any work is done.

    Raises ValueError if the server cannot provide the encoding. Returns the
    encoding to request, which drops parameters the server cannot parse but
    already satisfies.
    """
    if capabilities.output_formats is None:
        # Runtimes older than 1.13 read the entire Accept value as the output
        # format, so any parameter fails the request after the endpoint has run.
        # They never pass GPU arrays by reference, though, so gpu_transport=none
        # can go unsaid.
        if encoding.gpu_transport == "none":
            encoding = replace(encoding, gpu_transport=None)
        if encoding.params:
            names = " or ".join(encoding.params)
            raise ValueError(
                "This Tesseract's runtime is older than 1.13 and cannot be asked "
                f"for {names} per request. Rebuild it with a newer runtime, or "
                f"leave {names} unset."
            )
        return encoding

    for name, accepted in (
        ("output_format", capabilities.output_formats),
        ("gpu_transport", capabilities.gpu_transports),
        ("compression", capabilities.compressions),
    ):
        value = getattr(encoding, name)
        if value is not None and accepted is not None and value not in accepted:
            raise ValueError(
                f"This Tesseract does not accept {name}={value!r} "
                f"(accepted: {accepted})"
            )
    return encoding


# GPU transports a client picks by itself when a server offers them, in order of
# preference.
_AUTO_GPU_TRANSPORTS = ("cuda_ipc",)


@dataclass(frozen=True)
class _TransportCheck:
    """Whether a GPU transport works between a client and a server."""

    usable: bool | None
    """None if the server cannot be asked, because its runtime predates the check."""

    reason: str | None = None
    """Why the transport cannot be used, if it cannot."""


class Tesseract:
    """A Tesseract.

    This class represents a single Tesseract instance, either remote or local,
    and provides methods to run commands on it and retrieve results.

    Communication between a Tesseract and this class is done either via
    HTTP requests or directly via Python calls to the Tesseract API.
    """

    _spawn_config: dict | None = None
    # Which engine `serve()` should hand `_spawn_config` to.
    _spawn_backend: Literal["docker", "subprocess"] | None = None
    _serve_context: ServedTesseract | None = None
    _lastlog: str | None = None
    _client: HTTPClient | LocalClient | None = None
    _stream_logs: BoolOrCallable = False
    _timeout: float | tuple[float, float] | None = None
    _binref_pool_enabled: bool = False
    # Set on views created by with_encoding, which share their parent's client
    # without owning it.
    _encoding: RequestedEncoding | None = None
    _owns_client: bool = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "Tesseract cannot be instantiated directly. "
            "Use Tesseract.from_url(), Tesseract.from_image(), or "
            "Tesseract.from_tesseract_api() instead."
        )

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
        gpu_transport: str | None = None,
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
            gpu_transport: GPU transport to enable on the served Tesseract and use for
                this client's calls (see :meth:`with_encoding` to change it per call).
                ``none`` copies GPU arrays to the host, and ``cuda_ipc`` passes them by
                reference, which requires the container to have GPU access. Independent
                of ``output_format``, which governs CPU arrays. Resolved against
                ``runtime_config`` with the precedence described in ``engine.serve``.
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
        input_path, output_path, auto_dirs = _scratch_dirs(
            input_path, output_path, output_format
        )

        obj._stream_logs = stream_logs
        obj._timeout = timeout
        if experimental_binref_pool and not CONTAINERS_SUPPORT_BINREF_POOL:
            raise RuntimeError(
                "experimental_binref_pool=True is only supported for containerized "
                "Tesseracts on Linux, since it relies on the client and the "
                "container sharing a page cache. Elsewhere the container runs "
                "inside a VM, so bind mounts cross the VM boundary."
            )
        obj._binref_pool_enabled = experimental_binref_pool
        # Purge auto-created tempdirs when the object is garbage collected.
        # User-supplied paths are left untouched.
        for scratch in auto_dirs:
            weakref.finalize(obj, _purge_tempdir, str(scratch))
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
            gpu_transport=gpu_transport,
            runtime_config=runtime_config,
            port=port,
            host_ip=host_ip,
            debug=True,
            docker_args=docker_args,
            skip_health_check=skip_health_check,
            startup_timeout=startup_timeout,
        )
        obj._spawn_backend = "docker"
        return obj

    @classmethod
    def from_tesseract_api(
        cls,
        tesseract_api: str | Path | ModuleType,
        input_path: Path | None = None,
        output_path: Path | None = None,
        output_format: OutputFormat = "json+base64",
        gpu_transport: str | None = None,
        runtime_config: dict[str, Any] | None = None,
        stream_logs: BoolOrCallable = False,
    ) -> Tesseract:
        """Create a Tesseract instance from a Tesseract API module.

        Warning: This does not use a containerized Tesseract, but rather
        imports the Tesseract API directly. This is useful for debugging,
        but requires a matching runtime environment + all dependencies to be
        installed locally.

        Note: Uses a thread lock internally, so concurrent calls to Tesseracts
        created via `from_tesseract_api` will always run sequentially.

        Args:
            tesseract_api: Path to the `tesseract_api.py` file, or an
                already imported Tesseract API module.
            input_path: Path of input directory. All paths in the tesseract
                payload have to be relative to this path.
            output_path: Path of output directory. All paths in the tesseract
                result with be given relative to this path. Required when using json+binref.
            output_format: Format to use for the output data. json+binref requires output_path.
                This has no impact on what is returned to Python and only affects the format that is used internally.
            gpu_transport: Whether the endpoints accept GPU arrays, which only
                integrations that pass GPU arrays act on (see
                :meth:`resolve_gpu_transport`). ``none`` means they get GPU inputs
                copied to the host, and ``cuda_ipc`` means they get them as they
                are. Resolved against ``runtime_config`` with the precedence
                described in ``engine.serve``.
            runtime_config: Dictionary of runtime configuration options to pass to the Tesseract.
                For example, `{"profiling": True}` enables profiling.
            stream_logs: If True, stream logs to stdout while endpoints run.
                If a callable, stream logs to that callable instead.

        Returns:
            A Tesseract instance.
        """
        from tesseract_core.runtime.config import (
            override_config,
            snapshot_config,
            update_config,
        )

        # Runtime config is process-global. The update_config() calls below
        # need to rebuild it from scratch for this instance alone, without a
        # prior in-process Tesseract's explicit overrides leaking in.
        with override_config():
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
                resolved_output_path = engine._resolve_file_path(
                    output_path, make_dir=True
                )
                update_config(output_path=str(resolved_output_path))

            # Apply runtime_config options. Resolve the GPU transport with the same
            # precedence as serve() (explicit kwarg > runtime_config > "none"),
            # resolving it here so the config never receives None -- its field is a
            # plain str literal.
            config_kwargs: dict[str, Any] = {
                "output_format": output_format,
                "debug": True,
            }
            if runtime_config is not None:
                config_kwargs.update(runtime_config)
            if gpu_transport is not None:
                config_kwargs["gpu_transport"] = gpu_transport
            else:
                config_kwargs.setdefault("gpu_transport", "none")
            update_config(**config_kwargs)

            # Capture this instance's config so its endpoints run under it
            # later, regardless of what else touches the global config.
            config_snapshot = snapshot_config()

        obj = cls.__new__(cls)
        obj._stream_logs = stream_logs
        obj._client = LocalClient(
            tesseract_api,
            output_path=resolved_output_path,
            config_snapshot=config_snapshot,
        )
        return obj

    @classmethod
    def from_source(
        cls,
        tesseract_api: str | Path,
        input_path: Path | None = None,
        output_path: Path | None = None,
        output_format: Literal["json", "json+base64", "json+binref"] = "json+base64",
        gpu_transport: str | None = None,
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
        sharing them is unsafe (e.g. nesting JAX inside JAX can deadlock) and it
        lets the Tesseract run in a different environment than the caller (e.g.
        with conflicting dependencies).

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
            gpu_transport: GPU transport to enable on the served Tesseract and use for
                this client's calls (see :meth:`with_encoding` to change it per call).
                ``none`` copies GPU arrays to the host, and ``cuda_ipc`` passes them by
                reference, which needs no setup since processes on one host share an
                IPC namespace. Independent of ``output_format``, which governs CPU
                arrays. Resolved against ``runtime_config``, an explicit value winning.
            runtime_config: Dictionary of runtime configuration options to pass to the Tesseract.
                For example, `{"profiling": True}` enables profiling.
            stream_logs: If True, stream logs to stdout while endpoints run.
                If a callable, stream logs to that callable instead.
            python_executable: Interpreter to run the Tesseract on. It must
                have ``tesseract-core[runtime]`` and the Tesseract's
                requirements installed; pass ``sys.executable`` to use the
                SDK's own environment. If None, an environment is built from
                ``tesseract_config.yaml`` into ``.tesseract-venv`` next to the
                ``tesseract_api.py`` and reused while its requirements are
                unchanged. This needs ``uv``, or ``conda`` for
                ``requirements.provider: conda``.
            startup_timeout: How long to wait, in seconds, for the Tesseract to
                become healthy before giving up.
            experimental_binref_pool: Opt-in fast path for ``json+binref`` that
                reuses warm memory-mapped buffers instead of allocating a file
                per call. Only pays off when the binref directory is
                memory-backed (a ``tmpfs``) and has been observed to negatively affect
                ordinary disk-backed ``json+binref`` on occasion.
                See :doc:`/content/how-to/fast-local-runs`.

        Returns:
            A Tesseract instance.
        """
        obj = cls.__new__(cls)
        obj._stream_logs = stream_logs
        obj._spawn_backend = "subprocess"
        if experimental_binref_pool and not SUPPORTS_BINREF_POOL:
            raise RuntimeError(
                "experimental_binref_pool=True is not supported on this platform: "
                "it decodes outputs as read-only memory maps, which needs POSIX."
            )
        obj._binref_pool_enabled = experimental_binref_pool
        auto_dirs, obj._spawn_config = _subprocess_spawn_config(
            tesseract_api,
            input_path=input_path,
            output_path=output_path,
            output_format=output_format,
            gpu_transport=gpu_transport,
            runtime_config=runtime_config,
            python_executable=python_executable,
            startup_timeout=startup_timeout,
        )
        # Purge auto-created scratch dirs when the object is garbage collected.
        for scratch in auto_dirs:
            weakref.finalize(obj, _purge_tempdir, str(scratch))
        return obj

    def __enter__(self) -> Tesseract:  # noqa: PYI034 - typing.Self needs py3.11
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

        This will stop the Tesseract server if it is running, and release the
        resources held by the client.
        """
        if self._serve_context is None:
            # Nothing was served by us (e.g., from_url or from_tesseract_api), so
            # there is no container to stop, but an HTTP session is still ours to close
            if isinstance(self._client, HTTPClient) and self._owns_client:
                self._client.close()
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
            self._serve_context = local_client.serve(**self._spawn_config)
        else:
            _, self._serve_context = engine.serve(**self._spawn_config)

        self._lastlog = None
        output_path = self._spawn_config.get("output_path")
        input_path = self._spawn_config.get("input_path")
        output_format = self._spawn_config.get("output_format", "json+base64")
        # The output format and GPU transport configure the server, and are also
        # what this client requests by default (with_encoding overrides them per
        # call). Resolve the transport with the same precedence serve() applies
        # to the container: the explicit kwarg wins, else a value from
        # runtime_config. If neither sets it, the client does not ask, and gets
        # the server's default of "none".
        runtime_config = self._spawn_config.get("runtime_config") or {}
        gpu_transport = self._spawn_config.get("gpu_transport") or runtime_config.get(
            "gpu_transport"
        )
        self._client = HTTPClient(
            self._serve_context.url,
            output_path=Path(output_path) if output_path else None,
            output_format=output_format,
            timeout=self._timeout,
            input_path=Path(input_path) if input_path else None,
            experimental_binref_pool=self._binref_pool_enabled,
            gpu_transport=gpu_transport,
        )

        # Ensure that the Tesseract is torn down once the object is garbage collected,
        # to avoid orphaned containers or processes if the user forgets to call
        # .teardown()
        def _silent_teardown(handle: ServedTesseract) -> None:
            from tesseract_core.sdk.docker_client import NotFound

            try:
                handle.remove(force=True)
            except NotFound:
                pass

        self._atexit_finalizer = weakref.finalize(
            self, _silent_teardown, self._serve_context
        )

    def teardown(self) -> None:
        """Teardown the Tesseract.

        This will stop and remove the Tesseract container, or stop the dedicated
        process serving it.
        """
        if self._serve_context is None:
            raise RuntimeError("Tesseract is not being served.")
        self._lastlog = self.server_logs()
        self._serve_context.remove(force=True)
        if self._client is not None:
            self._client.close()
        self._client = None
        self._serve_context = None
        self._atexit_finalizer.detach()

    @property
    @requires_client
    def openapi_schema(self) -> dict:
        """Get the OpenAPI schema of this Tesseract.

        Returns:
            dictionary with the OpenAPI Schema.
        """
        return self._client.openapi_schema

    @property
    @requires_client
    def available_endpoints(self) -> list[str]:
        """Get the list of available endpoints.

        Returns:
            a list with all available endpoints for this Tesseract.
        """
        return [endpoint.lstrip("/") for endpoint in self.openapi_schema["paths"]]

    @property
    @requires_client
    def server_capabilities(self) -> ServerCapabilities | None:
        """The encodings this Tesseract's server accepts, read from its OpenAPI schema.

        A listed value can still fail for reasons the server cannot know about.
        For example, ``cuda_ipc`` requires client and server to share a host and a GPU.

        Returns:
            the advertised capabilities, or None for in-process Tesseracts created
            via :meth:`from_tesseract_api`, which do not serialize arrays.
        """
        return self._client.server_capabilities

    @requires_client
    def with_encoding(
        self,
        *,
        output_format: OutputFormat | None = None,
        gpu_transport: str | None = None,
        compression: Literal["none", "lz4"] | None = None,
    ) -> Tesseract:
        """Get a view of this Tesseract that requests a different encoding.

        The view shares this Tesseract's connection and stays usable while this
        Tesseract is served. It only calls endpoints, so serving, logs and
        container info stay with this Tesseract. Arguments left as None keep
        this Tesseract's setting.

            >>> with Tesseract.from_image(
            ...     "my_tesseract", gpu_transport="cuda_ipc"
            ... ) as t:
            ...     t.with_encoding(gpu_transport="none").apply(inputs)

        In-process Tesseracts created via :meth:`from_tesseract_api` pass arrays in
        memory, so only ``gpu_transport`` means anything for them: it is what
        :meth:`resolve_gpu_transport` returns, which tells integrations such as
        Tesseract-JAX whether to hand the endpoints GPU arrays as they are.

        Args:
            output_format: Format for CPU arrays in responses.
            gpu_transport: How GPU arrays cross the boundary in either direction.
                ``none`` copies them to the host, and ``cuda_ipc`` passes them by
                reference.
            compression: Compression for array buffers in responses (``none``
                disables it).

        Returns:
            A Tesseract that uses the requested encoding for every call.

        Raises:
            ValueError: if the server does not accept a requested value, or runs
                a runtime too old to be asked for it.
        """
        requested = RequestedEncoding(output_format, gpu_transport, compression)
        if isinstance(self._client, LocalClient):
            self._client.check_requested_gpu_transport(gpu_transport)
        else:
            # Calls check again with the full encoding, but failing here points
            # at the argument that caused it
            capabilities = self._client.server_capabilities
            _fit_encoding_to_server(requested, capabilities)
            if compression is not None and capabilities.compressions is None:
                warnings.warn(
                    "This Tesseract does not advertise which compressions it "
                    "accepts. Runtimes older than 1.14 ignore the requested "
                    "compression and use the one they were configured with.",
                    stacklevel=3,
                )

        view = Tesseract.__new__(Tesseract)
        view._client = self._client
        view._owns_client = False
        view._stream_logs = self._stream_logs
        view._encoding = (self._encoding or RequestedEncoding()).merge(requested)
        return view

    @property
    @requires_client
    def current_encoding(self) -> RequestedEncoding:
        """The encoding calls through this Tesseract request.

        The options this Tesseract was created with, overridden by those of
        :meth:`with_encoding` for a view. A field is None if calls leave it to the
        server's default, which for ``gpu_transport`` means copying GPU arrays to
        the host. To find out which GPU transport to use for GPU arrays, see
        :meth:`resolve_gpu_transport`.
        """
        return self._client.current_encoding(self._encoding)

    @requires_client
    def resolve_gpu_transport(self) -> str:
        """Get the GPU transport that calls through this Tesseract should use for GPU arrays.

        For integrations that pass GPU arrays, such as Tesseract-JAX and
        Tesseract-Torch, so that GPU arrays stay on the device whenever that
        works and are copied to the host otherwise:

            >>> transport = tess.resolve_gpu_transport()
            >>> tess.with_encoding(gpu_transport=transport).apply(gpu_inputs)

        If this Tesseract requests a GPU transport, either through
        :meth:`with_encoding` or the ``gpu_transport`` it was created with, that
        transport is returned once it is known to work. Otherwise this returns
        ``cuda_ipc`` if the server offers it and it works from this process, and
        ``none`` if not, with a warning if the server offers it but it does not
        work.

        Whether a transport works depends on how both processes are set up
        (``cuda_ipc`` needs them on one host with the same GPU visible to both,
        and container isolation can get in the way), so it is found out by
        exchanging a small GPU array with the server, once per connection (and
        again on the next call if the exchange reached no conclusion, such as
        when the server answered with an error). Runtimes that predate this
        exchange cannot take part in it, so a transport they offer, or that is
        requested from them, is used unchecked.

        In-process Tesseracts created via :meth:`from_tesseract_api` return the
        ``gpu_transport`` they were created with unless a view requests ``none``.

        Returns:
            ``none``, or the name of the GPU transport to use.

        Raises:
            ValueError: if the requested transport is not enabled on this Tesseract.
            RuntimeError: if the requested transport does not work from this process.
        """
        return self._client.resolve_gpu_transport(self._encoding)

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
        if self._spawn_backend != "docker":
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
        return self._client.run_tesseract(
            "apply", payload, run_id, self._stream_logs, encoding=self._encoding
        )

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
            "jacobian", payload, run_id, self._stream_logs, encoding=self._encoding
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
            "jacobian_vector_product",
            payload,
            run_id,
            self._stream_logs,
            encoding=self._encoding,
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
            "vector_jacobian_product",
            payload,
            run_id,
            self._stream_logs,
            encoding=self._encoding,
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
    gpu_transport: str | None,
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
        # Public error type; switching to TypeError would break callers catching ValueError.
        raise ValueError(  # noqa: TRY004
            "`from_source` requires a path to a `tesseract_api.py` file, but an "
            f"already imported module was given "
            f"({getattr(tesseract_api, '__name__', tesseract_api)!r}). A module "
            "cannot be shared with another process; pass `module.__file__`, or "
            "use `from_tesseract_api` to run it in this one."
        )

    tesseract_api_path = Path(tesseract_api).resolve(strict=True)
    if not tesseract_api_path.is_file():
        raise RuntimeError(f"Tesseract API path {tesseract_api_path} is not a file.")

    resolved_input_path, resolved_output_path, auto_dirs = _scratch_dirs(
        input_path, output_path, output_format
    )

    # Debug mode gives full tracebacks from the child and enables the `test`
    # endpoint, matching what the in-process path configures. The debugpy
    # listener it would normally imply is disabled separately, in
    # `local_client.serve`.
    config_kwargs: dict[str, Any] = {"debug": True}
    if runtime_config is not None:
        config_kwargs.update(runtime_config)

    # Same precedence as the other constructors: an explicit value (including
    # "none") wins over one in runtime_config, which wins over the default.
    # Unlike a container there is nothing to wire up for it -- two processes on
    # one host already share an IPC namespace, so cuda_ipc needs no equivalent
    # of the container's `--ipc=host`, and the child sees the host's GPUs.
    if gpu_transport is not None:
        config_kwargs["gpu_transport"] = gpu_transport
    else:
        config_kwargs.setdefault("gpu_transport", "none")

    if config_kwargs["gpu_transport"] not in ("none", "cuda_ipc"):
        raise ValueError(
            f"Unknown gpu_transport {config_kwargs['gpu_transport']!r}. "
            "Supported values: 'none', 'cuda_ipc'."
        )

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

    The ``cuda_ipc`` GPU transport lives in ``tesseract_core.runtime``, which is
    an optional install (``tesseract-core[runtime]``). A base SDK install lacks
    its dependencies, so surface a clear message pointing at the extra instead of
    a bare ``ModuleNotFoundError`` from deep in the import chain.
    """
    try:
        from tesseract_core.runtime.cuda import ipc as cuda_ipc
    except ImportError as exc:
        raise ImportError(
            "The 'cuda_ipc' GPU transport requires the Tesseract runtime, "
            "which is an optional dependency. Install it with "
            "'pip install tesseract-core[runtime]'."
        ) from exc
    return cuda_ipc


@dataclass
class EncodingContext:
    """Request-scoped context tracking state and resources during input array encoding."""

    input_dir: Path | None = None
    binref_pool: BinrefWritePool | None = None
    written_files: list[Path] = field(default_factory=list)
    checked_out_slots: list[BinrefSlot] = field(default_factory=list)
    # The cuda_ipc ExportGroup holding this request's exported arrays, if any.
    device_exports: Any = None


def _encode_binref(arr: Any, ctx: EncodingContext) -> dict:
    """Encode an array as binref, tracking written files or pool slots in ``ctx``."""
    if ctx.binref_pool is not None:
        return encode_array_binref_pooled(
            arr, ctx.binref_pool, ctx.checked_out_slots, ctx.written_files
        )
    if ctx.input_dir is not None:
        return encode_array_binref(arr, ctx.input_dir, ctx.written_files)
    raise ValueError(
        "EncodingContext.input_dir or binref_pool is required when encoding is 'binref'"
    )


def _close_encoding_context(ctx: EncodingContext) -> None:
    """Release resources held in ``ctx`` during encoding with resilient partial failure handling."""
    errors: list[BaseException] = []

    try:
        for f in ctx.written_files:
            try:
                f.unlink(missing_ok=True)
            except Exception as ex:  # noqa: BLE001 - collected and re-raised below
                errors.append(ex)
        ctx.written_files.clear()
    finally:
        try:
            if ctx.binref_pool is not None:
                for slot in ctx.checked_out_slots:
                    try:
                        ctx.binref_pool.checkin(slot)
                    except Exception as ex:  # noqa: BLE001 - collected and re-raised below
                        errors.append(ex)
                ctx.checked_out_slots.clear()
        finally:
            if ctx.device_exports is not None:
                try:
                    ctx.device_exports.release()
                except Exception as ex:  # noqa: BLE001 - collected and re-raised below
                    errors.append(ex)
                ctx.device_exports = None

    if errors:
        if len(errors) == 1:
            raise errors[0]
        exception_group_cls = getattr(builtins, "ExceptionGroup", None)
        if exception_group_cls is not None:
            raise exception_group_cls(
                "Errors occurred during EncodingContext cleanup", errors
            )
        raise RuntimeError(
            f"Multiple errors occurred during EncodingContext cleanup: {errors}"
        )


def _is_gpu_array(x: Any) -> bool:
    """Whether ``x`` exposes ``__cuda_array_interface__``.

    PyTorch raises RuntimeError rather than AttributeError for a CUDA tensor
    that requires grad, which is still a GPU array.
    """
    try:
        return hasattr(x, "__cuda_array_interface__")
    except RuntimeError:
        return True


def _without_autograd(arr: Any) -> Any:
    """``arr``, detached if it is a PyTorch tensor that requires grad.

    Such a tensor refuses ``__cuda_array_interface__`` and ``.numpy()``.
    Encoding only reads its values, so a detached view of the same memory
    serves instead.
    """
    if getattr(arr, "requires_grad", False) and callable(getattr(arr, "detach", None)):
        return arr.detach()
    return arr


def _gpu_array_to_host(arr: Any) -> np.ndarray:
    """Copy a GPU array to the host, for encodings that serialize its bytes."""
    # Import the runtime only when the flag is set or the array needs it, so a
    # base SDK install without it can still host-copy arrays that convert
    # themselves (e.g. JAX's). Keep the truthy values in sync with
    # check_device_host_copy.
    if os.environ.get("TESSERACT_FORBID_DEVICE_HOST_COPY", "").lower() in {
        "1",
        "true",
    }:
        _import_cuda_ipc().check_device_host_copy(f"a {type(arr).__name__} GPU array")
    try:
        host = np.asanyarray(arr)
    except (TypeError, ValueError):
        # PyTorch and CuPy refuse to convert device memory implicitly
        host = None
    if host is None or host.dtype == object:
        host = _import_cuda_ipc().cuda_array_to_host(arr)
    return host


def _encode_array(
    arr: Any,
    encoding: Literal["base64", "raw", "cuda_ipc", "cuda_vmm", "binref"] = "base64",
    ctx: EncodingContext | None = None,
) -> dict:
    """Encode an array into an arraydict representation.

    When ``encoding='cuda_ipc'``, GPU arrays are exported by handle via CUDA IPC,
    keeping the data on-device. An :class:`EncodingContext` is required to track
    the exported allocation so its pinned memory is released on context exit.
    Any other encoding of a GPU array is a host copy, subject to the runtime's
    ``check_device_host_copy`` guard.

    When ``encoding='binref'``, an :class:`EncodingContext` is required to write
    the buffer to the input directory or write pool and track the file/slot lifetime.
    """
    if _is_gpu_array(arr):
        arr = _without_autograd(arr)
        if encoding == "cuda_ipc":
            if ctx is None:
                raise ValueError(
                    "EncodingContext is required when encoding is 'cuda_ipc'"
                )
            cuda_ipc = _import_cuda_ipc()
            if ctx.device_exports is None:
                ctx.device_exports = cuda_ipc.ExportGroup()
            return cuda_ipc.dump_cuda_ipc_arraydict(arr, group=ctx.device_exports)
        arr = _gpu_array_to_host(arr)

    if encoding == "binref":
        if ctx is None:
            raise ValueError("EncodingContext is required when encoding is 'binref'")
        return _encode_binref(arr, ctx)

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
def _encode_payload(
    payload: dict | None,
    gpu_transport: str = "none",
    output_format: OutputFormat | None = None,
    input_path: PathLike | None = None,
    binref_pool: BinrefWritePool | None = None,
) -> Iterator[dict | None]:
    """Encode a request payload's arrays, managing device-export and binref-file lifetimes.

    Yields the encoded payload (or None for an empty payload). When a
    ``gpu_transport`` other than ``none`` is set, GPU arrays are exported by
    reference (keeping the data on-device), which pins each exported allocation
    in a process-global registry on the runtime side. Host arrays (and GPU
    arrays when ``gpu_transport`` is ``none``) are encoded according to
    ``output_format`` (``binref`` files when ``output_format == "json+binref"``
    and ``input_path`` is set, else ``base64``).

    Resources (pinned GPU memory allocations and temporary binref files/slots) are
    released on context exit, by which point the caller has read the full response.
    """
    if not payload:
        yield None
        return

    resolved_input_path = Path(input_path) if input_path is not None else None
    ctx = EncodingContext(
        input_dir=resolved_input_path,
        binref_pool=binref_pool,
    )
    use_binref = output_format == "json+binref" and (
        resolved_input_path is not None or binref_pool is not None
    )

    def _encode_leaf(x: Any) -> dict:
        if _is_gpu_array(x) and gpu_transport != "none":
            return _encode_array(x, encoding=gpu_transport, ctx=ctx)

        # Host array (or GPU array when gpu_transport is "none")
        if use_binref:
            return _encode_array(x, encoding="binref", ctx=ctx)

        return _encode_array(x, encoding="base64", ctx=ctx)

    def _is_leaf(x: Any) -> bool:
        return hasattr(x, "__array__") or _is_gpu_array(x)

    try:
        encoded_payload = _tree_map(_encode_leaf, payload, is_leaf=_is_leaf)
        yield encoded_payload
    finally:
        _close_encoding_context(ctx)


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

        # The buffer reference comes from the (untrusted) server response, so it
        # must stay within output_path. Otherwise an absolute path or `..`
        # traversal could read, or on the lazy path unlink, arbitrary client
        # files.
        if output_path is None:
            raise ValueError(
                "output_path must be set to decode a json+binref response."
            )
        base = Path(output_path).resolve()
        full_path = (base / bufferpath).resolve()
        if not full_path.is_relative_to(base):
            raise ValueError(
                f"Binref buffer reference {bufferpath!r} escapes output_path. "
                "Refusing to read a file outside the output directory."
            )

        if not full_path.exists():
            raise ValueError(
                f"Binary file not found: {full_path}. "
                "The server referenced a binref buffer that is not present in "
                "output_path."
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
        # Returns a client-owned device-array wrapper. The decode copies
        # device-to-device into our own memory, and the result exposes
        # __cuda_array_interface__ and __dlpack__ so Torch/JAX/CuPy can adopt it
        # zero-copy. The server may reuse/free the exported buffer as soon as
        # this returns (it holds it until the next request).
        #
        # cuda_ipc is strictly opt-in, so reaching here means the caller asked
        # for it. If this client has no usable CUDA context (no driver, no
        # visible/matching device, or the runtime extra not installed) it cannot
        # open the handle, so translate the low-level failure into an actionable
        # message rather than a bare CUDA/import error deep in the decode.
        try:
            return _import_cuda_ipc().load_cuda_ipc_arraydict(encoded_arr)
        except Exception as exc:
            raise RuntimeError(
                "Received a GPU array via the 'cuda_ipc' transport, but this "
                "client could not open it on the local GPU (no CUDA driver, no "
                "matching device, or the runtime extra is missing). Drop "
                "gpu_transport='cuda_ipc' to have arrays copied to the host "
                "instead, or ensure this process shares a GPU and IPC namespace "
                "with the Tesseract."
            ) from exc
    else:
        raise ValueError(f"Unexpected array encoding {encoding}. Cannot decode.")

    arr = arr.reshape(shape)
    return arr


# How a client and server agree when the server may release the device memory a
# response exported (see tesseract_core.runtime.serve, whose names these mirror).
_EXPORTS_HEADER = "Tesseract-Exports"
_EXPORTS_DONE_HEADER = "Tesseract-Exports-Done"


class HTTPClient:
    """HTTP Client for Tesseracts."""

    # Class-level defaults so instances built via ``__new__`` (e.g. in tests)
    # still expose the binref attributes the request/decode paths read.
    _input_path: Path | None = None
    _binref_pool: BinrefWritePool | None = None
    _output_format: str | None = None
    _gpu_transport: str | None = None
    # Guards the transport checks. Each instance gets its own in __init__, so a
    # check waiting on a busy server never holds up checks of other servers.
    _transport_lock: threading.Lock = threading.Lock()
    # Ids of responses whose device exports this client has finished reading,
    # not yet named to the server (see _EXPORTS_DONE_HEADER). None for
    # instances built without __init__, which then do not acknowledge.
    _done_exports: collections.deque[str] | None = None

    def __init__(
        self,
        url: str,
        output_path: str | Path | None = None,
        output_format: OutputFormat | None = None,
        timeout: float | tuple[float, float] | None = None,
        input_path: str | Path | None = None,
        experimental_binref_pool: bool = False,
        gpu_transport: str | None = None,
    ) -> None:
        self._url = self._sanitize_url(url)
        self._output_path = output_path
        # Requested by default, with None deferring to the server.
        self._output_format = output_format
        self._gpu_transport = gpu_transport
        self._input_path = Path(input_path) if input_path is not None else None
        self._timeout = timeout
        self._transport_lock = threading.Lock()
        self._done_exports = collections.deque()
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
        """Release resources held by the client (HTTP session, binref write pool)."""
        if self._done_exports:
            # Let the server release the exports of the last responses now
            # rather than when it gives up waiting for this client.
            try:
                self._send(
                    f"{self.url}/health", "GET", b"", {}, self._exports_done_headers()
                )
            except Exception:  # noqa: BLE001, S110 - best effort while closing
                pass
        if self._binref_pool is not None:
            self._binref_pool.close()
            self._binref_pool = None
        self._session.close()

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

    @property
    def default_encoding(self) -> RequestedEncoding:
        """What this client requests for calls that do not override it."""
        return RequestedEncoding(self._output_format, self._gpu_transport)

    @cached_property
    def openapi_schema(self) -> dict:
        """The server's OpenAPI schema, fetched once per client."""
        response = self._send(f"{self.url}/openapi.json", "GET", b"", {})
        return self._decode_response(response, "openapi.json")

    @cached_property
    def server_capabilities(self) -> ServerCapabilities:
        """The encodings the server advertises in its OpenAPI schema."""
        return ServerCapabilities.from_openapi_schema(self.openapi_schema)

    @cached_property
    def _transport_checks(self) -> dict[str, _TransportCheck]:
        """Results of :meth:`check_gpu_transport`, by transport."""
        return {}

    def check_gpu_transport(self, gpu_transport: str) -> _TransportCheck:
        """Find out whether ``gpu_transport`` works between this process and the server.

        Runs the check the first time it is asked for each transport and returns
        the same answer after that, unless the check reached no conclusion (the
        server answered with an error, or this process could not export GPU
        memory, as when it runs out of it), in which case the next call checks
        again.
        """
        with self._transport_lock:
            check = self._transport_checks.get(gpu_transport)
            if check is None:
                check, conclusive = self._run_transport_check(gpu_transport)
                if conclusive:
                    self._transport_checks[gpu_transport] = check
            return check

    def _run_transport_check(self, gpu_transport: str) -> tuple[_TransportCheck, bool]:
        """Run the check once, returning its result and whether it is conclusive."""
        if gpu_transport != "cuda_ipc":
            return _TransportCheck(
                False, f"unknown gpu_transport {gpu_transport!r}"
            ), True
        try:
            cuda_ipc = _import_cuda_ipc()
            request, expected, exported = cuda_ipc.start_transport_check()
        except Exception as exc:  # noqa: BLE001 - becomes the reason it cannot be used
            return _TransportCheck(
                False,
                "this process could not export GPU memory "
                f"({type(exc).__name__}: {exc})",
            ), False
        # The server reads the exported array while answering, so it must stay
        # alive until the response is in. It is not in the per-request export
        # registry, so holding it here is what keeps it alive.
        response = self._send(
            f"{self.url}/check_gpu_transport",
            "POST",
            orjson.dumps(request),
            {},
        )
        del exported
        if response.status_code == requests.codes.not_found:
            return _TransportCheck(None), True
        if not response.ok:
            return _TransportCheck(
                False,
                f"the Tesseract answered the check with error "
                f"{response.status_code}: {response.text}",
            ), False
        reply = from_json(response.content)
        if not reply.get("ok"):
            return _TransportCheck(False, reply.get("reason", "no reason given")), True
        try:
            cuda_ipc.finish_transport_check(reply.get("array"), expected)
        except Exception as exc:  # noqa: BLE001 - becomes the reason it cannot be used
            return _TransportCheck(
                False,
                "this process could not open GPU memory exported by the Tesseract "
                f"({type(exc).__name__}: {exc})",
            ), True
        return _TransportCheck(True), True

    def current_encoding(
        self, overrides: RequestedEncoding | None
    ) -> RequestedEncoding:
        """See :attr:`Tesseract.current_encoding`."""
        return self.default_encoding.merge(overrides)

    def resolve_gpu_transport(self, overrides: RequestedEncoding | None) -> str:
        """See :meth:`Tesseract.resolve_gpu_transport`."""
        requested = self.current_encoding(overrides).gpu_transport
        if requested == "none":
            return "none"

        if requested is not None:
            _fit_encoding_to_server(
                RequestedEncoding(gpu_transport=requested), self.server_capabilities
            )
            check = self.check_gpu_transport(requested)
            if check.usable is False:
                raise RuntimeError(
                    f"gpu_transport={requested!r} was requested, but it does not "
                    f"work between this process and the Tesseract: {check.reason}. "
                    "Request gpu_transport='none' to copy GPU arrays to the host."
                )
            return requested

        offered = self.server_capabilities.gpu_transports or ()
        for candidate in _AUTO_GPU_TRANSPORTS:
            if candidate not in offered:
                continue
            check = self.check_gpu_transport(candidate)
            # None: the server's runtime cannot be asked, so use the transport
            # unchecked, as for a requested one.
            if check.usable is not False:
                return candidate
            with self._transport_lock:
                warn = candidate not in self._transport_fallback_warned
                self._transport_fallback_warned.add(candidate)
            if warn:
                warnings.warn(
                    f"The Tesseract at {self.url} offers gpu_transport="
                    f"{candidate!r}, but it does not work from this process, so GPU "
                    f"arrays are copied to the host instead: {check.reason}",
                    stacklevel=4,
                )
        return "none"

    @cached_property
    def _transport_fallback_warned(self) -> set[str]:
        """Transports this client has warned about not being able to use."""
        return set()

    def _send(
        self,
        url: str,
        method: str,
        data: bytes,
        params: dict,
        headers: dict[str, str] | None = None,
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
        # Per-request headers, rather than session state, so concurrent calls
        # with different encodings cannot interfere.
        if headers:
            request_kwargs["headers"] = headers
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
        encoding: RequestedEncoding | None = None,
    ) -> dict:
        url = f"{self.url}/{endpoint.lstrip('/')}"
        params = {"run_id": run_id} if run_id is not None else {}
        encoding = self.default_encoding.merge(encoding)
        # Only parameters can trip up a server, and checking them costs a fetch
        # of the OpenAPI schema on first use
        if encoding.params:
            encoding = _fit_encoding_to_server(encoding, self.server_capabilities)
        accept = encoding.accept_header()
        # Only a request for a GPU transport can get exports back, so only
        # such requests need to say that this client names the ones it is done
        # with (and any request can carry names still pending).
        headers = {}
        if encoding.gpu_transport not in (None, "none") or self._done_exports:
            headers = self._exports_done_headers()
        if accept is not None:
            headers["Accept"] = accept

        with _encode_payload(
            payload,
            gpu_transport=encoding.gpu_transport or "none",
            output_format=encoding.output_format,
            input_path=self._input_path,
            binref_pool=self._binref_pool,
        ) as encoded_payload:
            try:
                response = self._send(
                    url, method, orjson.dumps(encoded_payload), params, headers or None
                )
            except Exception:
                # The server never saw them, so name them again next time.
                if headers.get(_EXPORTS_DONE_HEADER):
                    self._done_exports.extend(headers[_EXPORTS_DONE_HEADER].split(","))
                raise
        if _EXPORTS_DONE_HEADER not in headers:
            return self._decode_response(response, endpoint)
        try:
            return self._decode_response(response, endpoint)
        finally:
            # Decoding copied any device arrays out of the server's memory.
            export_id = response.headers.get(_EXPORTS_HEADER)
            if export_id:
                self._done_exports.append(export_id)

    def _exports_done_headers(self) -> dict[str, str]:
        """Headers naming the responses this client is done with, as a dict to extend.

        Sent even when there are none, so the server knows this client names
        them and keeps its responses' exports until it does.
        """
        if self._done_exports is None:
            return {}
        done = []
        while True:
            try:
                done.append(self._done_exports.popleft())
            except IndexError:
                break
        return {_EXPORTS_DONE_HEADER: ",".join(done)}

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
        encoding: RequestedEncoding | None = None,
    ) -> dict:
        """Run a Tesseract endpoint.

        Args:
            endpoint: The endpoint to run.
            payload: The payload to send to the endpoint.
            run_id: a string to identify the run. Run outputs will be located
                    in a directory suffixed with this id.
            stream_logs: If True, stream logs to stdout. If a callable, stream
                    logs to that callable.
            encoding: Overrides this client's default encoding for this call.

        Returns:
            The loaded JSON response from the endpoint, with decoded arrays.
        """
        method = "GET" if endpoint == "health" else "POST"

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
            return self._request(endpoint, method, payload, run_id, encoding)
        finally:
            if log_streamer is not None:
                log_streamer.stop()


class LocalClient:
    """Local Client for Tesseracts."""

    # Arrays are passed in memory, so there is no encoding to negotiate
    server_capabilities: ServerCapabilities | None = None

    def __init__(
        self,
        tesseract_api: ModuleType,
        output_path: Path | None = None,
        config_snapshot: ConfigSnapshot | None = None,
    ) -> None:
        # Import here to not depend on runtime dependencies globally
        from tesseract_core.runtime.config import override_config, snapshot_config
        from tesseract_core.runtime.core import create_endpoints
        from tesseract_core.runtime.serve import create_rest_api

        # Fall back to the current global config for direct LocalClient users
        # (Tesseract.from_tesseract_api passes its own snapshot).
        self._config_snapshot = (
            config_snapshot if config_snapshot is not None else snapshot_config()
        )

        with override_config(self._config_snapshot):
            self._endpoints = {
                func.__name__: func for func in create_endpoints(tesseract_api)
            }
            self.openapi_schema = create_rest_api(tesseract_api).openapi()

        if output_path is None:
            output_path = Path(tempfile.mkdtemp(prefix="tesseract_output_"))
            # Purge the auto-created tempdir when this client is garbage collected.
            weakref.finalize(self, _purge_tempdir, str(output_path))
        self._output_path = output_path
        # Allows external clients (e.g. tesseract-jax) to access module directly
        self.api_module = tesseract_api

    @property
    def gpu_transport(self) -> str:
        """The GPU transport this Tesseract was created with (``none`` if none)."""
        config = self._config_snapshot[0]
        return "none" if config is None else config.gpu_transport

    def check_requested_gpu_transport(self, gpu_transport: str | None) -> None:
        """Raise if ``gpu_transport`` is a transport this Tesseract was not created with."""
        if gpu_transport not in (None, "none", self.gpu_transport):
            raise ValueError(
                f"This in-process Tesseract does not accept gpu_transport="
                f"{gpu_transport!r}, since it was created with gpu_transport="
                f"{self.gpu_transport!r}. Create it with Tesseract.from_tesseract_api"
                f"(..., gpu_transport={gpu_transport!r}) if its endpoints accept GPU "
                "arrays."
            )

    def current_encoding(
        self, overrides: RequestedEncoding | None
    ) -> RequestedEncoding:
        """See :attr:`Tesseract.current_encoding`."""
        config = self._config_snapshot[0]
        if config is None:
            created = RequestedEncoding(gpu_transport="none")
        else:
            created = RequestedEncoding(
                config.output_format, config.gpu_transport, config.compression
            )
        return created.merge(overrides)

    def resolve_gpu_transport(self, overrides: RequestedEncoding | None) -> str:
        """See :meth:`Tesseract.resolve_gpu_transport`."""
        return self.current_encoding(overrides).gpu_transport

    def run_tesseract(
        self,
        endpoint: str,
        payload: dict | None = None,
        run_id: str | None = None,
        stream_logs: BoolOrCallable = False,
        encoding: RequestedEncoding | None = None,
    ) -> dict:
        """Run a Tesseract endpoint.

        Args:
            endpoint: The endpoint to run.
            payload: The payload to send to the endpoint.
            run_id: a string to identify the run.
            stream_logs: If True, stream logs to stdout. If a callable, stream logs to that callable.
            encoding: Ignored, since arrays are passed in memory. Accepted so
                both clients can be called the same way.

        Returns:
            The loaded JSON response from the endpoint, with decoded arrays.
        """
        if endpoint not in self._endpoints:
            raise RuntimeError(f"Endpoint {endpoint} not found in Tesseract API.")

        # Import here to not depend on runtime dependencies globally
        from tesseract_core.runtime.config import (
            get_config,
            override_config,
            snapshot_config,
        )
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

        # Run under this instance's own config rather than whatever the global
        # happens to be. Any update_config() the endpoint makes is captured back
        # into the snapshot afterwards, so it persists to later calls (matching
        # the containerized case).
        with override_config(self._config_snapshot):
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
            except Exception as ex:  # noqa: BLE001 - user endpoint code; re-raised with traceback
                # Some clients like Tesseract-JAX swallow tracebacks from re-raised exceptions, so we explicitly
                # format the traceback here to include it in the error message.
                tb = traceback.format_exc()
                raise RuntimeError(
                    f"{tb}\nError running Tesseract API {endpoint}: {ex} (see above for full traceback)"
                ) from None
            finally:
                self._config_snapshot = snapshot_config()

        if OutputSchema is not None:
            # Validate via schema, then dump to stay consistent with other clients
            if isinstance(OutputSchema, type) and issubclass(OutputSchema, BaseModel):
                result = OutputSchema.model_validate(result).model_dump()
            else:
                result = TypeAdapter(OutputSchema).validate_python(result)

        return result
