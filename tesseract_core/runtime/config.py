# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FilePath,
    field_validator,
)

from tesseract_core.runtime.file_interactions import (
    gpu_transport_type,
    supported_format_type,
)


def _eval_str(obj: Any) -> Any:
    """Evaluate a string into the corresponding Python object."""
    if isinstance(obj, str):
        try:
            return ast.literal_eval(obj)
        except SyntaxError as exc:
            raise ValueError("Could not parse string as Python object") from exc
    return obj


class RuntimeConfig(BaseModel):
    """Available runtime configuration."""

    api_path: FilePath = Path("tesseract_api.py")
    name: str = "Tesseract"
    description: str = ""
    version: str = "unknown"
    debug: bool = False
    debugpy_host: str = "127.0.0.1"
    debugpy_port: int = 5678
    input_path: str = "."
    output_path: str = "."
    output_format: supported_format_type = "json"
    output_file: str = ""
    compression: Literal["lz4"] | None = None
    mlflow_tracking_uri: str = ""
    mlflow_run_extra_args: Annotated[dict[str, Any], BeforeValidator(_eval_str)] = (
        Field(default_factory=dict)
    )
    profiling: bool = False
    tracing: bool = False
    # How device (GPU) arrays leave the process. Any value other than ``none``
    # (e.g. ``cuda_ipc``, set via TESSERACT_GPU_TRANSPORT=cuda_ipc) is an
    # experimental, unstable capability that may change or be removed without
    # notice: it exports device memory by reference without a host round-trip.
    # ``none`` (default) instead copies GPU arrays to the host and serializes
    # them via ``output_format`` like any CPU array, so a Tesseract never emits
    # by-reference handles unless explicitly opted in. Independent of
    # ``output_format``, which only governs CPU arrays.
    gpu_transport: gpu_transport_type = "none"
    # Directory the ``cuda_vmm`` transport binds its fd-passing Unix socket
    # under. Empty (default) lets the transport pick: the ``output_path``
    # shared mount when set (so a host consumer can reach a containerized
    # server's socket), else the system temp dir. Set it (e.g.
    # TESSERACT_VMM_SOCKET_DIR=/dev/shm) only to override that choice.
    vmm_socket_dir: str = ""

    @field_validator("input_path", "output_path")
    @classmethod
    def _resolve_path(cls, v: str) -> str:
        return str(Path(v).resolve())

    model_config = ConfigDict(frozen=True, extra="forbid")


def update_config(**kwargs: Any) -> None:
    """Create a new runtime configuration from the current environment.

    Passed keyword arguments will override environment variables.
    """
    global _current_config

    conf_settings = {}
    for field in RuntimeConfig.model_fields.keys():
        env_key = f"TESSERACT_{field.upper()}"
        if env_key in os.environ:
            conf_settings[field] = os.environ[env_key]

    for field in _config_overrides:
        conf_settings[field] = getattr(_current_config, field)

    conf_settings.update(kwargs)
    config = RuntimeConfig(**conf_settings)

    _config_overrides.update(set(conf_settings.keys()))
    _current_config = config


_current_config = None
_config_overrides = set()


def get_config() -> RuntimeConfig:
    """Return the current runtime configuration."""
    if _current_config is None:
        update_config()
    assert _current_config is not None
    return _current_config


ConfigSnapshot = tuple[RuntimeConfig | None, frozenset[str]]


def snapshot_config() -> ConfigSnapshot:
    """Capture the current (config, overrides) pair."""
    return _current_config, frozenset(_config_overrides)


@contextmanager
def override_config(snapshot: ConfigSnapshot | None = None) -> Iterator[None]:
    """Install ``snapshot`` as the runtime config for a block, restoring the previous one after.

    Passing ``None`` starts from a blank slate (environment variables only).
    The previous config is restored on exit, whether the block succeeds or
    raises.
    """
    global _current_config, _config_overrides
    if snapshot is None:
        snapshot = (None, frozenset())
    previous = (_current_config, frozenset(_config_overrides))
    _current_config, _config_overrides = snapshot[0], set(snapshot[1])
    try:
        yield
    finally:
        _current_config, _config_overrides = previous[0], set(previous[1])
