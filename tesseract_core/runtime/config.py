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

from tesseract_core.runtime.file_interactions import supported_format_type


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
    # Experimental, unstable opt-in: allow the json+cuda_ipc output format, which
    # passes GPU arrays by CUDA IPC handle instead of serializing their data.
    # Off by default so a Tesseract never produces IPC handles unless explicitly
    # enabled (e.g. TESSERACT_ENABLE_EXPERIMENTAL_CUDA_IPC=1). May change or be
    # removed without notice.
    enable_experimental_cuda_ipc: bool = False

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


def reset_config() -> None:
    """Discard all remembered runtime configuration overrides.

    update_config() re-applies every previously overridden field on each
    call, which a single Tesseract's config setup relies on to accumulate
    correctly across its several update_config() calls (input_path, then
    output_path, then general kwargs). But that same stickiness means a
    later, independent in-process Tesseract silently inherits an earlier
    one's explicit overrides instead of starting from environment variables
    and its own arguments. Callers that build config for a fresh instance
    should call this first.
    """
    global _current_config, _config_overrides
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
    """Capture the current (config, overrides) pair, to restore later.

    A Tesseract built via from_tesseract_api takes one right after its own
    update_config() calls settle, so its endpoints can later run under
    exactly that config; see active_config().
    """
    return _current_config, frozenset(_config_overrides)


@contextmanager
def active_config(snapshot: ConfigSnapshot) -> Iterator[None]:
    """Run a block under `snapshot` as the process-global runtime config.

    Restores whatever was active before on exit, whether the block succeeds
    or raises. Config is process-global (get_config()/update_config() share
    module state), which both a single Tesseract's own setup and this
    function itself rely on to freely rebuild it from scratch without
    disturbing anyone else. Two distinct uses: `from_tesseract_api` wraps
    its whole construction in this so a prior or concurrent in-process
    Tesseract's config is untouched once construction returns; `run_tesseract`
    wraps a single call in this so the config it sees matches the instance
    it belongs to, regardless of what any other in-process Tesseract's
    construction or calls do to the global in between.
    """
    global _current_config, _config_overrides
    previous = (_current_config, frozenset(_config_overrides))
    _current_config, _config_overrides = snapshot[0], set(snapshot[1])
    try:
        yield
    finally:
        _current_config, _config_overrides = previous[0], set(previous[1])
