# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, FilePath


class RuntimeConfig(BaseModel):
    """Available runtime configuration."""

    api_path: FilePath = Path("tesseract_api.py")
    name: str = "Tesseract"
    version: str = "0+unknown"
    debug: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def input_path(self) -> Path:
        """Read tesseract input path from environment variable."""
        path = os.environ.get("TESSERACT_INPUT_PATH", None)
        if path is None:
            raise ValueError("TESSERACT_INPUT_PATH environment variable is not set.")
        return Path(path).resolve()

    @property
    def output_path(self) -> Path:
        """Read tesseract output path from environment variable."""
        path = os.environ.get("TESSERACT_OUTPUT_PATH", None)
        if path is None:
            raise ValueError("TESSERACT_OUTPUT_PATH environment variable is not set.")
        return Path(path).resolve()


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
