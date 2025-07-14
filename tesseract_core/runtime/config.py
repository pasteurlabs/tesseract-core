# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, FilePath


class RuntimeConfig(BaseModel):
    """Available runtime configuration."""

    api_path: FilePath = Path("tesseract_api.py")
    required_files: str = ""
    name: str = "Tesseract"
    version: str = "0+unknown"
    debug: bool = False
    input_path: str = ""

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


def check_required_files(config: RuntimeConfig, path: Path) -> None:
    """Check if all required input files are present at path."""
    reqd_input_files = config.required_files.split(":")
    for file in reqd_input_files:
        file_path = path / file
        if not file_path.exists():
            # TODO: raise a different error here?
            raise FileNotFoundError(
                f"Required input file '{file}' not found in '{path}'. "
                "Please ensure that the required files are present."
            )
