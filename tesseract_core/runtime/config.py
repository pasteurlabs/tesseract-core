# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, FilePath


class RuntimeConfig(BaseModel):
    """Available runtime configuration."""

    tesseract_api_path: FilePath = Path("tesseract_api.py")
    tesseract_name: str = "Tesseract"
    tesseract_version: str = "0+unknown"

    model_config = ConfigDict(frozen=True, extra="forbid")


def update_config(**kwargs: Any) -> None:
    """Create a new runtime configuration from the current environment.

    Passed keyword arguments will override environment variables.
    """
    global _current_config

    conf_settings = {}
    for field in RuntimeConfig.model_fields.keys():
        env_key = field.upper()
        if env_key in os.environ:
            conf_settings[field] = os.environ[env_key]

    conf_settings.update(kwargs)

    config = RuntimeConfig(**conf_settings)
    _current_config = config


_current_config = None
update_config()


def get_config() -> RuntimeConfig:
    """Return the current runtime configuration."""
    return _current_config
