# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from .api_parse import ValidationError


def validate_executable(value: str | Sequence[str]) -> tuple[str, ...]:
    """Get the path to the requested program."""
    if isinstance(value, str):
        exe, *args = shlex.split(value)
    else:
        exe, *args = value

    exe = shutil.which(exe)
    if exe is None:
        raise FileNotFoundError(f"Executable `{value}` not found.")

    exe_path = Path(exe)
    if not exe_path.is_file():
        raise FileNotFoundError(f"{exe} is not a file.")
    if not os.access(exe_path, os.X_OK):
        raise PermissionError(f"{exe} is not executable.")

    return (str(exe_path.resolve()), *args)


def maybe_split_args(value: str | Sequence[str]) -> tuple[str, ...]:
    """Split arguments if they are passed as a string."""
    if isinstance(value, str):
        value = shlex.split(value)
    return tuple(value)


class RuntimeConfig(BaseModel):
    """Available runtime configuration."""

    docker_executable: Annotated[
        tuple[str, ...], BeforeValidator(validate_executable)
    ] = ("docker",)
    docker_compose_executable: Annotated[
        tuple[str, ...], BeforeValidator(validate_executable)
    ] = ("docker", "compose")
    docker_build_args: Annotated[
        tuple[str, ...], BeforeValidator(maybe_split_args)
    ] = ()

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

    conf_settings.update(kwargs)

    try:
        config = RuntimeConfig(**conf_settings)
    except PydanticValidationError as err:
        raise ValidationError(f"Invalid configuration: {err}") from err
    except (FileNotFoundError, PermissionError) as err:
        raise ValidationError(f"Executable not found or not executable: {err}") from err
    _current_config = config


_current_config = None


def get_config() -> RuntimeConfig:
    """Return the current runtime configuration."""
    if _current_config is None:
        update_config()
<<<<<<< ako/testexit
=======

>>>>>>> heitor/fix-docker-not-found
    return _current_config
