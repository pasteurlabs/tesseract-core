# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from typing import Annotated, NamedTuple

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    Strict,
)
from pydantic import ValidationError as PydanticValidationError


class _ApiObject(NamedTuple):
    name: str
    expected_type: type
    num_args: int | None = None
    optional: bool = False


EXPECTED_OBJECTS = (
    _ApiObject("apply", ast.FunctionDef, 1),
    _ApiObject("InputSchema", ast.ClassDef),
    _ApiObject("OutputSchema", ast.ClassDef),
    _ApiObject("jacobian", ast.FunctionDef, 3, optional=True),
    _ApiObject("jacobian_vector_product", ast.FunctionDef, 4, optional=True),
    _ApiObject("vector_jacobian_product", ast.FunctionDef, 4, optional=True),
    _ApiObject("abstract_eval", ast.FunctionDef, 1, optional=True),
)


def assert_relative_path(value: str) -> str:
    """Assert that a string encodes a relative path."""
    if Path(value).is_absolute():
        raise ValueError(f"value must be a relative path (got {value})")
    return value


RelativePath = Annotated[str, AfterValidator(assert_relative_path)]
StrictStr = Annotated[str, Strict()]


class TesseractBuildConfig(BaseModel):
    """Configuration options for building a Tesseract."""

    base_image: StrictStr = Field(
        "python:3.12-slim-bookworm",
        description="Base Docker image for the build. Must be Debian-based.",
    )
    target_platform: StrictStr = Field(
        "native",
        description=(
            "Target platform for the Docker build. Must be a valid Docker platform, "
            "or 'native' to build for the host platform. "
            "In general, images built for one platform will not run on another."
        ),
    )
    extra_packages: tuple[StrictStr, ...] = Field(
        (), description="Extra packages to install during build via apt-get."
    )
    package_data: tuple[tuple[RelativePath, StrictStr], ...] | None = Field(
        (),
        description=(
            "Additional files to copy into the Docker image, in the format ``(source, destination)``. "
            "Source paths are relative to the Tesseract directory."
        ),
    )
    custom_build_steps: tuple[StrictStr, ...] | None = Field(
        (),
        description=(
            "Custom steps to run during ``docker build`` (after everything else is installed). "
            "Example: ``[\"RUN echo 'Hello, world!'\"]``"
        ),
    )

    model_config = ConfigDict(extra="forbid")


# Allow None to be passed as a valid value for build_config, for example in YAML that comments out all options.
OptionalBuildConfig = Annotated[
    TesseractBuildConfig,
    BeforeValidator(lambda v: TesseractBuildConfig() if v is None else v),
]


class TesseractConfig(BaseModel):
    """Configuration options for Tesseracts. Defines valid options in ``tesseract_config.yaml``."""

    name: StrictStr = Field(..., description="Name of the Tesseract.")
    version: StrictStr = Field("0+unknown", description="Version of the Tesseract.")
    description: StrictStr = Field(
        "",
        description="Free-text description of what the Tesseract does.",
    )
    build_config: OptionalBuildConfig = Field(
        default_factory=TesseractBuildConfig,
        description="Configuration options for building the Tesseract.",
    )

    model_config = ConfigDict(extra="forbid")


class ValidationError(Exception):
    """Raised when inputs needed to build a tesseract are invalid."""

    pass


def _get_func_argnums(func: ast.FunctionDef) -> int:
    """Get the number of the arguments of a function node.

    See:
    https://docs.python.org/3/library/ast.html#ast.FunctionDef
    https://docs.python.org/3/library/ast.html#ast.arguments
    """
    func_args = func.args
    if func_args.kwonlyargs:
        raise ValidationError(
            f"Function {func.name} must not have keyword-only arguments"
        )
    if func_args.posonlyargs:
        raise ValidationError(
            f"Function {func.name} must not have positional-only arguments"
        )
    return len(func_args.args)


def validate_tesseract_api(src_dir: Path) -> None:
    """Check that given folder contains a Tesseract API that satisfies our constraints.

    This function does not return anything, but it raises `ValidationError` in
    case something goes wrong. In particular, we are checking that:
      *  The mandatory endpoints needed for a tesseract are actually
         implemented
      *  The implemented functions have the correct signature
      *  Both InputSchema and OutputSchema are `pydantic.BaseModel`s.

    Args:
        src_dir (Path): Path to the directory containing tesseract_api.py and tesseract_config.yaml.
    """
    tesseract_api_location = src_dir / "tesseract_api.py"
    config_location = src_dir / "tesseract_config.yaml"

    if not tesseract_api_location.exists():
        raise ValidationError(f"No file found at {tesseract_api_location}")

    if not config_location.exists():
        raise ValidationError(f"No file found at {config_location}")

    # Validate config
    try:
        get_config(src_dir)
    except PydanticValidationError as err:
        raise ValidationError(
            f"Invalid configuration in {config_location}: {err}"
        ) from err

    # Parse Tesseract API
    with open(tesseract_api_location) as f:
        tesseract_api_code = f.read()

    try:
        tesseract_api = ast.parse(tesseract_api_code)
    except SyntaxError as err:
        raise ValidationError(
            f"Syntax error in {tesseract_api_location}: {err}"
        ) from err

    # Check if expected attributes are defined
    toplevel_objects = {
        node.name: node for node in tesseract_api.body if hasattr(node, "name")
    }

    for obj in EXPECTED_OBJECTS:
        if obj.name not in toplevel_objects:
            if obj.optional:
                continue

            raise ValidationError(f"{obj.name} not defined in {tesseract_api_location}")

        if not isinstance(toplevel_objects[obj.name], obj.expected_type):
            raise ValidationError(
                f"{obj.name} is not a {obj.expected_type.__name__} in {tesseract_api_location}"
            )

        if obj.num_args is not None:
            if _get_func_argnums(toplevel_objects[obj.name]) != obj.num_args:
                raise ValidationError(f"{obj.name} must have {obj.num_args} arguments")

    # Check InputSchema and OutputSchema are pydantic BaseModels
    for schema in ("InputSchema", "OutputSchema"):
        obj = toplevel_objects[schema]
        if not obj.bases:
            subclass = None
        else:
            subclass = obj.bases[0].id

        if subclass != "BaseModel":
            raise ValidationError(
                f"{schema} must be a subclass of pydantic.BaseModel (got: {subclass})"
            )


def get_config(src_dir: Path) -> TesseractConfig:
    """Get configuration options from a tesseract_config.yaml file."""
    config_file = src_dir / "tesseract_config.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"No file found at {config_file}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    try:
        return TesseractConfig(**config)
    except PydanticValidationError as err:
        raise ValidationError(f"Invalid configuration: {err}") from err


def get_non_base_fields_in_tesseract_config() -> list[tuple[str, type]]:
    """Gets fields in Tesseract Config that are not a base fields."""
    base_fields = (str, int, float, bool, bytes)
    non_base_fields = []
    for field_name, field_info in TesseractConfig.model_fields.items():
        if field_info.annotation not in base_fields:
            non_base_fields.append((field_name, field_info.annotation))
    return non_base_fields
