# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from pathlib import Path
from typing import (
    Annotated,
    Any,
)

from pydantic import (
    AfterValidator,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    ValidationInfo,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from tesseract_core.runtime.config import get_config
from tesseract_core.runtime.file_interactions import PathLike


class InputPath(Path):
    """Path type for Tesseract input files.

    Resolves relative paths against ``RuntimeConfig().input_path`` and validates
    that the resolved path stays within the input directory and exists.

    Use ``Annotated[InputPath, AfterValidator(...)]`` to add custom validation
    that runs after path resolution (receives the absolute, resolved path).
    """

    @classmethod
    def _resolve(cls, path: Path, info: ValidationInfo | None) -> Path:
        ctx = info.context if info else None
        skip = ctx.get("skip_path_checks", False) if ctx else False
        input_path = Path(get_config().input_path).resolve()
        tess_path = (input_path / path).resolve()
        if not tess_path.is_relative_to(input_path):
            raise ValueError(
                f"Invalid input file reference: {path}. "
                f"Expected path to be relative to {input_path}, but got {tess_path}. "
                "File references have to be relative to --input-path."
            )
        if not skip and not tess_path.exists():
            raise ValueError(f"Input path {tess_path} does not exist.")
        return tess_path

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.with_info_after_validator_function(
                cls._validate,
                core_schema.str_schema(),
            ),
            python_schema=core_schema.with_info_after_validator_function(
                cls._validate,
                core_schema.union_schema(
                    [
                        core_schema.is_instance_schema(Path),
                        core_schema.str_schema(),
                    ]
                ),
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda p: str(p),
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        schema = handler(_core_schema)
        schema["format"] = "path"
        return schema

    @classmethod
    def _validate(cls, value: Any, info: ValidationInfo) -> Path:
        return cls._resolve(Path(value), info)


class OutputPath(Path):
    """Path type for Tesseract output files.

    Validates that paths exist inside ``RuntimeConfig().output_path`` and
    strips the prefix when serializing.

    In-memory values are **absolute** paths. Use
    ``Annotated[OutputPath, AfterValidator(...)]`` to add custom validation;
    validators receive absolute paths.
    """

    @classmethod
    def _validate(cls, value: Any, info: ValidationInfo) -> Path:
        """Resolve to absolute, check containment and existence."""
        output_path = Path(get_config().output_path).resolve()
        path = Path(value)
        resolved = (
            (output_path / path).resolve() if not path.is_absolute() else path.resolve()
        )
        if not resolved.is_relative_to(output_path):
            raise ValueError(
                f"Output path {path} escapes output directory {output_path}."
            )

        ctx = info.context if info else None
        skip = ctx.get("skip_path_checks", False) if ctx else False
        if not skip and not resolved.exists():
            if path.is_relative_to(output_path) or not path.is_absolute():
                raise ValueError(
                    f"Output path {resolved} does not exist inside Tesseract"
                )
            elif path.exists():
                raise ValueError(
                    f"Output path {path} is not in {output_path}. "
                    f"All output data must be copied to `--output-path` ({output_path})."
                )
            else:
                raise ValueError(
                    f"Output path {path} is not in {output_path} or Tesseract root"
                )
        return resolved

    @classmethod
    def _serialize(cls, path: Path) -> str:
        """Strip output_path prefix for serialization."""
        output_path = Path(get_config().output_path).resolve()
        if path.is_relative_to(output_path):
            return str(path.relative_to(output_path))
        return str(path)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.with_info_after_validator_function(
                cls._validate,
                core_schema.str_schema(),
            ),
            python_schema=core_schema.with_info_after_validator_function(
                cls._validate,
                core_schema.union_schema(
                    [
                        core_schema.is_instance_schema(Path),
                        core_schema.str_schema(),
                    ]
                ),
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize,
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        schema = handler(_core_schema)
        schema["format"] = "path"
        return schema


def _resolve_input_file(path: Path, info: ValidationInfo) -> Path:
    warnings.warn(
        "InputFileReference is deprecated, use InputPath instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    tess_path = InputPath._resolve(path, info)
    ctx = info.context if info else None
    skip = ctx.get("skip_path_checks", False) if ctx else False
    if not skip and not tess_path.is_file():
        raise ValueError(f"Input path {tess_path} is not a file.")
    return tess_path


def _strip_output_file(path: Path, info: ValidationInfo) -> Path:
    warnings.warn(
        "OutputFileReference is deprecated, use OutputPath instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    ctx = info.context if info else None
    skip = ctx.get("skip_path_checks", False) if ctx else False
    # Resolve relative paths against output_path so the file check works on
    # both absolute paths returned from apply() and on relative paths that
    # come back through a re-validation pass.
    output_path = Path(get_config().output_path).resolve()
    resolved = (output_path / path).resolve() if not path.is_absolute() else path
    if not skip and not resolved.is_file():
        raise ValueError(f"Output path {path} is not a file.")
    return Path(OutputPath._serialize(resolved))


InputFileReference = Annotated[Path, AfterValidator(_resolve_input_file)]
OutputFileReference = Annotated[Path, AfterValidator(_strip_output_file)]


def require_file(file_path: PathLike) -> Path:
    """Designate a file which is required to be present at runtime.

    Args:
        file_path: Path to required file. Must be relative to `input_path` assigned in `tesseract run`.
    """
    # Read the flag through the package namespace so that runtime mutations of
    # ``tesseract_core.runtime.experimental.SKIP_REQUIRED_FILE_CHECK`` (done by
    # the CLI / during build) are picked up here.
    from tesseract_core.runtime import experimental

    if experimental.SKIP_REQUIRED_FILE_CHECK:
        return Path(file_path)

    file_path = InputPath._resolve(Path(file_path), None)

    if not file_path.is_file():
        raise FileNotFoundError(f"Required file not found: {file_path}")

    return file_path
