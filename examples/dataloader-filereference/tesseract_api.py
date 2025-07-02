# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel


def resolve_input_path(path: Path) -> Path:
    from tesseract_core.runtime.file_interactions import INPUT_PATH

    tess_path = INPUT_PATH / path
    if not tess_path.exists():
        raise FileNotFoundError(f"Input file {tess_path} does not exist.")
    return tess_path.resolve()


InputFileReference = Annotated[Path, AfterValidator(resolve_input_path)]


class InputSchema(BaseModel):
    # NOTE: no file references here
    data: list[InputFileReference]


class OutputSchema(BaseModel):
    pass


def apply(inputs: InputSchema) -> OutputSchema:
    print(inputs.data)
    return OutputSchema()
