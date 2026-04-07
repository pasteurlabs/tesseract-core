# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

from pydantic import BaseModel

from tesseract_core.runtime.config import get_config
from tesseract_core.runtime.experimental import (
    InputPathReference,
    OutputPathReference,
)


class InputSchema(BaseModel):
    paths: list[InputPathReference]


class OutputSchema(BaseModel):
    paths: list[OutputPathReference]


def apply(inputs: InputSchema) -> OutputSchema:
    output_path = Path(get_config().output_path)
    result = []
    for src in inputs.paths:
        if src.is_dir():
            dest = output_path / src.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            dest = output_path / src.with_suffix(".copy").name
            shutil.copy(src, dest)
        result.append(dest)
    return OutputSchema(paths=result)
