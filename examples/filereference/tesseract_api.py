# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

from pydantic import BaseModel

from tesseract_core.runtime import (
    InputFileReference,
    OutputFileReference,
)
from tesseract_core.runtime.config import get_config


class InputSchema(BaseModel):
    data: list[InputFileReference]


class OutputSchema(BaseModel):
    data: list[OutputFileReference]


def apply(inputs: InputSchema) -> OutputSchema:
    output_path = Path(get_config().output_path)
    files = []
    for source in inputs.data:
        # source is a pathlib.Path starting with /path/to/input_path/...
        target = output_path / source.name
        # target must be a pathlib.Path at /path/to/output_path
        target = target.with_suffix(".copy")
        shutil.copy(source, target)
        files.append(target)
    return OutputSchema(data=files)
