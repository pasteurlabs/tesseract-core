# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

from pydantic import BaseModel

from tesseract_core.runtime.config import get_config
from tesseract_core.runtime.experimental import (
    InputDirectoryReference,
    OutputDirectoryReference,
)


class InputSchema(BaseModel):
    dirs: list[InputDirectoryReference]


class OutputSchema(BaseModel):
    dirs: list[OutputDirectoryReference]


def apply(inputs: InputSchema) -> OutputSchema:
    output_path = Path(get_config().output_path)
    result = []
    for src in inputs.dirs:
        # src is an absolute Path to the input directory
        dest = output_path / src.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        result.append(dest)
    return OutputSchema(dirs=result)
