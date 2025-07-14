# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil

from pydantic import BaseModel

from tesseract_core.runtime import (
    InputFileReference,
    OutputFileReference,
    get_output_path,
)


class InputSchema(BaseModel):
    data: list[InputFileReference]


class OutputSchema(BaseModel):
    data: list[OutputFileReference]


def apply(inputs: InputSchema) -> OutputSchema:
    output_path = get_output_path()
    files = []
    for source in inputs.data:
        target = output_path / source.name
        target = target.with_suffix(".copy")
        shutil.copy(source, target)
        files.append(target)
    return OutputSchema(data=files)
