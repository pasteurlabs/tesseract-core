# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil

from pydantic import BaseModel

from tesseract_core.runtime.file_interactions import get_output_path
from tesseract_core.runtime.schema_types import InputFileReference, OutputFileReference


class InputSchema(BaseModel):
    data: list[InputFileReference]


class OutputSchema(BaseModel):
    data: list[OutputFileReference]


def apply(inputs: InputSchema) -> OutputSchema:
    files = []
    output_path = get_output_path()
    for source in inputs.data:
        target = output_path / source.name
        shutil.copy(source, target)
        files.append(target)
    return OutputSchema(data=files)
