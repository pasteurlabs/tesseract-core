# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel


# This would go somewhere in tesseract_core.runtime
def resolve_input_path(path: Path) -> Path:
    from tesseract_core.runtime.file_interactions import INPUT_PATH

    tess_path = INPUT_PATH / path
    if not tess_path.exists():
        raise FileNotFoundError(f"Input file {tess_path} does not exist.")
    return tess_path.resolve()


InputFileReference = Annotated[Path, AfterValidator(resolve_input_path)]
###################################################


class InputSchema(BaseModel):
    data: list[InputFileReference]


class OutputSchema(BaseModel):
    pass


def apply(inputs: InputSchema) -> OutputSchema:
    print(inputs.data)
    # prints:
    # [PosixPath('/Users/niklas/repos/pasteur/tesseract-core/examples/dataloader-filereference/testdata/sample_7.json'),
    # PosixPath('/Users/niklas/repos/pasteur/tesseract-core/examples/dataloader-filereference/testdata/sample_6.json'),
    # PosixPath('/Users/niklas/repos/pasteur/tesseract-core/examples/dataloader-filereference/testdata/sample_1.json'),
    # PosixPath('/Users/niklas/repos/pasteur/tesseract-core/examples/dataloader-filereference/testdata/sample_0.json'),
    # PosixPath('/Users/niklas/repos/pasteur/tesseract-core/examples/dataloader-filereference/testdata/sample_3.json'),
    # PosixPath('/Users/niklas/repos/pasteur/tesseract-core/examples/dataloader-filereference/testdata/sample_2.json'),
    # PosixPath('/Users/niklas/repos/pasteur/tesseract-core/examples/dataloader-filereference/testdata/sample_9.json'),
    # PosixPath('/Users/niklas/repos/pasteur/tesseract-core/examples/dataloader-filereference/testdata/sample_5.json'),
    # PosixPath('/Users/niklas/repos/pasteur/tesseract-core/examples/dataloader-filereference/testdata/sample_4.json'),
    # PosixPath('/Users/niklas/repos/pasteur/tesseract-core/examples/dataloader-filereference/testdata/sample_8.json')]
    return OutputSchema()
