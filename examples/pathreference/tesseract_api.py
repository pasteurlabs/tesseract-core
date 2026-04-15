# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

from pydantic import AfterValidator, BaseModel

from tesseract_core.runtime.config import get_config
from tesseract_core.runtime.experimental import (
    InputPathReference,
    OutputPathReference,
    compose_validator,
)


def bin_reference(path: Path) -> str | None:
    """Return the name of the .bin file if the json at 'path' references one, else None."""
    with open(path) as f:
        contents = json.load(f)
        if contents["data"]["encoding"] == "binref":
            return contents["data"]["buffer"].split(":")[0]
    return None


def has_bin_sidecar(path: Path) -> Path:
    """Pydantic validator to check for .bin file next to any json file that references one."""
    if path.is_file():
        name = bin_reference(path)
        if name is not None:
            bin = path.parent / name
            assert bin.exists(), (
                f"Expected .bin file for json {json} not found at {bin}"
            )
    elif path.is_dir():
        return path
    else:
        raise ValueError(f"{path} does not exist.")
    return path


InputPath = compose_validator(InputPathReference, AfterValidator(has_bin_sidecar))
OutputPath = compose_validator(OutputPathReference, AfterValidator(has_bin_sidecar))


class InputSchema(BaseModel):
    # paths: list[Annotated[InputPathReference, AfterValidator(has_bin_sidecar)]]
    paths: list[InputPath]


class OutputSchema(BaseModel):
    # paths: list[Annotated[OutputPathReference, AfterValidator(has_bin_sidecar)]]
    paths: list[OutputPath]


def apply(inputs: InputSchema) -> OutputSchema:
    output_path = Path(get_config().output_path)
    result = []

    for src in inputs.paths:
        if src.is_dir():
            # copy any folder that is given
            dest = output_path / src.name
            shutil.copytree(src, dest)
        else:
            # copy any file that is given, and if it references a .bin file, copy that too
            dest = output_path / src.with_suffix(".copy").name
            shutil.copy(src, dest)
            bin = bin_reference(src)
            if bin is not None:
                shutil.copy(src.parent / bin, dest.parent / bin)
        result.append(dest)
    return OutputSchema(paths=result)
