# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated

import pytest
from pydantic import AfterValidator, BaseModel, ValidationError

from tesseract_core.runtime.config import update_config
from tesseract_core.runtime.experimental import InputPath, OutputPath


@pytest.fixture(autouse=True)
def path_dirs(tmp_path):
    """Set up input/output paths for all tests in this module."""
    input_dir = (tmp_path / "input").resolve()
    output_dir = (tmp_path / "output").resolve()
    input_dir.mkdir()
    output_dir.mkdir()
    update_config(
        input_path=str(input_dir),
        output_path=str(output_dir),
    )
    return input_dir, output_dir


# -- Schema generation --


class PathModel(BaseModel):
    inp: InputPath
    out: OutputPath


class ListPathModel(BaseModel):
    inputs: list[InputPath]
    outputs: list[OutputPath]


def test_json_schema_basic():
    schema = PathModel.model_json_schema()
    assert schema["properties"]["inp"] == {"title": "Inp", "type": "string"}
    assert schema["properties"]["out"] == {"title": "Out", "type": "string"}
    assert set(schema["required"]) == {"inp", "out"}


def test_json_schema_in_list():
    schema = ListPathModel.model_json_schema()
    inp_prop = schema["properties"]["inputs"]
    assert inp_prop["type"] == "array"
    assert inp_prop["items"]["type"] == "string"


def test_json_schema_with_annotated_validator():
    """AfterValidator on top of InputPath/OutputPath should not change the schema."""

    def noop(p: Path) -> Path:
        return p

    class M(BaseModel):
        inp: Annotated[InputPath, AfterValidator(noop)]
        out: Annotated[OutputPath, AfterValidator(noop)]

    schema = M.model_json_schema()
    assert schema["properties"]["inp"]["type"] == "string"
    assert schema["properties"]["out"]["type"] == "string"


# -- InputPath validation --


def test_input_path_resolves(path_dirs):
    input_dir, _ = path_dirs
    (input_dir / "data.json").touch()

    class M(BaseModel):
        inp: InputPath

    model = M.model_validate({"inp": "data.json"})
    assert model.inp == input_dir / "data.json"
    assert model.inp.is_absolute()


def test_input_path_rejects_missing():
    class M(BaseModel):
        inp: InputPath

    with pytest.raises(ValidationError, match="does not exist"):
        M.model_validate({"inp": "nonexistent.json"})


def test_input_path_rejects_traversal():
    class M(BaseModel):
        inp: InputPath

    with pytest.raises(ValidationError, match="relative to"):
        M.model_validate(
            {"inp": "../escape.txt"},
            context={"skip_path_checks": True},
        )


def test_input_path_skip_checks():
    """With skip_path_checks, missing files are accepted."""

    class M(BaseModel):
        inp: InputPath

    model = M.model_validate(
        {"inp": "missing.json"},
        context={"skip_path_checks": True},
    )
    assert model.inp.name == "missing.json"


# -- OutputPath validation --


def test_output_path_strips_prefix(path_dirs):
    _, output_dir = path_dirs
    (output_dir / "result.json").touch()

    class M(BaseModel):
        out: OutputPath

    model = M.model_validate({"out": str(output_dir / "result.json")})
    assert model.out == Path("result.json")


def test_output_path_accepts_relative(path_dirs):
    _, output_dir = path_dirs
    (output_dir / "result.json").touch()

    class M(BaseModel):
        out: OutputPath

    model = M.model_validate({"out": "result.json"})
    assert model.out == Path("result.json")


def test_output_path_rejects_missing(path_dirs):
    _, output_dir = path_dirs

    class M(BaseModel):
        out: OutputPath

    with pytest.raises(ValidationError, match="does not exist"):
        M.model_validate({"out": str(output_dir / "ghost.txt")})


def test_output_path_rejects_escape():
    class M(BaseModel):
        out: OutputPath

    with pytest.raises(ValidationError, match="escapes output directory"):
        M.model_validate(
            {"out": "/some/other/place.txt"},
            context={"skip_path_checks": True},
        )


def test_output_path_skip_checks(path_dirs):
    _, output_dir = path_dirs

    class M(BaseModel):
        out: OutputPath

    model = M.model_validate(
        {"out": str(output_dir / "future.bin")},
        context={"skip_path_checks": True},
    )
    assert model.out == Path("future.bin")


# -- Serialization roundtrip --


def test_serialization_roundtrip(path_dirs):
    input_dir, output_dir = path_dirs
    (input_dir / "a.txt").touch()
    (output_dir / "b.txt").touch()

    model = PathModel.model_validate({"inp": "a.txt", "out": str(output_dir / "b.txt")})
    dumped = model.model_dump()
    assert isinstance(dumped["inp"], str)
    assert isinstance(dumped["out"], str)

    json_str = model.model_dump_json()
    assert "a.txt" in json_str
    assert "b.txt" in json_str


# -- Annotated composition --


def test_annotated_input_path_runs_extra_validator(path_dirs):
    input_dir, _ = path_dirs
    (input_dir / "good.json").touch()
    (input_dir / "bad.txt").touch()

    def require_json(path: Path) -> Path:
        if path.suffix != ".json":
            raise ValueError("Must be a .json file")
        return path

    class M(BaseModel):
        f: Annotated[InputPath, AfterValidator(require_json)]

    m = M.model_validate({"f": "good.json"})
    assert m.f.name == "good.json"

    with pytest.raises(ValidationError, match=r"Must be a \.json file"):
        M.model_validate({"f": "bad.txt"})


def test_annotated_output_path_runs_extra_validator(path_dirs):
    _, output_dir = path_dirs
    (output_dir / "result.csv").touch()

    def require_csv(path: Path) -> Path:
        if path.suffix != ".csv":
            raise ValueError("Must be a .csv file")
        return path

    class M(BaseModel):
        f: Annotated[OutputPath, AfterValidator(require_csv)]

    m = M.model_validate({"f": str(output_dir / "result.csv")})
    assert m.f == Path("result.csv")
