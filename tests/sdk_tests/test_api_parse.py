# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from tesseract_core.sdk.api_parse import ValidationError, validate_tesseract_api


@pytest.fixture
def valid_tesseract_api() -> str:
    return dedent(
        """
        from typing import List
        from pydantic import BaseModel, Field

        class InputSchema(BaseModel):
            a: List[float] = Field(description="Just an argument.")

        class OutputSchema(BaseModel):
            result: List[float] = Field(description="Whatever.")

        def apply(inputs: InputSchema) -> OutputSchema:
            return OutputSchema(result=inputs.a)

        def abstract_eval(abstract_inputs):
            return {"result": abstract_inputs["a"]}

        # This isn't runtime-valid, but should pass static checks
        def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
            pass
        """
    )


@pytest.fixture
def valid_tesseract_config() -> str:
    return dedent(
        """
        name: foo
        version: "0+unknown"

        build_config:
            package_data:
              - ["path/to/source", "path/to/destination"]

            custom_build_steps:
              - RUN echo "Hello, World!"
        """
    )


def _write_tesseract_api_to_file(tesseract_api: str, path: Path):
    apifile = path / "tesseract_api.py"
    with open(apifile, "w") as file:
        file.write(tesseract_api)


def _write_tesseract_config_to_file(tesseract_config: str, path: Path):
    configfile = path / "tesseract_config.yaml"
    with open(configfile, "w") as file:
        file.write(tesseract_config)


def test_valid_input_passes_checks(
    tmp_path, valid_tesseract_api, valid_tesseract_config
):
    _write_tesseract_api_to_file(valid_tesseract_api, tmp_path)
    _write_tesseract_config_to_file(valid_tesseract_config, tmp_path)
    validate_tesseract_api(tmp_path)


def test_invalid_config_error(tmp_path, valid_tesseract_api, valid_tesseract_config):
    _write_tesseract_api_to_file(valid_tesseract_api, tmp_path)

    invalid_config = yaml.safe_load(valid_tesseract_config)
    invalid_config["version"] = 1
    _write_tesseract_config_to_file(yaml.dump(invalid_config), tmp_path)

    with pytest.raises(ValidationError, match="should be a valid string"):
        validate_tesseract_api(tmp_path)

    invalid_config = yaml.safe_load(valid_tesseract_config)
    invalid_config["build_config"]["custom_build_steps"] = [1]
    _write_tesseract_config_to_file(yaml.dump(invalid_config), tmp_path)

    with pytest.raises(ValidationError, match="should be a valid string"):
        validate_tesseract_api(tmp_path)

    invalid_config = yaml.safe_load(valid_tesseract_config)
    invalid_config["build_config"]["package_data"] = [["/etc/shadow", "/tmp"]]
    _write_tesseract_config_to_file(yaml.dump(invalid_config), tmp_path)

    with pytest.raises(ValidationError, match="must be a relative path"):
        validate_tesseract_api(tmp_path)


def test_api_not_defined_raises_filenotfound():
    path = Path("/non/existent/path")

    with pytest.raises(
        ValidationError,
        match="No file found at",
    ):
        validate_tesseract_api(path)


def test_invalid_syntax(tmp_path, valid_tesseract_config):
    tesseract_api = "!bad syntax!"
    _write_tesseract_api_to_file(tesseract_api, tmp_path)
    _write_tesseract_config_to_file(valid_tesseract_config, tmp_path)

    with pytest.raises(ValidationError, match="Syntax error"):
        validate_tesseract_api(tmp_path)


def test_missing_required_definition_error(
    tmp_path, valid_tesseract_api, valid_tesseract_config
):
    tesseract_api = valid_tesseract_api.replace("apply", "foobar")

    _write_tesseract_api_to_file(tesseract_api, tmp_path)
    _write_tesseract_config_to_file(valid_tesseract_config, tmp_path)

    with pytest.raises(ValidationError, match="apply not defined"):
        validate_tesseract_api(tmp_path)


def test_apply_signature_errors(tmp_path, valid_tesseract_api, valid_tesseract_config):
    _write_tesseract_config_to_file(valid_tesseract_config, tmp_path)

    tesseract_api = valid_tesseract_api.replace("apply(inputs:", "apply(*args, inputs:")
    _write_tesseract_api_to_file(tesseract_api, tmp_path)
    with pytest.raises(ValidationError, match="keyword-only"):
        validate_tesseract_api(tmp_path)

    tesseract_api = valid_tesseract_api.replace("apply(inputs: ", "apply(*, inputs:")
    _write_tesseract_api_to_file(tesseract_api, tmp_path)
    with pytest.raises(ValidationError, match="keyword-only"):
        validate_tesseract_api(tmp_path)

    tesseract_api = valid_tesseract_api.replace(
        "apply(inputs: InputSchema", "apply(inputs: InputSchema, /"
    )
    _write_tesseract_api_to_file(tesseract_api, tmp_path)
    with pytest.raises(ValidationError, match="positional-only"):
        validate_tesseract_api(tmp_path)

    tesseract_api = valid_tesseract_api.replace(
        "(inputs: InputSchema)",
        "(inputs: InputSchema, sneezy)",
    )
    _write_tesseract_api_to_file(tesseract_api, tmp_path)
    with pytest.raises(ValidationError, match="apply must have 1 argument"):
        validate_tesseract_api(tmp_path)


def test_optional_signature_check(
    tmp_path, valid_tesseract_api, valid_tesseract_config
):
    _write_tesseract_config_to_file(valid_tesseract_config, tmp_path)

    tesseract_api = valid_tesseract_api.replace(
        "jacobian(inputs:", "jacobian(foo, inputs:"
    )
    _write_tesseract_api_to_file(tesseract_api, tmp_path)
    with pytest.raises(ValidationError, match="jacobian must have 3 arguments"):
        validate_tesseract_api(tmp_path)


def test_schema_parent_class_is_checked(
    tmp_path, valid_tesseract_api, valid_tesseract_config
):
    tesseract_api = valid_tesseract_api.replace("(BaseModel)", "")
    _write_tesseract_api_to_file(tesseract_api, tmp_path)
    _write_tesseract_config_to_file(valid_tesseract_config, tmp_path)

    with pytest.raises(ValidationError, match="subclass of pydantic.BaseModel"):
        validate_tesseract_api(tmp_path)
