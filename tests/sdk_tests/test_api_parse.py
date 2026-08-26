# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from tesseract_core.sdk.api_parse import ValidationError, validate_tesseract_api

LOGGER = logging.getLogger(__name__)


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
        version: "1.2.3-rc2"

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

    invalid_config = yaml.safe_load(valid_tesseract_config)
    invalid_config["env"] = {"VALID_KEY": 123}
    _write_tesseract_config_to_file(yaml.dump(invalid_config), tmp_path)

    with pytest.raises(ValidationError, match="should be a valid string"):
        validate_tesseract_api(tmp_path)


def test_config_with_metadata(tmp_path, valid_tesseract_api, valid_tesseract_config):
    _write_tesseract_api_to_file(valid_tesseract_api, tmp_path)

    config_with_metadata = yaml.safe_load(valid_tesseract_config)
    config_with_metadata["metadata"] = {
        "tags": ["ml", "physics"],
        "nested": {"key": "value"},
    }
    _write_tesseract_config_to_file(yaml.dump(config_with_metadata), tmp_path)
    validate_tesseract_api(tmp_path)


def test_config_metadata_defaults_to_empty(
    tmp_path, valid_tesseract_api, valid_tesseract_config
):
    _write_tesseract_api_to_file(valid_tesseract_api, tmp_path)
    _write_tesseract_config_to_file(valid_tesseract_config, tmp_path)

    from tesseract_core.sdk.api_parse import get_config

    config = get_config(tmp_path)
    assert config.metadata == {}


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


def test_config_with_python_version(
    tmp_path, valid_tesseract_api, valid_tesseract_config
):
    _write_tesseract_api_to_file(valid_tesseract_api, tmp_path)

    config_with_python_version = yaml.safe_load(valid_tesseract_config)
    config_with_python_version["build_config"]["python_version"] = "3.12"
    _write_tesseract_config_to_file(yaml.dump(config_with_python_version), tmp_path)
    validate_tesseract_api(tmp_path)

    from tesseract_core.sdk.api_parse import get_config

    config = get_config(tmp_path)
    assert config.build_config.python_version == "3.12"


def test_config_python_version_rejects_conda(
    tmp_path, valid_tesseract_api, valid_tesseract_config
):
    _write_tesseract_api_to_file(valid_tesseract_api, tmp_path)

    config = yaml.safe_load(valid_tesseract_config)
    config["build_config"]["python_version"] = "3.12"
    config["build_config"]["requirements"] = {"provider": "conda"}
    _write_tesseract_config_to_file(yaml.dump(config), tmp_path)

    with pytest.raises(
        ValidationError, match="python_version cannot be used with conda"
    ):
        validate_tesseract_api(tmp_path)


def test_config_python_version_rejects_inherit_base_image_packages(
    tmp_path, valid_tesseract_api, valid_tesseract_config
):
    _write_tesseract_api_to_file(valid_tesseract_api, tmp_path)

    config = yaml.safe_load(valid_tesseract_config)
    config["build_config"]["python_version"] = "3.12"
    config["build_config"]["inherit_base_image_packages"] = True
    _write_tesseract_config_to_file(yaml.dump(config), tmp_path)

    with pytest.raises(
        ValidationError,
        match="python_version cannot be used with inherit_base_image_packages",
    ):
        validate_tesseract_api(tmp_path)


def test_config_python_version_defaults_to_none(
    tmp_path, valid_tesseract_api, valid_tesseract_config
):
    _write_tesseract_api_to_file(valid_tesseract_api, tmp_path)
    _write_tesseract_config_to_file(valid_tesseract_config, tmp_path)

    from tesseract_core.sdk.api_parse import get_config

    config = get_config(tmp_path)
    assert config.build_config.python_version is None


def test_schema_parent_class_is_checked(
    tmp_path, valid_tesseract_api, valid_tesseract_config
):
    for schema in ("InputSchema", "OutputSchema"):
        tesseract_api = valid_tesseract_api.replace(f"{schema}(BaseModel)", schema)
        _write_tesseract_api_to_file(tesseract_api, tmp_path)
        _write_tesseract_config_to_file(valid_tesseract_config, tmp_path)

        with pytest.raises(
            ValidationError, match=f"{schema} must inherit from pydantic.BaseModel"
        ):
            validate_tesseract_api(tmp_path)


def test_generated_config_schema_is_wellformed():
    from tesseract_core.sdk.api_parse import CONFIG_SCHEMA_URL, generate_config_schema

    schema = generate_config_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == CONFIG_SCHEMA_URL
    assert schema["type"] == "object"
    # Unknown keys must be rejected by the editor, mirroring extra="forbid".
    assert schema["additionalProperties"] is False
    # Top-level properties must cover exactly the fields TesseractConfig accepts,
    # so the published schema never drifts from the model it is generated from.
    from tesseract_core.sdk.api_parse import TesseractConfig

    assert set(schema["properties"]) == set(TesseractConfig.model_fields)
    assert schema["required"] == ["name"]


def test_generated_config_schema_matches_get_config(
    tmp_path, valid_tesseract_config, monkeypatch
):
    """A config that passes the model must satisfy the generated schema, and vice versa.

    The generated schema is a first-line editor check, not a replacement for the
    Pydantic validators (e.g. the semantic version regex), so we only assert
    agreement on structural acceptance/rejection here.
    """
    jsonschema = pytest.importorskip("jsonschema")

    from tesseract_core.sdk.api_parse import generate_config_schema, get_config

    schema = generate_config_schema()
    jsonschema.Draft202012Validator.check_schema(schema)

    # A valid config satisfies both the model and the schema.
    _write_tesseract_config_to_file(valid_tesseract_config, tmp_path)
    config_dict = yaml.safe_load(valid_tesseract_config)
    get_config(tmp_path)  # does not raise
    jsonschema.validate(config_dict, schema)  # does not raise

    # The issue's counter-example (an unknown top-level key) is rejected.
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"name": "foo", "general": "kenobi"}, schema)


SCHEMASTORE_DIR = Path(__file__).parents[2] / "extra" / "schemastore"


def test_schemastore_fixtures_match_generated_schema(tmp_path):
    """The fixtures staged for the SchemaStore submission must stay valid.

    ``positive.yaml`` and ``negative.yaml`` are the artifacts we submit to
    SchemaStore, but nothing else exercises them, so they can silently drift out
    of sync with the schema. Validate them here so the positive one is provably
    accepted (and by the model) and the negative one provably rejected.
    """
    jsonschema = pytest.importorskip("jsonschema")

    from tesseract_core.sdk.api_parse import generate_config_schema, get_config

    schema = generate_config_schema()

    positive = (SCHEMASTORE_DIR / "positive.yaml").read_text()
    negative = (SCHEMASTORE_DIR / "negative.yaml").read_text()

    # The positive fixture must satisfy both the schema and the Pydantic model.
    jsonschema.validate(yaml.safe_load(positive), schema)  # does not raise
    _write_tesseract_config_to_file(positive, tmp_path)
    get_config(tmp_path)  # does not raise

    # The negative fixture must be rejected by the schema.
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(yaml.safe_load(negative), schema)
