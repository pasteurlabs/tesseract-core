import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel
from typeguard import suppress_type_checks

from tesseract_core.runtime import Array, Float32
from tesseract_core.runtime.file_interactions import output_to_bytes


class OutputSchema(BaseModel):
    scalar: float
    string: str
    list: list[float]
    array: Array[(3,), Float32]
    nested: dict


@pytest.fixture
def output_data():
    return OutputSchema(
        scalar=1.0,
        string="hello 世界 🌍",
        list=[1.0, 2.0, 3.0],
        array=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        nested={"inner": [4.0, 5.0]},
    )


def _check_decoded_data(decoded: dict, array_encoding: str):
    """Assert basic scalar and nested fields are correct and that the array field is properly encoded."""
    assert decoded["scalar"] == 1.0
    assert decoded["string"] == "hello 世界 🌍"
    assert decoded["nested"]["inner"] == [4.0, 5.0]
    assert decoded["list"] == [1.0, 2.0, 3.0]
    assert decoded["array"]["object_type"] == "array"
    assert decoded["array"]["data"]["encoding"] == array_encoding
    # Here we do not check array contents, subject to separate array encoding unit tests


def test_output_to_bytes_json(output_data):
    result = output_to_bytes(output_data, "json")
    assert isinstance(result, bytes)

    decoded = json.loads(result.decode())
    _check_decoded_data(decoded, "json")


def test_output_to_bytes_json_base64(output_data):
    result = output_to_bytes(output_data, "json+base64")
    assert isinstance(result, bytes)

    decoded = json.loads(result.decode())
    _check_decoded_data(decoded, "base64")


def test_output_to_bytes_json_binref(output_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        binref_dir = base_dir / "binrefs"

        result = output_to_bytes(
            output_data, "json+binref", base_dir=base_dir, binref_dir=binref_dir
        )
        assert isinstance(result, bytes)

        decoded = json.loads(result.decode())
        _check_decoded_data(decoded, "binref")


def test_output_to_bytes_unsupported_format(output_data):
    with pytest.raises(ValueError, match="Unsupported format invalid"):
        with suppress_type_checks():
            output_to_bytes(output_data, "invalid")  # type: ignore


def test_output_to_bytes_empty_dict():
    """Test with empty dictionary."""
    result = output_to_bytes({}, "json")
    assert isinstance(result, bytes)
    decoded = json.loads(result.decode())
    assert decoded == {}


def test_output_to_bytes_scalar_only():
    """Test with just a scalar value."""
    result = output_to_bytes(42.0, "json")
    assert isinstance(result, bytes)
    decoded = json.loads(result.decode())
    assert decoded == 42.0
