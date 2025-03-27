# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from pathlib import Path

import jsf
import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from tesseract_core.runtime.experimental import LazySequence
from tesseract_core.runtime.schema_types import (
    Array,
    ArrayFlags,
    Bool,
    Differentiable,
    Float64,
    Int32,
    Int64,
    is_differentiable,
)


class MyModel(BaseModel):
    array_int: Array[(2, 3), Int64]
    array_float: Differentiable[Array[(None, 3), Float64]]
    array_bool: Array[..., Bool]
    scalar_int: Differentiable[Int32]


arr_int = np.array([[1, 2, 3], [4, 5, 6]])
arr_float = np.array([[1.0, 2.0, 3.0]])
arr_bool = np.array([True, False, True]).reshape(1, 1, 3)
scalar_int = np.int32(42)


def test_generic_array_type_repr():
    typ = Array[(2, 3), Int64]
    assert repr(typ) == "typing.Annotated[numpy.ndarray, Array[(2, 3), 'int64']]"


def test_flag_propagation():
    get_flags = lambda model, field: model.model_fields[field].metadata[0].flags
    assert get_flags(MyModel, "array_int") == ()
    assert get_flags(MyModel, "array_float") == (ArrayFlags.DIFFERENTIABLE,)
    assert get_flags(MyModel, "array_bool") == ()
    assert get_flags(MyModel, "scalar_int") == (ArrayFlags.DIFFERENTIABLE,)


def test_is_differentiable():
    assert is_differentiable(Array[(2, 3), Int64]) is False
    assert is_differentiable(Differentiable[Array[(2, 3), Int64]]) is True
    assert is_differentiable(Differentiable[Int32]) is True
    assert is_differentiable(Int32) is False


def test_model_creation():
    model = MyModel(
        array_int=arr_int,
        array_float=arr_float,
        array_bool=arr_bool,
        scalar_int=scalar_int,
    )

    assert isinstance(model, MyModel)
    assert isinstance(model.array_int, np.ndarray)
    assert isinstance(model.array_float, np.ndarray)
    assert isinstance(model.array_bool, np.ndarray)
    assert isinstance(model.scalar_int, np.int32)
    assert model.array_int.shape == (2, 3)
    assert model.array_float.shape == (1, 3)
    assert model.array_bool.shape == (1, 1, 3)

    assert np.array_equal(model.array_int, arr_int)
    assert np.array_equal(model.array_float, arr_float.reshape(1, 3))
    assert np.array_equal(model.array_bool, arr_bool)
    assert model.scalar_int == scalar_int


def test_model_creation_from_pyobj():
    model = MyModel(
        array_int=arr_int.tolist(),
        array_float=arr_float.tolist(),
        array_bool=arr_bool.tolist(),
        scalar_int=int(scalar_int),
    )

    assert isinstance(model, MyModel)
    assert isinstance(model.array_int, np.ndarray)
    assert isinstance(model.array_float, np.ndarray)
    assert isinstance(model.array_bool, np.ndarray)
    assert isinstance(model.scalar_int, np.int32)
    assert model.array_int.shape == (2, 3)
    assert model.array_float.shape == (1, 3)
    assert model.array_bool.shape == (1, 1, 3)

    assert np.array_equal(model.array_int, arr_int)
    assert np.array_equal(model.array_float, arr_float.reshape(1, 3))
    assert np.array_equal(model.array_bool, arr_bool)
    assert model.scalar_int == scalar_int


def test_json_base64_rountrip():
    model = MyModel(
        array_int=arr_int,
        array_float=arr_float,
        array_bool=arr_bool,
        scalar_int=scalar_int,
    )
    serialized = model.model_dump_json(context={"array_encoding": "base64"})

    assert isinstance(serialized, str)

    try:
        json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise AssertionError("Invalid JSON") from exc

    roundtrip = MyModel.model_validate_json(serialized)

    for field in model.model_fields:
        assert np.array_equal(getattr(roundtrip, field), getattr(model, field))


def test_json_binref_roundtrip(tmpdir):
    dumpdir = Path(tmpdir) / "dumpdir"
    model = MyModel(
        array_int=np.array([[1, 2, 3], [4, 5, 6]]),  # 48 bytes
        array_float=np.array([[1.0, 2.0, 3.0]]),  # 28 bytes
        array_bool=np.array([True, False, True]).reshape(1, 1, 3),  # 3 bytes
        scalar_int=np.int32(42),  # 4 bytes
    )
    serialized = model.model_dump_json(
        context={
            "array_encoding": "binref",
            "base_dir": dumpdir,
            # set file size so that arr_int will be written to one buffer
            # and the remaining arrays to a second buffer
            "max_file_size": 40,
        }
    )

    assert isinstance(serialized, str)
    assert len(list(dumpdir.glob("*.bin"))) == 2

    try:
        json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise AssertionError("Invalid JSON") from exc

    # standard round-trip
    roundtrip = MyModel.model_validate_json(serialized, context={"base_dir": dumpdir})
    for field in model.model_fields:
        assert np.array_equal(getattr(roundtrip, field), getattr(model, field))

    # move all binary files to a different directory and try to load
    # making sure that they contain paths relative to the passed base_dir
    movedir = dumpdir.parent / "movedir"
    dumpdir.rename(movedir)
    roundtrip = MyModel.model_validate_json(serialized, context={"base_dir": movedir})
    for field in model.model_fields:
        assert np.array_equal(getattr(roundtrip, field), getattr(model, field))

    with pytest.raises(ValidationError, match="relative path but no base_dir"):
        model.model_validate_json(serialized)

    with pytest.raises(ValueError, match="base_dir"):
        model.model_dump_json(context={"array_encoding": "binref"})


def test_python_dump_array_encoding():
    model = MyModel(
        array_int=arr_int,
        array_float=arr_float,
        array_bool=arr_bool,
        scalar_int=scalar_int,
    )
    serialized = model.model_dump()
    assert isinstance(serialized["array_int"], np.ndarray)
    assert np.array_equal(serialized["array_int"], arr_int)


def test_python_roundtrip():
    model = MyModel(
        array_int=arr_int,
        array_float=arr_float,
        array_bool=arr_bool,
        scalar_int=scalar_int,
    )
    serialized = model.model_dump()
    roundtrip = MyModel.model_validate(serialized)

    for field in model.model_fields:
        assert np.array_equal(getattr(roundtrip, field), getattr(model, field))


def test_model_from_json(tmpdir):
    payload = {
        "array_int": {
            "shape": [2, 3],
            "dtype": "int64",
            "data": {
                "buffer": base64.b64encode(arr_int.tobytes()).decode(),
                "encoding": "base64",
            },
            # we include a 'object_type' field in all serializations.
            # for backwards compatibility this field is not required, but if
            # given must be Literal["array"].
            "object_type": "array",
        },
        "array_float": {
            "shape": [1, 3],
            "dtype": "float64",
            "data": {
                "buffer": arr_float.flatten().tolist(),
                "encoding": "json",
            },
        },
        "array_bool": {
            "shape": [1, 1, 3],
            "dtype": "bool",
            "data": {
                "buffer": str(
                    tmpdir / "data.bin"
                ),  # test without offset and absolute path
                "encoding": "binref",
            },
        },
        "scalar_int": {
            "shape": [],
            "dtype": "int32",
            "data": {
                "buffer": "data.bin:3",  # test with offset and relative path
                "encoding": "binref",
            },
        },
    }
    with open(tmpdir / "data.bin", "wb") as fi:
        fi.write(arr_bool.tobytes())
        fi.write(scalar_int.tobytes())
    payload_json = json.dumps(payload)

    model = MyModel.model_validate_json(
        payload_json, context={"base_dir": Path(tmpdir)}
    )
    ref_model = MyModel(
        array_int=arr_int,
        array_float=arr_float,
        array_bool=arr_bool,
        scalar_int=scalar_int,
    )

    for field in ref_model.model_fields:
        assert np.array_equal(getattr(model, field), getattr(ref_model, field))


def fix_fake_arrays(fakedata, target_shape, seed=42):
    is_array = (
        lambda x: isinstance(x, dict) and "shape" in x and "dtype" in x and "data" in x
    )
    rng = np.random.RandomState(seed)

    def _walk(data):
        if is_array(data):
            if data["shape"]:  # don't touch scalars
                data["shape"] = target_shape

            new_data = rng.uniform(0, 100, data["shape"]).astype(data["dtype"])

            if data["data"]["encoding"] == "base64":
                data["data"]["buffer"] = base64.b64encode(new_data.tobytes()).decode()
            elif data["data"]["encoding"] == "json":
                data["data"]["buffer"] = new_data.flatten().tolist()
            elif data["data"]["encoding"] == "binref":
                # replace binref with base64 so we don't need to write files
                data["data"]["encoding"] = "base64"
                data["data"]["buffer"] = base64.b64encode(new_data.tobytes()).decode()
        elif isinstance(data, dict):
            for key, value in data.items():
                data[key] = _walk(value)
        elif isinstance(data, list):
            for idx, value in enumerate(data):
                data[idx] = _walk(value)

        return data

    return _walk(fakedata)


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_json_schema(mode):
    """Test that array JSON schemas produce valid payloads."""
    schema = MyModel.model_json_schema(mode=mode)

    faker = jsf.JSF(schema)
    payload = faker.generate()

    # Fix fake arrays to match the expected shape and be properly encoded
    # TODO: remove shape argument to have it inferred from the schema once JSF supports it
    # (see https://github.com/ghandic/jsf/issues/118)
    payload = fix_fake_arrays(payload, target_shape=(2, 3))

    try:
        MyModel.model_validate_json(json.dumps(payload))
    except Exception as exc:
        raise AssertionError("Failed to validate generated payload") from exc


def test_json_schema_flags_and_typenames():
    schema = MyModel.model_json_schema()

    array_int_typeref = schema["properties"]["array_int"]["$ref"]
    assert array_int_typeref == "#/$defs/EncodedArrayModel__2_3__int64__noflags"
    array_int_def = schema["$defs"][array_int_typeref.split("/")[-1]]
    assert array_int_def["array_flags"] == []

    array_float_typeref = schema["properties"]["array_float"]["$ref"]
    assert (
        array_float_typeref
        == "#/$defs/EncodedArrayModel__any_3__float64__DIFFERENTIABLE"
    )
    array_float_def = schema["$defs"][array_float_typeref.split("/")[-1]]
    assert array_float_def["array_flags"] == ["DIFFERENTIABLE"]

    array_bool_typeref = schema["properties"]["array_bool"]["$ref"]
    assert array_bool_typeref == "#/$defs/EncodedArrayModel__anyrank__bool__noflags"
    array_bool_def = schema["$defs"][array_bool_typeref.split("/")[-1]]
    assert array_bool_def["array_flags"] == []

    scalar_int_typeref = schema["properties"]["scalar_int"]["$ref"]
    assert (
        scalar_int_typeref == "#/$defs/EncodedArrayModel__scalar__int32__DIFFERENTIABLE"
    )
    scalar_int_def = schema["$defs"][scalar_int_typeref.split("/")[-1]]
    assert scalar_int_def["array_flags"] == ["DIFFERENTIABLE"]


def test_lazy_sequence_array(tmpdir):
    class MyLazyModel(BaseModel):
        array_int_seq: LazySequence[Array[(2, 3), Int64]]

    # Test with in-memory arrays
    model = MyLazyModel(array_int_seq=[arr_int, arr_int])
    assert len(model.array_int_seq) == 2

    for arr in model.array_int_seq:
        assert np.array_equal(arr, arr_int)
        assert arr.shape == (2, 3)
        assert arr.dtype == np.int64

    # Test with files
    encoded_array = {
        "object_type": "array",
        "shape": [2, 3],
        "dtype": "int64",
        "data": {
            "buffer": base64.b64encode(arr_int.tobytes()).decode(),
            "encoding": "base64",
        },
    }

    with open(tmpdir / "inputs_1.json", "w") as f:
        json.dump(encoded_array, f)

    with open(tmpdir / "inputs_2.json", "w") as f:
        json.dump(encoded_array, f)

    model = MyLazyModel(array_int_seq=f"@{tmpdir / 'inputs_*.json'}")
    assert len(model.array_int_seq) == 2
    for arr in model.array_int_seq:
        assert np.array_equal(arr, arr_int)
        assert arr.shape == (2, 3)
        assert arr.dtype == np.int64

    # Test with a file that violates the wrapped schema
    with open(tmpdir / "inputs_3.json", "w") as f:
        json.dump(
            {
                "object_type": "array",
                "shape": [2, 3],
                "dtype": "int64",
                "data": {"buffer": "not a valid buffer", "encoding": "base64"},
            },
            f,
        )

    model = MyLazyModel(array_int_seq=f"@{tmpdir / 'inputs_*.json'}")
    assert len(model.array_int_seq) == 3
    assert np.array_equal(model.array_int_seq[0], arr_int)

    with pytest.raises(ValidationError):
        model.array_int_seq[2]

    # Test serialization
    model = MyLazyModel(array_int_seq=[arr_int, arr_int])
    dumped = model.model_dump(context={"array_encoding": "base64"}, mode="json")
    assert dumped == {"array_int_seq": [encoded_array, encoded_array]}

    dumped = model.model_dump(mode="python")
    assert np.array_equal(dumped["array_int_seq"][0], arr_int)
    assert np.array_equal(dumped["array_int_seq"][1], arr_int)


def test_lazy_sequence_pyobj(tmpdir):
    class MyLazyModel(BaseModel):
        py_seq: LazySequence[tuple[str, int]]

    test_seq = [("a", 1), ("b", 2)]
    model = MyLazyModel(py_seq=test_seq)
    assert list(model.py_seq) == test_seq

    dumped = model.model_dump()
    assert dumped["py_seq"] == test_seq

    # Test with files
    with open(tmpdir / "inputs_1.json", "w") as f:
        json.dump(test_seq[0], f)

    with open(tmpdir / "inputs_2.json", "w") as f:
        json.dump(test_seq[1], f)

    model = MyLazyModel(py_seq=f"@{tmpdir / 'inputs_*.json'}")

    assert list(model.py_seq) == test_seq

    # Test with a file that violates the wrapped schema
    with open(tmpdir / "inputs_3.json", "w") as f:
        json.dump({"key": "value"}, f)

    model = MyLazyModel(py_seq=f"@{tmpdir / 'inputs_*.json'}")
    assert len(model.py_seq) == 3
    assert model.py_seq[0] == test_seq[0]

    with pytest.raises(ValidationError):
        model.py_seq[2]


def test_lazy_sequence_json_schema():
    class MyLazyModel(BaseModel):
        array_int_seq: LazySequence[Array[(2, 3), Int64]]

    schema = MyLazyModel.model_json_schema()

    allowed_types = schema["properties"]["array_int_seq"]["anyOf"]
    assert len(allowed_types) == 2
    assert allowed_types[0]["type"] == "array"
    assert allowed_types[1]["type"] == "string"


def test_dtype_casting():
    json_payload_str = MyModel(
        array_int=arr_int,
        array_float=arr_float,
        array_bool=arr_bool,
        scalar_int=scalar_int,
    ).model_dump_json()

    # Base case: proper int data (should work fine)
    json_payload = json.loads(json_payload_str)
    json_payload["array_int"]["data"] = {
        "buffer": arr_int.flatten().tolist(),
        "encoding": "json",
    }
    res = MyModel.model_validate(json_payload)
    assert np.array_equal(res.array_int, arr_int)

    # Case 1: floats in JSON array w/o fractional parts (should work fine)
    json_payload = json.loads(json_payload_str)
    json_payload["array_int"]["data"] = {
        "buffer": arr_int.astype(float).flatten().tolist(),
        "encoding": "json",
    }
    res = MyModel.model_validate(json_payload)
    assert np.array_equal(res.array_int, arr_int)

    # Case 2: floats in JSON array w/ fractional parts (should raise)
    json_payload = json.loads(json_payload_str)
    json_payload["array_int"]["data"] = {
        "buffer": (arr_int.astype(float) + 1e-6).flatten().tolist(),
        "encoding": "json",
    }
    with pytest.raises(ValidationError, match="Expected integer data"):
        MyModel.model_validate(json_payload)

    # Case 3: pass NumPy array directly (should work fine)
    json_payload = json.loads(json_payload_str)
    json_payload["array_int"] = arr_int
    res = MyModel.model_validate(json_payload)
    assert np.array_equal(res.array_int, arr_int)

    # Case 4: pass NumPy array with incompatible dtype (should raise)
    json_payload = json.loads(json_payload_str)
    json_payload["array_int"] = arr_int.astype(np.float32)
    with pytest.raises(ValidationError, match="cannot be cast"):
        MyModel.model_validate(json_payload)

    # Case 5: pass JSON data directly (should work fine)
    json_payload = json.loads(json_payload_str)
    json_payload["array_int"] = arr_int.tolist()
    res = MyModel.model_validate(json_payload)
    assert np.array_equal(res.array_int, arr_int)

    # Case 6: pass JSON data with incompatible dtype (should raise)
    json_payload = json.loads(json_payload_str)
    json_payload["array_int"] = arr_int.astype(np.float32).tolist()
    with pytest.raises(ValidationError, match="cannot be cast"):
        MyModel.model_validate(json_payload)

    # Case 7: Pass non-numeric data (should raise)
    json_payload = json.loads(json_payload_str)
    json_payload["array_int"] = ["a", "b", "c"]
    with pytest.raises(ValidationError, match="Could not convert object"):
        MyModel.model_validate(json_payload)

    # Case 8: Pass non-numeric Python object (should raise)
    json_payload = json.loads(json_payload_str)
    json_payload["array_int"] = object()
    with pytest.raises(ValidationError, match="Could not convert object"):
        MyModel.model_validate(json_payload)
