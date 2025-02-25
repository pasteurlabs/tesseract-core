# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from copy import deepcopy
from typing import Annotated, Optional

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from tesseract_core.runtime import Array, Differentiable, Float32, Float64, Int64, UInt8
from tesseract_core.runtime.experimental import LazySequence
from tesseract_core.runtime.schema_generation import (
    create_abstract_eval_schema,
    create_apply_schema,
    create_autodiff_schema,
)


class SubModel(BaseModel):
    foo: Float32
    bar: list[Differentiable[Array[..., Int64]]]


class NestedModel(BaseModel):
    testdiffarr: Differentiable[Array[(5, None), Float64]]
    testfoo: list[SubModel] | None
    testbar: dict[str, SubModel]
    testbaz: Array[(1, 2, 3), UInt8]
    testset: set[int]
    testtuple: tuple[int, str]
    testlazysequence: LazySequence[tuple[str, Differentiable[Array[(None,), Float32]]]]


def make_array(shape, dtype):
    rng = np.random.default_rng()
    return rng.integers(0, 10, shape).astype(dtype)


def replace_arrays(inpobj, replace_fn):
    """Recursively replace NumPy arrays in nested inpobj by calling replace_fn(arr)."""
    if isinstance(inpobj, dict) and "shape" in inpobj and "dtype" in inpobj:
        return replace_fn(inpobj)
    elif isinstance(inpobj, dict):
        return {k: replace_arrays(v, replace_fn) for k, v in inpobj.items()}
    elif isinstance(inpobj, Iterable) and not isinstance(inpobj, str):
        return type(inpobj)([replace_arrays(v, replace_fn) for v in inpobj])
    else:
        return inpobj


testinput = NestedModel(
    testdiffarr=make_array((5, 6), "float64"),
    testfoo=[SubModel(foo=1.0, bar=[1, 2, 3]), SubModel(foo=4.0, bar=[])],
    testbar={"hey there!": SubModel(foo=5.0, bar=[2])},
    testbaz=make_array((1, 2, 3), "uint8"),
    testset={1, 2, 3},
    testtuple=(1, "hello"),
    testlazysequence=[
        ("a", make_array((3,), "float32")),
        ("b", make_array((4,), "float32")),
    ],
).model_dump(
    mode="json",
    # This encoding simplifies how replace_arrays works.
    context={"array_encoding": "base64"},
)

testinput_arrays_only = {
    "testdiffarr": testinput["testdiffarr"],
    "testfoo": [
        {"bar": [testinput["testfoo"][0]["bar"][0]]},
        {"bar": []},
    ],
    "testbar": {"hey there!": {"bar": [testinput["testbar"]["hey there!"]["bar"][0]]}},
    "testlazysequence": [
        (testinput["testlazysequence"][0][1],),
        (testinput["testlazysequence"][1][1],),
    ],
}


def test_create_apply_schema():
    InputSchema, OutputSchema = create_apply_schema(NestedModel, NestedModel)

    InputSchema.model_validate({"inputs": testinput})
    OutputSchema.model_validate(testinput)

    # Extra keys
    with pytest.raises(ValidationError):
        InputSchema.model_validate({"inputs": testinput, "foo": 1})

    # Extra keys (nested dict)
    with pytest.raises(ValidationError):
        inp = testinput.copy()
        inp["foo"] = 1
        InputSchema.model_validate({"inputs": inp})

    # Extra keys (nested model)
    inp = deepcopy(testinput)
    inp["testfoo"][0] = {"foo": 1, "bar": [1, 2, 3]}
    InputSchema.model_validate({"inputs": inp})

    with pytest.raises(ValidationError):
        inp["testfoo"][0] = {"foo": 1, "bar": [1, 2, 3], "extra": 1}
        InputSchema.model_validate({"inputs": inp})

    # Extra keys (output)
    with pytest.raises(ValidationError):
        OutputSchema.model_validate({**testinput, "foo": 1})


def test_create_abstract_eval_schema():
    testinput_abstract = replace_arrays(
        testinput, lambda arr: {"shape": arr["shape"], "dtype": arr["dtype"]}
    )

    AbstractInputSchema, AbstractOutputSchema = create_abstract_eval_schema(
        NestedModel, NestedModel
    )

    UnwrappedInputSchema = AbstractInputSchema.model_fields["inputs"].annotation
    UnwrappedOutputSchema = AbstractOutputSchema.model_fields["root"].annotation

    # should have identical structure
    assert NestedModel.model_fields.keys() == UnwrappedInputSchema.model_fields.keys()
    assert NestedModel.model_fields.keys() == UnwrappedOutputSchema.model_fields.keys()

    # should not raise
    AbstractInputSchema.model_validate({"inputs": testinput_abstract})
    AbstractOutputSchema.model_validate(testinput_abstract)


def test_create_jacobian_schema():
    arr = testinput["testdiffarr"]
    testoutput = {
        "testdiffarr": {
            "testdiffarr": {
                "object_type": "array",
                "shape": (*arr["shape"], *arr["shape"]),
                "dtype": arr["dtype"],
                "data": {
                    "encoding": "json",
                    "buffer": np.zeros((*arr["shape"], *arr["shape"])).tolist(),
                },
            },
        },
    }

    InputSchema, OutputSchema = create_autodiff_schema(
        NestedModel, NestedModel, "jacobian"
    )

    valid_jac_input = {
        "inputs": testinput,
        "jac_inputs": ["testdiffarr"],
        "jac_outputs": ["testdiffarr"],
    }

    InputSchema.model_validate(valid_jac_input)

    for var in ("jac_inputs", "jac_outputs"):
        jac_testinput = valid_jac_input.copy()

        jac_testinput[var] = [
            "testdiffarr",
            "testfoo.[0].bar.[0]",
            "testbar.{hey there!}.bar.[0]",
        ]
        InputSchema.model_validate(jac_testinput)

        with pytest.raises(ValidationError):
            jac_testinput[var] = []
            InputSchema.model_validate(jac_testinput)

        with pytest.raises(ValidationError):
            # non-differentiable array
            jac_testinput[var] = ["testbaz"]
            InputSchema.model_validate(jac_testinput)

        with pytest.raises(ValidationError):
            # non-existing
            jac_testinput[var] = ["moo"]
            InputSchema.model_validate(jac_testinput)

    OutputSchema.model_validate(testoutput)

    with pytest.raises(ValidationError):
        OutputSchema.model_validate(testinput)

    # Extra keys
    with pytest.raises(ValidationError):
        OutputSchema.model_validate({**testoutput, "foo": 1})

    # Extra keys (nested)
    with pytest.raises(ValidationError):
        out = testoutput.copy()
        out["testdiffarr"]["foo"] = 1
        OutputSchema.model_validate(out)


def test_create_jvp_schema():
    testoutput = {
        "testdiffarr": testinput["testdiffarr"],
    }
    InputSchema, OutputSchema = create_autodiff_schema(NestedModel, NestedModel, "jvp")

    valid_jvp_input = {
        "inputs": testinput,
        "jvp_inputs": ["testdiffarr"],
        "jvp_outputs": ["testdiffarr"],
        "tangent_vector": {"testdiffarr": testinput["testdiffarr"]},
    }

    InputSchema.model_validate(valid_jvp_input)

    for var in ("jvp_inputs", "jvp_outputs"):
        jvp_testinput = valid_jvp_input.copy()

        jvp_testinput[var] = [
            "testdiffarr",
            "testfoo.[0].bar.[0]",
            "testlazysequence.[0].[1]",
        ]
        InputSchema.model_validate(jvp_testinput)

        with pytest.raises(ValidationError):
            jvp_testinput[var] = []
            InputSchema.model_validate(jvp_testinput)

        with pytest.raises(ValidationError):
            # non-differentiable array
            jvp_testinput[var] = ["testbaz"]
            InputSchema.model_validate(jvp_testinput)

        with pytest.raises(ValidationError):
            # non-existing
            jvp_testinput[var] = ["moo"]
            InputSchema.model_validate(jvp_testinput)

    OutputSchema.model_validate(testoutput)

    with pytest.raises(ValidationError):
        OutputSchema.model_validate(testinput)


def test_create_vjp_schema():
    testoutput = {
        "testdiffarr": testinput["testdiffarr"],
    }

    InputSchema, OutputSchema = create_autodiff_schema(NestedModel, NestedModel, "vjp")

    valid_vjp_input = {
        "inputs": testinput,
        "vjp_inputs": ["testdiffarr"],
        "vjp_outputs": ["testdiffarr"],
        "cotangent_vector": {"testdiffarr": testinput["testdiffarr"]},
    }

    InputSchema.model_validate(valid_vjp_input)

    for var in ("vjp_inputs", "vjp_outputs"):
        vjp_testinput = valid_vjp_input.copy()

        vjp_testinput[var] = [
            "testdiffarr",
            "testfoo.[0].bar.[0]",
            "testlazysequence.[0].[1]",
        ]
        InputSchema.model_validate(vjp_testinput)

        with pytest.raises(ValidationError):
            vjp_testinput[var] = []
            InputSchema.model_validate(vjp_testinput)

        with pytest.raises(ValidationError):
            # non-differentiable array
            vjp_testinput[var] = ["testbaz"]
            InputSchema.model_validate(vjp_testinput)

        with pytest.raises(ValidationError):
            # non-existing
            vjp_testinput[var] = ["moo"]
            InputSchema.model_validate(vjp_testinput)

    OutputSchema.model_validate(testoutput)

    with pytest.raises(ValidationError):
        OutputSchema.model_validate(testinput)


def test_untyped_container_schema_generation():
    # Test with some stuff that's legal in Pydantic but may break the schema generation
    class WeirdModel(BaseModel):
        generic_dict: dict
        generic_list: list
        generic_tuple: tuple
        generic_set: set

    InputSchema, _ = create_apply_schema(WeirdModel, WeirdModel)

    InputSchema.model_validate(
        {
            "inputs": {
                "generic_dict": {"a": 1, "b": "c"},
                "generic_list": [1, 2, 3],
                "generic_tuple": (1, 2, 3),
                "generic_set": {1, 2, 3},
            },
        }
    )


def test_recursive_model():
    class RecursiveModel(BaseModel):
        foo: int
        bar: Optional["RecursiveModel"] | None

    valid_data = {"foo": 1, "bar": {"foo": 2, "bar": {"foo": 3, "bar": None}}}

    # valid pydantic model
    RecursiveModel.model_validate(valid_data)

    # not valid as Tesseract schema
    with pytest.raises(ValueError, match="Recursive model definition"):
        create_apply_schema(RecursiveModel, RecursiveModel)


@pytest.mark.parametrize("endpoint", ["apply", "abstract_eval", "autodiff"])
def test_fancy_pydantic_model(endpoint):
    # Define a model with some fancy features
    from pydantic import AfterValidator, Field, computed_field, model_validator

    def _array_validator(arr):
        # Sharp edge: in abstract eval, this field won't be a NumPy array
        # so we just pass it through
        if not isinstance(arr, np.ndarray):
            return arr
        assert arr.sum() < 2
        return arr

    class InputSchema(BaseModel):
        someint: int = Field(gt=0)

        myarray: Annotated[
            Differentiable[Array[(None,), Float64]], AfterValidator(_array_validator)
        ]

        # All those also include fail-safes for `abstract_eval`
        @computed_field
        @property
        def array_sum(self) -> float:
            if not isinstance(self.myarray, np.ndarray):
                return 0.0
            return float(self.myarray.sum())

        @property
        def array_sum2(self) -> float:
            if not isinstance(self.myarray, np.ndarray):
                return 0.0
            return float(self.myarray.sum())

        @model_validator(mode="after")
        def validate_model(self):
            if not isinstance(self.myarray, np.ndarray):
                return self
            assert self.myarray.sum() + self.someint < 10
            return self

    valid_data = {
        "someint": 1,
        "myarray": {
            "object_type": "array",
            "shape": [5],
            "dtype": "float64",
            "data": {"encoding": "json", "buffer": [0.1, 0.2, 0.3, 0.4, 0.5]},
        },
    }

    # Validate that InputSchema is a valid pydantic model
    InputSchema.model_validate(valid_data)

    # Test that the schema generation works as expected
    if endpoint == "apply":
        ApplyInputSchema, ApplyOutputSchema = create_apply_schema(
            InputSchema, InputSchema
        )

        parsed_inputs = ApplyInputSchema.model_validate({"inputs": valid_data}).inputs

        roundtrip = parsed_inputs.model_dump(mode="json")
        array_sum = roundtrip.pop("array_sum")
        assert array_sum == parsed_inputs.myarray.sum()
        assert roundtrip == valid_data

        # also check output schema
        parsed_outputs = ApplyOutputSchema.model_validate(valid_data)
        assert "array_sum" in parsed_outputs.model_dump()

        # trigger field validator
        with pytest.raises(ValidationError, match="greater_than"):
            ApplyInputSchema.model_validate(
                {
                    "inputs": {
                        "someint": -1,
                        "myarray": valid_data["myarray"],
                    }
                }
            )

        with pytest.raises(ValidationError, match="sum"):
            ApplyInputSchema.model_validate(
                {
                    "inputs": {
                        "someint": 1,
                        "myarray": {
                            "object_type": "array",
                            "shape": [5],
                            "dtype": "float64",
                            "data": {"encoding": "json", "buffer": [1, 1, 1, 1, 1]},
                        },
                    }
                }
            )

        # trigger model validator
        with pytest.raises(ValidationError, match="sum"):
            ApplyInputSchema.model_validate(
                {
                    "inputs": {
                        "someint": 10,
                        "myarray": valid_data["myarray"],
                    }
                }
            )

    elif endpoint == "abstract_eval":
        # Keep everything but replace with dummy values

        # Invalid for abstract eval
        valid_data["myarray"].pop("data")
        valid_data["myarray"].pop("object_type")

        AbstractEvalSchema, _ = create_abstract_eval_schema(InputSchema, InputSchema)
        parsed_inputs = AbstractEvalSchema.model_validate({"inputs": valid_data}).inputs

        roundtrip = parsed_inputs.model_dump(mode="json")
        array_sum = roundtrip.pop("array_sum")
        assert array_sum == 0.0
        assert roundtrip == valid_data

        # trigger field validator on non-array arg
        with pytest.raises(ValidationError, match="greater_than"):
            AbstractEvalSchema.model_validate(
                {
                    "inputs": {
                        "someint": -1,
                        "myarray": valid_data["myarray"],
                    }
                }
            )

    elif endpoint == "autodiff":
        AutodiffSchema, _ = create_autodiff_schema(InputSchema, InputSchema, "jacobian")
        parsed_inputs = AutodiffSchema.model_validate(
            {
                "inputs": valid_data,
                "jac_inputs": ["myarray"],
                "jac_outputs": ["myarray"],
            }
        ).inputs

        roundtrip = parsed_inputs.model_dump(mode="json")
        array_sum = roundtrip.pop("array_sum")
        assert array_sum == parsed_inputs.myarray.sum()
        assert roundtrip == valid_data

        # trigger field validator
        with pytest.raises(ValidationError, match="greater_than"):
            AutodiffSchema.model_validate(
                {
                    "inputs": {
                        "someint": -1,
                        "myarray": valid_data["myarray"],
                    },
                    "jac_inputs": ["myarray"],
                    "jac_outputs": ["myarray"],
                }
            )

        with pytest.raises(ValidationError, match="sum"):
            AutodiffSchema.model_validate(
                {
                    "inputs": {
                        "someint": 1,
                        "myarray": {
                            "object_type": "array",
                            "shape": [5],
                            "dtype": "float64",
                            "data": {"encoding": "json", "buffer": [1, 1, 1, 1, 1]},
                        },
                    },
                    "jac_inputs": ["myarray"],
                    "jac_outputs": ["myarray"],
                }
            )

        # trigger model validator
        with pytest.raises(ValidationError, match="sum"):
            AutodiffSchema.model_validate(
                {
                    "inputs": {
                        "someint": 10,
                        "myarray": valid_data["myarray"],
                    },
                    "jac_inputs": ["myarray"],
                    "jac_outputs": ["myarray"],
                }
            )
