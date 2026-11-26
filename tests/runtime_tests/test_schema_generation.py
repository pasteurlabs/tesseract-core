# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import re
from collections.abc import Iterable
from copy import deepcopy
from typing import Annotated

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError

from tesseract_core.runtime import Array, Differentiable, Float32, Float64, Int64, UInt8
from tesseract_core.runtime.experimental import LazySequence
from tesseract_core.runtime.schema_generation import (
    DICT_INDEX_SENTINEL,
    SEQ_INDEX_SENTINEL,
    _path_to_pattern,
    apply_function_to_model_tree,
    create_abstract_eval_schema,
    create_apply_schema,
    create_gradient_schema,
)
from tesseract_core.runtime.schema_types import ShapeDType
from tesseract_core.runtime.tree_transforms import escape_dict_key, path_to_index_op


class SubModel(BaseModel):
    foo: Float32
    bar: list[Differentiable[Array[..., Int64]]]


class SubRootModel(RootModel):
    root: Float32


class NestedModel(BaseModel):
    testdiffarr: Differentiable[Array[(5, None), Float64]]
    testfoo: list[SubModel] | None
    testbar: dict[str, SubModel]
    testbaz: Array[(1, 2, 3), UInt8]
    testset: set[int]
    testtuple: tuple[int, str]
    testlazysequence: LazySequence[tuple[str, Differentiable[Array[(None,), Float32]]]]
    testrootmodel: SubRootModel


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
    testbar={"hey there": SubModel(foo=5.0, bar=[2])},
    testbaz=make_array((1, 2, 3), "uint8"),
    testset={1, 2, 3},
    testtuple=(1, "hello"),
    testlazysequence=[
        ("a", make_array((3,), "float32")),
        ("b", make_array((4,), "float32")),
    ],
    testrootmodel=2.3,
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
    "testbar": {"hey there": {"bar": [testinput["testbar"]["hey there"]["bar"][0]]}},
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


def test_field_order_exported():
    InputSchema, OutputSchema = create_apply_schema(NestedModel, NestedModel)

    expected = {
        "": list(NestedModel.model_fields.keys()),
        "testfoo.[]": list(SubModel.model_fields.keys()),
        "testbar.{}": list(SubModel.model_fields.keys()),
        "testrootmodel": ["root"],
    }
    assert InputSchema.model_json_schema()["field_order"] == expected
    assert OutputSchema.model_json_schema()["field_order"] == expected


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


def test_abstract_eval_schema_array_defaults():
    class Sub(BaseModel):
        s: Differentiable[Float32] = 1.0

    class Model(BaseModel):
        x: Differentiable[Array[(None,), Float32]]
        y: Array[(2,), Float64] = [1, 2]
        z: Array[(2,), Float64] = None
        sub: Sub

    AbstractInputSchema, _ = create_abstract_eval_schema(Model, Model)
    inputs = AbstractInputSchema.model_validate(
        {"inputs": {"x": {"shape": [3], "dtype": "float32"}, "sub": {}}}
    ).inputs

    assert inputs.model_dump() == {
        "x": {"shape": (3,), "dtype": "float32"},
        "y": {"shape": (2,), "dtype": "float64"},
        "z": None,
        "sub": {"s": {"shape": (), "dtype": "float32"}},
    }


def test_abstract_eval_schema_array_default_factory():
    class Model(BaseModel):
        s: Array[..., Float32] = Field(default_factory=lambda: np.float32(1.0))
        y: Array[(None,), Float64] = Field(default_factory=lambda: np.zeros(3))
        n: list[int] = Field(default_factory=list)

    AbstractInputSchema, _ = create_abstract_eval_schema(Model, Model)
    inputs = AbstractInputSchema.model_validate({"inputs": {}}).inputs

    assert isinstance(inputs.s, ShapeDType)
    assert inputs.model_dump() == {
        "s": {"shape": (), "dtype": "float32"},
        "y": {"shape": (3,), "dtype": "float64"},
        "n": [],
    }

    # Factories still run per instance, so mutable defaults are not shared
    other = AbstractInputSchema.model_validate({"inputs": {}}).inputs
    assert other.n is not inputs.n

    # Values the caller sets are left alone
    inputs = AbstractInputSchema.model_validate(
        {"inputs": {"y": {"shape": [5], "dtype": "float64"}}}
    ).inputs
    assert inputs.y == ShapeDType(shape=(5,), dtype="float64")


@pytest.mark.parametrize("use_factory", [False, True])
def test_abstract_eval_schema_nested_array_defaults(use_factory):
    class Sub(BaseModel):
        v: Array[(None,), Float32]

    class Aliased(BaseModel):
        v: Array[(None,), Float32] = Field(alias="vee")

    class Root(RootModel):
        root: list[Array[(None,), Float64]]

    def default(value):
        if use_factory:
            return Field(default_factory=lambda: deepcopy(value))
        return value

    class Model(BaseModel):
        seq: list[Array[(None,), Float64]] = default([[1, 2]])
        opt: Array[(None,), Float64] | None = default([1, 2, 3])
        mapping: dict[str, Array[(None,), Float64]] = default({"a": [1]})
        sub: Sub = default(Sub(v=np.zeros(4, dtype=np.float32)))
        aliased: Aliased = default(Aliased(vee=np.zeros(2, dtype=np.float32)))
        root: Root = default(Root([np.zeros(5)]))
        lazy: LazySequence[Array[(None,), Float64]] = default([[1.0, 2.0]])
        none: Array[(None,), Float64] | None = default(None)
        # Unhashable metadata must not break building the converter
        meta: Annotated[Array[(2,), Float64], {"unhashable": "metadata"}] = default(
            [1, 2]
        )

    AbstractInputSchema, _ = create_abstract_eval_schema(Model, Model)
    inputs = AbstractInputSchema.model_validate({"inputs": {}}).inputs

    # Dumping an abstract LazySequence fails regardless of defaults, so check it apart
    assert list(inputs.lazy) == [ShapeDType(shape=(2,), dtype="float64")]
    assert inputs.model_dump(exclude={"lazy"}) == {
        # Cast to the declared dtype, not the dtype NumPy would infer
        "seq": [{"shape": (2,), "dtype": "float64"}],
        "opt": {"shape": (3,), "dtype": "float64"},
        "mapping": {"a": {"shape": (1,), "dtype": "float64"}},
        "sub": {"v": {"shape": (4,), "dtype": "float32"}},
        "aliased": {"v": {"shape": (2,), "dtype": "float32"}},
        "root": [{"shape": (5,), "dtype": "float64"}],
        "none": None,
        "meta": {"shape": (2,), "dtype": "float64"},
    }
    # A model default becomes an instance of the abstract model
    assert type(inputs.sub).__name__ == "AbstractEval_Sub"


def test_abstract_eval_schema_rejects_real_arrays():
    """Defaults are converted, but callers still have to send shape and dtype."""

    class Model(BaseModel):
        x: Array[(None,), Float64] = Field(default_factory=lambda: np.zeros(3))
        seq: list[Array[(None,), Float64]] = [[1.0]]

    AbstractInputSchema, _ = create_abstract_eval_schema(Model, Model)
    for inputs in ({"x": np.zeros(3)}, {"x": [1.0, 2.0]}, {"seq": [np.zeros(2)]}):
        with pytest.raises(ValidationError):
            AbstractInputSchema.model_validate({"inputs": inputs})


def test_abstract_eval_schema_advertises_abstract_defaults():
    class Model(BaseModel):
        y: Array[(2,), Float64] = [1, 2]
        seq: list[Array[(None,), Float64]] = [[1, 2]]

    AbstractInputSchema, _ = create_abstract_eval_schema(Model, Model)
    props = AbstractInputSchema.model_json_schema()["$defs"]["AbstractEval_Model"][
        "properties"
    ]
    assert props["y"]["default"] == {"shape": [2], "dtype": "float64"}
    assert props["seq"]["default"] == [{"shape": [2], "dtype": "float64"}]


def test_apply_function_to_model_tree_default_factory_takes_data():
    class Model(BaseModel):
        a: int = 1
        b: int = Field(default_factory=lambda data: data["a"] + 1)

    NewModel = apply_function_to_model_tree(
        Model,
        lambda obj, _: obj,
        make_default_converter=lambda _, __: lambda default: default * 10,
    )
    # a's default becomes 10; b's factory sees a=10 and its output is scaled too
    assert NewModel().model_dump() == {"a": 10, "b": 110}
    assert NewModel(a=2).model_dump() == {"a": 2, "b": 30}
    # An explicitly set value is never converted
    assert NewModel(b=5).model_dump() == {"a": 10, "b": 5}


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

    InputSchema, OutputSchema = create_gradient_schema(
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
            "testbar.{hey there}.bar.[0]",
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

        for almost in ("testfoo.[0].bar.[0].moo", "moo testfoo.[0].bar.[0]"):
            with pytest.raises(ValidationError):
                # a valid path with something around it
                jac_testinput[var] = [almost]
                InputSchema.model_validate(jac_testinput)

    jac_testinput = valid_jac_input.copy()
    # Test LookupError (from IndexError) (only raised on jac_inputs)
    with pytest.raises(ValidationError):
        jac_testinput["jac_inputs"] = ["testfoo.[2].bar.[0]"]
        InputSchema.model_validate(jac_testinput)

    with pytest.raises(ValidationError):
        jac_testinput["jac_inputs"] = ["testlazysequence.[0].[2]"]
        InputSchema.model_validate(jac_testinput)

    # Test LookupError (from KeyError) (only raised on jac_inputs)
    with pytest.raises(ValidationError):
        jac_testinput["jac_inputs"] = ["testbar.{hey ther}.bar.[0]"]
        InputSchema.model_validate(jac_testinput)

    ctx = {
        "input_keys": valid_jac_input["jac_inputs"],
        "output_keys": valid_jac_input["jac_outputs"],
    }

    OutputSchema.model_validate(testoutput, context=ctx)

    # Input instead of output
    with pytest.raises(ValidationError):
        OutputSchema.model_validate(testinput, context=ctx)

    # Extra keys
    with pytest.raises(ValidationError):
        OutputSchema.model_validate({**testoutput, "foo": 1}, context=ctx)

    # Extra keys (nested)
    with pytest.raises(ValidationError):
        out = testoutput.copy()
        out["testdiffarr"]["foo"] = 1
        OutputSchema.model_validate(out, context=ctx)

    # Missing keys
    with pytest.raises(ValidationError):
        OutputSchema.model_validate({**testoutput, "testdiffarr": {}}, context=ctx)

    # Flat dict
    with pytest.raises(ValidationError):
        OutputSchema.model_validate(testoutput["testdiffarr"], context=ctx)


@pytest.mark.parametrize(
    "input_shape,output_shape,jac_shape,valid",
    [
        # Both shapes concrete
        ((3,), (5,), (5, 3), True),
        ((3,), (5,), (3, 5), False),
        # Output shape is Ellipsis
        ((3,), ..., (5, 3), True),
        ((3,), ..., (3, 3), True),
        ((3,), ..., (5, 2, 3), True),
        ((3,), ..., (3,), True),
        ((), ..., (5,), True),
        ((3,), ..., (3, 5), False),
        # Input shape is Ellipsis
        (..., (5,), (5, 3), True),
        (..., (5,), (5, 3, 2), True),
        (..., (5,), (5,), True),
        (..., (), (3,), True),
        (..., (5,), (3, 5), False),
    ],
)
def test_jacobian_output_shape_validation(input_shape, output_shape, jac_shape, valid):
    class InputSchema(BaseModel):
        x: Differentiable[Array[input_shape, Float64]]

    class OutputSchema(BaseModel):
        y: Differentiable[Array[output_shape, Float64]]

    _, JacobianOutputSchema = create_gradient_schema(
        InputSchema, OutputSchema, "jacobian"
    )
    jac = {"y": {"x": np.zeros(jac_shape)}}
    ctx = {"input_keys": {"x"}, "output_keys": {"y"}}

    if valid:
        JacobianOutputSchema.model_validate(jac, context=ctx)
    else:
        with pytest.raises(ValidationError, match="Jacobian result"):
            JacobianOutputSchema.model_validate(jac, context=ctx)


def test_jacobian_output_validation_accepts_torch_gpu_tensor():
    """GPU tensors are validated unconverted, and torch dtypes are not NumPy dtype names."""

    class TorchDtype:
        def __str__(self):
            return "torch.float32"

    class TorchCudaTensor:
        shape = (5, 3)
        dtype = TorchDtype()

        @property
        def __cuda_array_interface__(self):
            return {
                "shape": (5, 3),
                "typestr": "<f4",
                "data": (0, False),
                "strides": None,
                "version": 3,
            }

    class InputSchema(BaseModel):
        x: Differentiable[Array[(3,), Float32]]

    class OutputSchema(BaseModel):
        y: Differentiable[Array[(5,), Float32]]

    _, JacobianOutputSchema = create_gradient_schema(
        InputSchema, OutputSchema, "jacobian"
    )
    tensor = TorchCudaTensor()
    result = JacobianOutputSchema.model_validate(
        {"y": {"x": tensor}}, context={"input_keys": {"x"}, "output_keys": {"y"}}
    )
    assert result.root["y"]["x"] is tensor


def test_create_jvp_schema():
    testoutput = {
        "testdiffarr": testinput["testdiffarr"],
    }
    InputSchema, OutputSchema = create_gradient_schema(NestedModel, NestedModel, "jvp")

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
            "testbar.{hey there}.bar.[0]",
            "testlazysequence.[0].[1]",
        ]
        if var == "jvp_inputs":
            jvp_testinput["tangent_vector"] = {
                **jvp_testinput["tangent_vector"],
                "testfoo.[0].bar.[0]": testinput["testfoo"][0]["bar"][0],
                "testbar.{hey there}.bar.[0]": testinput["testbar"]["hey there"]["bar"][
                    0
                ],
                "testlazysequence.[0].[1]": testinput["testlazysequence"][0][1],
            }
        InputSchema.model_validate(jvp_testinput)

        with pytest.raises(ValidationError):
            jvp_testinput[var] = []
            if var == "jvp_inputs":
                jvp_testinput["tangent_vector"] = {}
            InputSchema.model_validate(jvp_testinput)

        with pytest.raises(ValidationError):
            # non-differentiable array
            jvp_testinput[var] = ["testbaz"]
            if var == "jvp_inputs":
                jvp_testinput["tangent_vector"] = {"testbaz": testinput["testbaz"]}
            InputSchema.model_validate(jvp_testinput)

        with pytest.raises(ValidationError):
            # non-existing
            jvp_testinput[var] = ["moo"]
            if var == "jvp_inputs":
                jvp_testinput["tangent_vector"] = {"moo": testinput["testdiffarr"]}
            InputSchema.model_validate(jvp_testinput)

    jvp_testinput = valid_jvp_input.copy()
    # Test LookupError (from IndexError) (only raised on jvp_inputs)
    with pytest.raises(ValidationError):
        jvp_testinput["jvp_inputs"] = ["testfoo.[2].bar.[0]"]
        jvp_testinput["tangent_vector"] = {
            "testfoo.[2].bar.[0]": testinput["testdiffarr"]
        }
        InputSchema.model_validate(jvp_testinput)

    with pytest.raises(ValidationError):
        jvp_testinput["jvp_inputs"] = ["testlazysequence.[0].[2]"]
        jvp_testinput["tangent_vector"] = {
            "testlazysequence.[0].[2]": testinput["testdiffarr"]
        }
        InputSchema.model_validate(jvp_testinput)

    # Test LookupError (from KeyError) (only raised on jvp_inputs)
    with pytest.raises(ValidationError):
        jvp_testinput["jvp_inputs"] = ["testbar.{hey ther}.bar.[0]"]
        jvp_testinput["tangent_vector"] = {
            "testbar.{hey ther}.bar.[0]": testinput["testbar"]["hey there"]["bar"][0]
        }
        InputSchema.model_validate(jvp_testinput)

    # Mis-aligned tangent vector
    jvp_testinput = valid_jvp_input.copy()
    with pytest.raises(ValidationError):
        jvp_testinput["tangent_vector"] = {**jvp_testinput["tangent_vector"], "foo": 1}
        InputSchema.model_validate(jvp_testinput)

    OutputSchema.model_validate(
        testoutput, context={"output_keys": valid_jvp_input["jvp_outputs"]}
    )

    with pytest.raises(ValidationError):
        OutputSchema.model_validate(
            testinput, context={"output_keys": valid_jvp_input["jvp_outputs"]}
        )

    # Extra keys
    with pytest.raises(ValidationError):
        OutputSchema.model_validate(
            {**testoutput, "foo": 1},
            context={"output_keys": valid_jvp_input["jvp_outputs"]},
        )

    # Missing keys
    with pytest.raises(ValidationError):
        OutputSchema.model_validate(
            {}, context={"output_keys": valid_jvp_input["jvp_outputs"]}
        )


def test_create_vjp_schema():
    testoutput = {
        "testdiffarr": testinput["testdiffarr"],
    }

    InputSchema, OutputSchema = create_gradient_schema(NestedModel, NestedModel, "vjp")

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
        if var == "vjp_outputs":
            vjp_testinput["cotangent_vector"] = {
                **vjp_testinput["cotangent_vector"],
                "testfoo.[0].bar.[0]": 1,
                "testlazysequence.[0].[1]": testinput["testlazysequence"][0][1],
            }
        InputSchema.model_validate(vjp_testinput)

        with pytest.raises(ValidationError):
            vjp_testinput[var] = []
            if var == "vjp_outputs":
                vjp_testinput["cotangent_vector"] = {}
            InputSchema.model_validate(vjp_testinput)

        with pytest.raises(ValidationError):
            # non-differentiable array
            vjp_testinput[var] = ["testbaz"]
            if var == "vjp_outputs":
                vjp_testinput["cotangent_vector"] = {"testbaz": testinput["testbaz"]}
            InputSchema.model_validate(vjp_testinput)

        with pytest.raises(ValidationError):
            # non-existing
            vjp_testinput[var] = ["moo"]
            if var == "vjp_outputs":
                vjp_testinput["cotangent_vector"] = {"moo": 1}
            InputSchema.model_validate(vjp_testinput)

    vjp_testinput = valid_vjp_input.copy()
    # Test LookupError (from IndexError) (only raised on vjp_inputs)
    with pytest.raises(ValidationError):
        vjp_testinput["vjp_inputs"] = ["testfoo.[2].bar.[0]"]
        InputSchema.model_validate(vjp_testinput)

    with pytest.raises(ValidationError):
        vjp_testinput["vjp_inputs"] = ["testlazysequence.[0].[2]"]
        InputSchema.model_validate(vjp_testinput)

    # Test LookupError (from KeyError) (only raised on vjp_inputs)
    with pytest.raises(ValidationError):
        vjp_testinput["vjp_inputs"] = ["testbar.{hey ther}.bar.[0]"]
        InputSchema.model_validate(vjp_testinput)

    # Mis-aligned cotangent vector
    vjp_testinput = valid_vjp_input.copy()
    with pytest.raises(ValidationError):
        vjp_testinput["cotangent_vector"] = {
            **vjp_testinput["cotangent_vector"],
            "foo": 1,
        }
        InputSchema.model_validate(vjp_testinput)

    OutputSchema.model_validate(
        testoutput, context={"input_keys": valid_vjp_input["vjp_inputs"]}
    )

    with pytest.raises(ValidationError):
        OutputSchema.model_validate(
            testinput, context={"input_keys": valid_vjp_input["vjp_inputs"]}
        )

    # Extra keys
    with pytest.raises(ValidationError):
        OutputSchema.model_validate(
            {**testoutput, "foo": 1},
            context={"input_keys": valid_vjp_input["vjp_inputs"]},
        )

    # Missing keys
    with pytest.raises(ValidationError):
        OutputSchema.model_validate(
            {}, context={"input_keys": valid_vjp_input["vjp_inputs"]}
        )


def test_jvp_tangent_vector_shape_validation():
    """Test that tangent vector shapes are validated against input shapes."""
    InputSchema, _ = create_gradient_schema(NestedModel, NestedModel, "jvp")

    valid_jvp_input = {
        "inputs": testinput,
        "jvp_inputs": ["testdiffarr"],
        "jvp_outputs": ["testdiffarr"],
        "tangent_vector": {"testdiffarr": testinput["testdiffarr"]},
    }

    # Valid input should pass
    InputSchema.model_validate(valid_jvp_input)

    # Wrong shape should fail - use a proper array with wrong shape
    wrong_shape_arr = make_array((3, 4), "float64")
    wrong_shape_input = valid_jvp_input.copy()
    wrong_shape_input["tangent_vector"] = {
        "testdiffarr": wrong_shape_arr,
    }
    with pytest.raises(ValidationError, match="shape"):
        InputSchema.model_validate(wrong_shape_input)


def test_jvp_reports_invalid_inputs():
    """Invalid inputs are reported as such, not as a crash in the tangent validator."""
    InputSchema, _ = create_gradient_schema(NestedModel, NestedModel, "jvp")

    jvp_input = {
        "inputs": {**testinput, "testbaz": make_array((1, 2), "uint8")},
        "jvp_inputs": ["testdiffarr"],
        "jvp_outputs": ["testdiffarr"],
        "tangent_vector": {"testdiffarr": testinput["testdiffarr"]},
    }

    with pytest.raises(ValidationError, match="testbaz"):
        InputSchema.model_validate(jvp_input)


def test_jvp_reports_a_tangent_along_an_input_that_was_not_sent():
    """So is a tangent along an optional input that is absent."""

    class Offset(BaseModel):
        a: Differentiable[Array[(3,), Float64]]

    class Inputs(BaseModel):
        x: Differentiable[Array[(3,), Float64]]
        offset: Offset | None = None

    InputSchema, _ = create_gradient_schema(Inputs, Inputs, "jvp")

    arr = make_array((3,), "float64")
    jvp_input = {
        "inputs": {"x": arr},
        "jvp_inputs": ["offset.a"],
        "jvp_outputs": ["x"],
        "tangent_vector": {"offset.a": arr},
    }

    with pytest.raises(ValidationError, match="Could not find tangent_vector path"):
        InputSchema.model_validate(jvp_input)


def test_jvp_tangent_vector_dtype_casting():
    """Test that integer tangent vectors are cast to float for AD compatibility."""
    InputSchema, _ = create_gradient_schema(NestedModel, NestedModel, "jvp")

    # Create input with integer dtype tangent vector (same shape as testdiffarr: 5x6)
    int_tangent = make_array((5, 6), "int32")

    jvp_input = {
        "inputs": testinput,
        "jvp_inputs": ["testdiffarr"],
        "jvp_outputs": ["testdiffarr"],
        "tangent_vector": {"testdiffarr": int_tangent},
    }

    # Should not raise - integers should be cast to float
    result = InputSchema.model_validate(jvp_input)
    # The tangent vector should now have float dtype
    tangent = result.tangent_vector["testdiffarr"]
    assert np.issubdtype(tangent.dtype, np.floating)


def test_vjp_cotangent_vector_shape_validation():
    """Test that cotangent vector shapes are validated against schema patterns."""
    InputSchema, _ = create_gradient_schema(NestedModel, NestedModel, "vjp")

    valid_vjp_input = {
        "inputs": testinput,
        "vjp_inputs": ["testdiffarr"],
        "vjp_outputs": ["testdiffarr"],
        "cotangent_vector": {"testdiffarr": testinput["testdiffarr"]},
    }

    # Valid input should pass
    InputSchema.model_validate(valid_vjp_input)

    # Wrong number of dimensions should fail - use array with 1 dimension instead of 2
    wrong_dims_arr = make_array((5,), "float64")
    wrong_dims_input = valid_vjp_input.copy()
    wrong_dims_input["cotangent_vector"] = {
        "testdiffarr": wrong_dims_arr,
    }
    with pytest.raises(ValidationError, match="dimensions"):
        InputSchema.model_validate(wrong_dims_input)


def test_vjp_cotangent_vector_dtype_casting():
    """Test that integer cotangent vectors are cast to float for AD compatibility."""
    InputSchema, _ = create_gradient_schema(NestedModel, NestedModel, "vjp")

    # Create input with integer dtype cotangent vector (same shape as testdiffarr: 5x6)
    int_cotangent = make_array((5, 6), "int32")

    vjp_input = {
        "inputs": testinput,
        "vjp_inputs": ["testdiffarr"],
        "vjp_outputs": ["testdiffarr"],
        "cotangent_vector": {"testdiffarr": int_cotangent},
    }

    # Should not raise - integers should be cast to float
    result = InputSchema.model_validate(vjp_input)
    # The cotangent vector should now have float dtype
    cotangent = result.cotangent_vector["testdiffarr"]
    assert np.issubdtype(cotangent.dtype, np.floating)


def test_gradient_paths_match_whole_field_name():
    """A field whose name starts with a sibling's must get its own annotation.

    Path patterns for fields inside a list or dict are regexes, and an
    unanchored match let the pattern for `a` claim `layers.[0].alpha`. The
    first declared field won, so `alpha` was validated against `a`'s shape and
    dtype: a correct cotangent was rejected, or silently downcast.
    """

    class Layer(BaseModel):
        a: Differentiable[Array[(3,), Float32]]
        alpha: Differentiable[Array[(4,), Float64]]

    class Layers(BaseModel):
        layers: list[Layer]

    InputSchema, _ = create_gradient_schema(Layers, Layers, "vjp")
    inputs = {
        "layers": [
            {"a": make_array((3,), "float32"), "alpha": make_array((4,), "float64")}
        ]
    }
    cotangent = np.array([1.0, 1 / 3, np.pi, 2 / 3], dtype="float64")

    validated = InputSchema.model_validate(
        {
            "inputs": inputs,
            "vjp_inputs": ["layers.[0].alpha"],
            "vjp_outputs": ["layers.[0].alpha"],
            "cotangent_vector": {"layers.[0].alpha": cotangent},
        }
    )

    got = validated.cotangent_vector["layers.[0].alpha"]
    assert got.dtype == np.dtype("float64")
    np.testing.assert_array_equal(got, cotangent)


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
        bar: "RecursiveModel | None"

    valid_data = {"foo": 1, "bar": {"foo": 2, "bar": {"foo": 3, "bar": None}}}

    # valid pydantic model
    RecursiveModel.model_validate(valid_data)

    # not valid as Tesseract schema
    with pytest.raises(ValueError, match="Recursive model definition"):
        create_apply_schema(RecursiveModel, RecursiveModel)


@pytest.mark.parametrize("endpoint", ["apply", "abstract_eval", "gradient"])
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

    elif endpoint == "gradient":
        GradientSchema, _ = create_gradient_schema(InputSchema, InputSchema, "jacobian")
        parsed_inputs = GradientSchema.model_validate(
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
            GradientSchema.model_validate(
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
            GradientSchema.model_validate(
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
            GradientSchema.model_validate(
                {
                    "inputs": {
                        "someint": 10,
                        "myarray": valid_data["myarray"],
                    },
                    "jac_inputs": ["myarray"],
                    "jac_outputs": ["myarray"],
                }
            )


@pytest.mark.parametrize(
    "endpoint", ["apply", "abstract_eval", "jacobian", "jvp", "vjp"]
)
def test_json_schema(endpoint):
    if endpoint == "apply":
        InputSchema, OutputSchema = create_apply_schema(NestedModel, NestedModel)
    elif endpoint == "abstract_eval":
        InputSchema, OutputSchema = create_abstract_eval_schema(
            NestedModel, NestedModel
        )
    else:
        InputSchema, OutputSchema = create_gradient_schema(
            NestedModel, NestedModel, endpoint
        )

    schema_inputs = InputSchema.model_json_schema()
    schema_outputs = OutputSchema.model_json_schema()

    # Test that the JSON schema is valid JSON
    json.dumps(schema_inputs)
    json.dumps(schema_outputs)


def test_model_config_extra_forbid():
    class Child(BaseModel):
        x: str
        model_config: ConfigDict = ConfigDict(extra="allow")

    class Parent(BaseModel):
        child: Child

    ApplyParent = apply_function_to_model_tree(
        Parent,
        lambda x, y: x,
        default_model_config=dict(extra="forbid"),
    )
    ApplyChild = ApplyParent.model_fields["child"].annotation
    assert ApplyChild.model_config["extra"] == "allow"
    assert ApplyParent.model_config["extra"] == "forbid"

    ApplyParent.model_validate({"child": {"x": "foo"}})
    ApplyParent.model_validate({"child": {"x": "foo", "extra": 1}})

    with pytest.raises(ValidationError):
        ApplyParent.model_validate({"child": {"x": "foo"}, "extra": 1})


@pytest.mark.parametrize(
    "index,advertised",
    [("[0]", True), ("[7]", True), ("[-1]", False), ("[-7]", False)],
)
def test_advertised_sequence_indices_are_resolvable(index, advertised):
    """What the sequence pattern advertises has to survive path_to_index_op.

    The pattern reaches users through the OpenAPI schema and through the
    validation error that quotes it.
    """
    pattern = _path_to_pattern((SEQ_INDEX_SENTINEL,))
    assert bool(re.fullmatch(pattern, index)) is advertised
    if advertised:
        path_to_index_op(index)


@pytest.mark.parametrize(
    "key",
    [
        "plain",
        "with space",
        "with-hyphen",
        "a/b",
        "a:b",
        "layer.0.weight",
        "bar{foobar}",
        "café",
    ],
)
def test_advertised_dict_keys_are_resolvable(key):
    """Every key the schema advertises has to survive path_to_index_op.

    The pattern reaches users through the OpenAPI schema, so a key it refuses
    is a key whose gradients cannot be requested at all.
    """
    pattern = _path_to_pattern((DICT_INDEX_SENTINEL,))
    segment = "{" + escape_dict_key(key) + "}"
    assert re.fullmatch(pattern, segment), f"pattern refuses {segment}"
    assert path_to_index_op(segment) == ("dict", key)


_CONTAINER_API = """
import dataclasses
from typing import NamedTuple

from typing import Annotated

import pydantic
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from tesseract_core.runtime import Array, Differentiable, Float32

A = Differentiable[Array[(None,), Float32]]


class NT(NamedTuple):
    a: A
    n: int = 3


@dataclasses.dataclass
class DC:
    a: A
    n: int = 3
    tags: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class FrozenDC:
    a: A


@pydantic.dataclasses.dataclass
class PDC:
    a: A


class TD(TypedDict):
    a: A
    n: NotRequired[int]
    scale: NotRequired[Annotated[float, Field(gt=0)]]


class InputSchema(BaseModel):
    nt: NT
    dc: DC
    frozen_dc: FrozenDC
    pdc: PDC
    td: TD


class OutputSchema(BaseModel):
    y: A


def apply(inputs):
    total = inputs.nt.a + inputs.dc.a + inputs.frozen_dc.a + inputs.pdc.a + inputs.td["a"]
    return OutputSchema(y=total)


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return {path: cotangent_vector["y"] for path in vjp_inputs}


def abstract_eval(abstract_inputs):
    return {"y": {"shape": abstract_inputs.nt.a.shape, "dtype": "float32"}}
"""

_CONTAINER_PATHS = {"nt.[]", "dc.a", "frozen_dc.a", "pdc.a", "td.a"}


@pytest.fixture(scope="module")
def container_api(tmp_path_factory):
    """A Tesseract whose only differentiable arrays sit inside non-model containers."""
    import importlib.util

    path = tmp_path_factory.mktemp("containers") / "tesseract_api.py"
    path.write_text(_CONTAINER_API)
    spec = importlib.util.spec_from_file_location("container_tesseract_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


def _abstract(shape):
    return {"shape": list(shape), "dtype": "float32"}


def test_arrays_in_namedtuples_dataclasses_and_typeddicts_are_differentiable(
    container_api,
):
    _, api = container_api
    ApplyInputSchema, _ = create_apply_schema(api.InputSchema, api.OutputSchema)
    assert set(ApplyInputSchema.differentiable_arrays) == _CONTAINER_PATHS


def test_apply_schema_keeps_the_user_container_types(container_api):
    _, api = container_api
    ApplyInputSchema, _ = create_apply_schema(api.InputSchema, api.OutputSchema)
    x = np.ones(2, dtype=np.float32)
    inputs = ApplyInputSchema.model_validate(
        {
            "inputs": {
                "nt": [x],
                "dc": {"a": x},
                "frozen_dc": {"a": x},
                "pdc": {"a": x},
                "td": {"a": x},
            }
        }
    ).inputs
    assert type(inputs.nt) is api.NT
    assert type(inputs.dc) is api.DC
    assert type(inputs.frozen_dc) is api.FrozenDC
    assert type(inputs.pdc) is api.PDC


def test_abstract_eval_schema_reaches_arrays_in_containers(container_api):
    _, api = container_api
    AbstractInputSchema, _ = create_abstract_eval_schema(
        api.InputSchema, api.OutputSchema
    )
    inputs = AbstractInputSchema.model_validate(
        {
            "inputs": {
                "nt": [_abstract((2,))],
                "dc": {"a": _abstract((2,))},
                "frozen_dc": {"a": _abstract((2,))},
                "pdc": {"a": _abstract((2,))},
                "td": {"a": _abstract((2,))},
            }
        }
    ).inputs
    for value in (inputs.nt.a, inputs.dc.a, inputs.frozen_dc.a, inputs.pdc.a):
        assert isinstance(value, ShapeDType)
    # Rebuilt dataclasses subclass the user's own, so methods and isinstance work
    assert isinstance(inputs.dc, api.DC)
    assert isinstance(inputs.frozen_dc, api.FrozenDC)
    assert isinstance(inputs.pdc, api.PDC)
    assert isinstance(inputs.td["a"], ShapeDType)
    # Defaults and constraints of the rebuilt containers are kept
    assert inputs.nt.n == 3
    assert inputs.dc.n == 3
    assert inputs.dc.tags == []
    with pytest.raises(ValidationError, match="greater than 0"):
        AbstractInputSchema.model_validate(
            {
                "inputs": {
                    "nt": [_abstract((2,))],
                    **{k: {"a": _abstract((2,))} for k in ("dc", "frozen_dc", "pdc")},
                    "td": {"a": _abstract((2,)), "scale": -1.0},
                }
            }
        )


def test_tesseract_differentiates_arrays_in_containers(container_api):
    from tesseract_core import Tesseract

    path, _ = container_api
    tesseract = Tesseract.from_tesseract_api(path)
    x = np.ones(2, dtype=np.float32)
    inputs = {
        "nt": [x],
        "dc": {"a": x},
        "frozen_dc": {"a": x},
        "pdc": {"a": x},
        "td": {"a": x},
    }
    abstract = tesseract.abstract_eval(
        {
            "nt": [_abstract((2,))],
            **{k: {"a": _abstract((2,))} for k in ("dc", "frozen_dc", "pdc", "td")},
        }
    )
    assert tuple(abstract["y"]["shape"]) == (2,)

    vjp_inputs = ["nt.[0]", "dc.a", "frozen_dc.a", "pdc.a", "td.a"]
    vjp = tesseract.vector_jacobian_product(inputs, vjp_inputs, ["y"], {"y": x})
    assert set(vjp) == set(vjp_inputs)


def test_dropping_fields_inside_containers(container_api):
    """A function returning None removes a NamedTuple or TypedDict field."""
    from tesseract_core.runtime.schema_generation import apply_function_to_model_tree
    from tesseract_core.runtime.schema_types import (
        PydanticArrayAnnotation,
        safe_issubclass,
    )

    _, api = container_api
    Stripped = apply_function_to_model_tree(
        api.InputSchema,
        lambda t, _: None if safe_issubclass(t, PydanticArrayAnnotation) else t,
    )
    fields = Stripped.model_fields
    assert fields["nt"].annotation._fields == ("n",)
    assert fields["nt"].annotation._field_defaults == {"n": 3}
    # A dataclass is rebuilt as a subclass, which cannot remove an inherited field,
    # just as for pydantic models
    assert [f.name for f in dataclasses.fields(fields["dc"].annotation)] == [
        "a",
        "n",
        "tags",
    ]
    assert set(fields["td"].annotation.__annotations__) == {"n", "scale"}


def test_check_gradients_reaches_arrays_in_containers(container_api):
    """Gradient checks now cover arrays inside these containers, frozen ones too."""
    from tesseract_core.runtime.testing.finite_differences import check_gradients

    _, api = container_api
    x = [1.0, 2.0]
    inputs = {
        "inputs": {
            "nt": [x],
            "dc": {"a": x},
            "frozen_dc": {"a": x},
            "pdc": {"a": x},
            "td": {"a": x},
        }
    }
    [(endpoint, failures, _)] = check_gradients(
        api,
        inputs,
        endpoints=["vector_jacobian_product"],
        input_paths=[*sorted(_CONTAINER_PATHS - {"nt.[]"}), "nt.[0]"],
        max_evals=50,
        show_progress=False,
        seed=0,
    )
    assert endpoint == "vector_jacobian_product"
    assert failures == []


_CUSTOM_DATACLASS_API = """
import dataclasses

import numpy as np
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float32

A = Differentiable[Array[(None,), Float32]]


class AutoDataclass:
    \"\"\"Makes every subclass a frozen dataclass, as some libraries' base classes do.\"\"\"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(frozen=True)(cls)


class Params(AutoDataclass):
    w: A
    b: A

    def combined(self):
        return np.asarray(self.w) + np.asarray(self.b)


@dataclasses.dataclass(frozen=True)
class Scaled:
    w: A
    scale: float

    def __init__(self, weights, factor=2.0):
        object.__setattr__(self, "w", weights)
        object.__setattr__(self, "scale", factor)


class InputSchema(BaseModel):
    params: Params
    scaled: Scaled


class OutputSchema(BaseModel):
    y: A


def apply(inputs):
    assert isinstance(inputs.params, Params) and isinstance(inputs.scaled, Scaled)
    y = inputs.params.combined() + inputs.scaled.scale * np.asarray(inputs.scaled.w)
    return OutputSchema(y=y)


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return {path: cotangent_vector["y"] for path in vjp_inputs}


def abstract_eval(abstract_inputs):
    assert isinstance(abstract_inputs.params, Params)
    return {"y": {"shape": abstract_inputs.params.w.shape, "dtype": "float32"}}
"""


def test_custom_dataclasses_as_schema_fields(tmp_path):
    """Dataclasses with methods, custom constructors and dataclass-making bases work."""
    from tesseract_core import Tesseract

    path = tmp_path / "tesseract_api.py"
    path.write_text(_CUSTOM_DATACLASS_API)
    tesseract = Tesseract.from_tesseract_api(path)
    published = tesseract.openapi_schema["components"]["schemas"]["ApplyInputSchema"]
    assert set(published["differentiable_arrays"]) == {
        "params.w",
        "params.b",
        "scaled.w",
    }

    x = np.ones(2, dtype=np.float32)
    inputs = {"params": {"w": x, "b": x}, "scaled": {"w": x, "scale": 3.0}}
    np.testing.assert_allclose(tesseract.apply(inputs)["y"], [5.0, 5.0])
    abstract = {"shape": [2], "dtype": "float32"}
    tesseract.abstract_eval(
        {
            "params": {"w": abstract, "b": abstract},
            "scaled": {"w": abstract, "scale": 3.0},
        }
    )
    vjp = tesseract.vector_jacobian_product(
        inputs, ["params.w", "scaled.w"], ["y"], {"y": x}
    )
    assert set(vjp) == {"params.w", "scaled.w"}


def test_containers_with_unresolvable_annotations_are_leaves():
    """If get_type_hints can't resolve a container's annotations, it isn't recursed into.

    pydantic can still resolve names defined in the enclosing function, so such
    a schema keeps working as it did before containers were recursed into.
    """

    class Local(BaseModel):
        n: int

    @dataclasses.dataclass
    class Holder:
        local: "Local"

    class Model(BaseModel):
        holder: Holder

    InputSchema, _ = create_apply_schema(Model, Model)
    value = InputSchema.model_validate({"inputs": {"holder": {"local": {"n": 1}}}})
    assert value.inputs.holder.local.n == 1


def test_rebuilt_dataclass_fields_keep_their_options():
    """A re-declared field keeps its default or default factory."""
    from tesseract_core.runtime.schema_generation import apply_function_to_model_tree

    @dataclasses.dataclass
    class Settings:
        count: int = 3
        tags: list = dataclasses.field(default_factory=list)

    class Model(BaseModel):
        settings: Settings

    Retyped = apply_function_to_model_tree(
        Model, lambda t, _: float if t is int else (tuple if t is list else t)
    )
    new = Retyped.model_fields["settings"].annotation
    assert issubclass(new, Settings)
    assert {f.name: f.type for f in dataclasses.fields(new)} == {
        "count": float,
        "tags": tuple,
    }
    assert new().count == 3
    assert new().tags == []


def test_containers_left_without_fields_are_dropped():
    """A NamedTuple or TypedDict whose every field is removed is removed itself."""
    from typing import NamedTuple

    from typing_extensions import TypedDict

    from tesseract_core.runtime.schema_generation import apply_function_to_model_tree
    from tesseract_core.runtime.schema_types import (
        PydanticArrayAnnotation,
        safe_issubclass,
    )

    A = Array[(None,), Float32]

    class OnlyArrayTuple(NamedTuple):
        a: A

    class OnlyArrayDict(TypedDict):
        a: A

    class Outer(NamedTuple):
        inner_tuple: OnlyArrayTuple
        inner_dict: OnlyArrayDict
        n: int

    class Model(BaseModel):
        outer: Outer

    Stripped = apply_function_to_model_tree(
        Model,
        lambda t, _: None if safe_issubclass(t, PydanticArrayAnnotation) else t,
    )
    assert Stripped.model_fields["outer"].annotation._fields == ("n",)
