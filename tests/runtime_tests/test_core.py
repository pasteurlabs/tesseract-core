# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for validation logic in tesseract_runtime/core.py.

Test a couple of edge cases and failure modes regarding schema validation and AD inputs / outputs.
"""

from collections.abc import Sequence
from types import ModuleType

import numpy as np
import pytest
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.core import create_endpoints
from tesseract_core.runtime.tree_transforms import get_at_path


@pytest.fixture
def testmodule():
    class _InputSchema(BaseModel):
        array_seq: Sequence[Differentiable[Array[(3, 1), Float32]]]
        array_dict: dict[str, Differentiable[Array[(None,), Float32]]]
        scalar: Float32
        scalar_diff: Differentiable[Float32]

    class _OutputSchema(BaseModel):
        result_seq: Sequence[Differentiable[Array[(10,), Float32]]]
        result_arr: Array[(None,), Float32]

    class FakeModule(ModuleType):
        @property
        def InputSchema(self):
            return _InputSchema

        @property
        def OutputSchema(self):
            return _OutputSchema

        def apply(self, inputs):
            res_seq = [np.ones(10)] * 3
            res_float = np.zeros(5)
            return _OutputSchema(result_seq=res_seq, result_arr=res_float)

        def jacobian(self, inputs, jac_inputs, jac_outputs):
            out = {}
            for out_key in jac_outputs:
                out_shape = (10,)
                out[out_key] = {}
                for in_key in jac_inputs:
                    if in_key.startswith("array_seq"):
                        container, idx = in_key.split(".")
                        idx = int(idx[1:-1])
                    elif in_key.startswith("array_dict"):
                        container, idx = in_key.split(".")
                        idx = idx[1:-1]
                    else:
                        container = in_key
                        idx = None

                    if container == "array_seq":
                        out[out_key][in_key] = np.ones(
                            (*out_shape, *inputs.array_seq[idx].shape)
                        )
                    elif container == "array_dict":
                        out[out_key][in_key] = np.ones(
                            (*out_shape, *inputs.array_dict[idx].shape)
                        )
                    else:
                        out[out_key][in_key] = np.ones(out_shape)
            return out

        def jacobian_vector_product(
            self, inputs, jvp_inputs, jvp_outputs, tangent_vector
        ):
            out = {}
            for out_key in jvp_outputs:
                out_shape = (10,)
                out[out_key] = np.ones(out_shape)
            return out

        def vector_jacobian_product(
            self, inputs, vjp_inputs, vjp_outputs, cotangent_vector
        ):
            out = {}
            for in_key in vjp_inputs:
                if in_key.startswith("array_seq"):
                    container, idx = in_key.split(".")
                    idx = int(idx[1:-1])
                elif in_key.startswith("array_dict"):
                    container, idx = in_key.split(".")
                    idx = idx[1:-1]
                else:
                    container = in_key
                    idx = None

                if container == "array_seq":
                    out[in_key] = np.ones(inputs.array_seq[idx].shape)
                elif container == "array_dict":
                    out[in_key] = np.ones(inputs.array_dict[idx].shape)
                else:
                    out[in_key] = np.ones(())

            return out

        def abstract_eval(self, inputs):
            return {
                "result_seq": [{"shape": (10,), "dtype": "float32"}] * 3,
                "result_arr": {"shape": (5,), "dtype": "float32"},
            }

    return FakeModule("TestModule")


test_input = {
    "array_seq": [
        np.array([1.0, 2.0, 3.0]).reshape(3, 1),
        np.array([4.0, 5.0, 6.0]).reshape(3, 1),
    ],
    "array_dict": {"a": np.array([7.0, 8.0, 9.0, 10.0]), "b": np.array([11.0, 12.0])},
    "scalar": 1.0,
    "scalar_diff": np.array(2.0),
}
test_output = {
    "result_seq": [np.ones(10)] * 3,
    "result_arr": np.zeros(5),
}


def _recurse_pytree(pytree, func):
    if isinstance(pytree, dict):
        return {k: _recurse_pytree(v, func) for k, v in pytree.items()}
    elif isinstance(pytree, list | tuple):
        return [_recurse_pytree(v, func) for v in pytree]
    else:
        return func(pytree)


def _find_endpoint(endpoint_list, endpoint_name):
    for endpoint in endpoint_list:
        if endpoint.__name__ == endpoint_name:
            schema = endpoint.__annotations__["payload"]
            return endpoint, schema
    raise ValueError(f"Endpoint {endpoint_name} not found.")


def test_create_endpoints(testmodule):
    endpoints = create_endpoints(testmodule)
    assert endpoints


def test_apply_endpoint(testmodule):
    endpoints = create_endpoints(testmodule)

    apply_endpoint, EndpointSchema = _find_endpoint(endpoints, "apply")
    assert apply_endpoint.__name__ == "apply"

    inputs = EndpointSchema.model_validate({"inputs": test_input})
    result = apply_endpoint(inputs)
    np.testing.assert_equal(result.model_dump(), test_output)


def test_apply_returns_dict(testmodule):
    # Mock the apply method to return a dictionary
    def apply(inputs):
        return test_output

    testmodule.apply = apply

    endpoints = create_endpoints(testmodule)

    apply_endpoint, EndpointSchema = _find_endpoint(endpoints, "apply")
    assert apply_endpoint.__name__ == "apply"

    inputs = EndpointSchema.model_validate({"inputs": test_input})
    result = apply_endpoint(inputs)
    np.testing.assert_equal(result.model_dump(), test_output)


def test_abstract_eval_endpoint(testmodule):
    endpoints = create_endpoints(testmodule)

    abstract_eval_endpoint, EndpointSchema = _find_endpoint(endpoints, "abstract_eval")
    assert abstract_eval_endpoint.__name__ == "abstract_eval"

    def _array_to_shapedtype(obj):
        obj = np.asarray(obj)
        return {"shape": obj.shape, "dtype": str(obj.dtype)}

    inputs_shapedtype = _recurse_pytree(test_input, _array_to_shapedtype)
    inputs = EndpointSchema.model_validate({"inputs": inputs_shapedtype})
    result = abstract_eval_endpoint(inputs)
    assert result

    with pytest.raises(ValueError):
        failing_inputs_shapedtype = inputs_shapedtype.copy()
        failing_inputs_shapedtype["scalar"]["shape"] = (1, 2, 3)
        inputs = EndpointSchema.model_validate({"inputs": failing_inputs_shapedtype})


@pytest.mark.parametrize(
    "ad_inout",
    [
        # Valid combinations for jac_inputs and jac_outputs
        ({"array_seq.[0]", "array_dict.{a}", "scalar_diff"}, {"result_seq.[0]"}),
        ({"array_seq.[0]", "array_dict.{a}"}, {"result_seq.[0]"}),
        ({"array_seq.[0]", "array_dict.{a}", "scalar_diff"}, {"result_seq.[0]"}),
        ({"array_seq.[0]", "array_dict.{a}"}, {"result_seq.[0]"}),
        ({"array_seq.[0]", "array_seq.[1]"}, {"result_seq.[0]"}),
    ],
)
@pytest.mark.parametrize(
    "endpoint_name", ["jacobian", "jacobian_vector_product", "vector_jacobian_product"]
)
def test_ad_endpoint(testmodule, ad_inout, endpoint_name):
    endpoints = create_endpoints(testmodule)

    endpoint_func, EndpointSchema = _find_endpoint(endpoints, endpoint_name)
    assert endpoint_func.__name__ == endpoint_name

    ad_inp, ad_out = ad_inout

    if endpoint_name == "jacobian":
        inputs = {"inputs": test_input, "jac_inputs": ad_inp, "jac_outputs": ad_out}
    elif endpoint_name == "jacobian_vector_product":
        tangent_vector = {k: get_at_path(test_input, k) for k in ad_inp}
        inputs = {
            "inputs": test_input,
            "jvp_inputs": ad_inp,
            "jvp_outputs": ad_out,
            "tangent_vector": tangent_vector,
        }
    elif endpoint_name == "vector_jacobian_product":
        cotangent_vector = {k: get_at_path(test_output, k) for k in ad_out}
        inputs = {
            "inputs": test_input,
            "vjp_inputs": ad_inp,
            "vjp_outputs": ad_out,
            "cotangent_vector": cotangent_vector,
        }

    inputs = EndpointSchema.model_validate(inputs)
    result = endpoint_func(inputs)
    assert result


@pytest.mark.parametrize(
    "ad_inout_invalid",
    [
        # -- Invalid jac_inputs, valid jac_outputs --
        # Unknown keys
        (
            "String should match pattern",
            {"array_seq.[0]", "array_dict.{a}", "scalar_diff", "invalid_key"},
            {"result_seq.[0]"},
        ),
        # Invalid container index
        (
            "String should match pattern",
            {"array_seq.[foobar]", "array_dict.{a}", "scalar_diff", "invalid_key"},
            {"result_seq.[0]"},
        ),
        # Non-existent container index
        (
            "Could not find input path array_seq.[100]",
            {"array_seq.[100]", "array_dict.{a}", "scalar_diff"},
            {"result_seq.[0]"},
        ),
        (
            "Could not find input path array_dict.{xyz}",
            {"array_seq.[0]", "array_dict.{xyz}", "scalar_diff"},
            {"result_seq.[0]"},
        ),
        # Non-differentiable keys
        (
            "String should match pattern",
            {"array_seq.[0]", "array_dict.{a}", "scalar"},
            {"result_seq.[0]"},
        ),
        # Container without index
        (
            "String should match pattern",
            {"array_seq", "array_dict.{a}", "scalar_diff"},
            {"result_seq.[0]"},
        ),
        (
            "String should match pattern",
            {"array_seq.[0]", "array_dict", "scalar_diff"},
            {"result_seq.[0]"},
        ),
        # Missing keys
        ("at least 1 item", set(), {"result_seq.[0]"}),
        # -- Valid jac_inputs, invalid jac_outputs --
        # Unknown keys
        (
            "String should match pattern",
            {"array_seq.[0]", "array_dict.{a}", "scalar_diff"},
            {"result_seq.[0]", "invalid_key"},
        ),
        # Non-differentiable keys
        (
            "String should match pattern",
            {"array_seq.[0]", "array_dict.{a}", "scalar_diff"},
            {"result_seq.[0]", "result_arr"},
        ),
        # Container without index
        (
            "String should match pattern",
            {"array_seq.[0]", "array_dict.{a}", "scalar_diff"},
            {"result_seq"},
        ),
        # Missing keys
        ("at least 1 item", {"scalar_diff"}, set()),
    ],
)
@pytest.mark.parametrize(
    "endpoint_name", ["jacobian", "jacobian_vector_product", "vector_jacobian_product"]
)
def test_ad_endpoint_invalid(testmodule, ad_inout_invalid, endpoint_name):
    endpoints = create_endpoints(testmodule)

    endpoint_func, EndpointSchema = _find_endpoint(endpoints, endpoint_name)
    assert endpoint_func.__name__ == endpoint_name

    msg, ad_inp, ad_out = ad_inout_invalid

    if endpoint_name == "jacobian":
        inputs = {"inputs": test_input, "jac_inputs": ad_inp, "jac_outputs": ad_out}
    elif endpoint_name == "jacobian_vector_product":
        tangent_vector = {}
        for k in ad_inp:
            try:
                tangent_vector[k] = get_at_path(test_input, k)
            except (KeyError, ValueError, IndexError):
                # Path is invalid, add a dummy value
                tangent_vector[k] = np.ones(10)
        inputs = {
            "inputs": test_input,
            "jvp_inputs": ad_inp,
            "jvp_outputs": ad_out,
            "tangent_vector": tangent_vector,
        }
    elif endpoint_name == "vector_jacobian_product":
        cotangent_vector = {}
        for k in ad_out:
            try:
                cotangent_vector[k] = get_at_path(test_output, k)
            except (KeyError, ValueError, IndexError):
                # Path is invalid, add a dummy value
                cotangent_vector[k] = np.ones(10)
        inputs = {
            "inputs": test_input,
            "vjp_inputs": ad_inp,
            "vjp_outputs": ad_out,
            "cotangent_vector": cotangent_vector,
        }

    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        inputs = EndpointSchema.model_validate(inputs)
        endpoint_func(inputs)

    assert msg in str(excinfo.value)


@pytest.mark.parametrize(
    "endpoint_name", ["jacobian_vector_product", "vector_jacobian_product"]
)
@pytest.mark.parametrize("failure_mode", ["missing", "extra", "invalid"])
def test_ad_endpoint_bad_tangent(testmodule, endpoint_name, failure_mode):
    endpoints = create_endpoints(testmodule)

    endpoint_func, EndpointSchema = _find_endpoint(endpoints, endpoint_name)
    assert endpoint_func.__name__ == endpoint_name

    ad_inp = {"array_seq.[0]", "array_dict.{a}", "scalar_diff"}
    ad_out = {"result_seq.[0]"}

    if endpoint_name == "jacobian_vector_product":
        if failure_mode == "missing":
            tangent_vector = {}
            msg = "Expected tangent vector with keys"
        elif failure_mode == "extra":
            tangent_vector = {k: get_at_path(test_input, k) for k in ad_inp}
            tangent_vector["invalid_key"] = np.ones(10)
            msg = "String should match pattern"
        elif failure_mode == "invalid":
            tangent_vector = {k: "ahoy" for k in ad_inp}
            msg = "Got invalid dtype"

        inputs = {
            "inputs": test_input,
            "jvp_inputs": ad_inp,
            "jvp_outputs": ad_out,
            "tangent_vector": tangent_vector,
        }

    elif endpoint_name == "vector_jacobian_product":
        if failure_mode == "missing":
            cotangent_vector = {}
            msg = "Expected cotangent vector with keys"
        elif failure_mode == "extra":
            cotangent_vector = {k: get_at_path(test_output, k) for k in ad_out}
            cotangent_vector["invalid_key"] = np.ones(10)
            msg = "String should match pattern"
        elif failure_mode == "invalid":
            cotangent_vector = {k: "ahoy" for k in ad_out}
            msg = "Got invalid dtype"

        inputs = {
            "inputs": test_input,
            "vjp_inputs": ad_inp,
            "vjp_outputs": ad_out,
            "cotangent_vector": cotangent_vector,
        }

    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        inputs = EndpointSchema.model_validate(inputs)
        endpoint_func(inputs)

    assert msg in str(excinfo.value)
