from types import ModuleType

import numpy as np
import pytest
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.finite_differences import check_gradients
from tesseract_core.runtime.tree_transforms import get_at_path


class DummyModule(ModuleType):
    def __init__(self, *args, correct_gradients: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.correct_gradients = correct_gradients

    class InputSchema(BaseModel):
        in_data: Differentiable[Array[(None, 3), Float32]]
        in_dict: dict[str, Differentiable[Array[(3, 3, 3), Float32]]]
        in_scalar: Differentiable[Float32]
        in_aux: str

    class OutputSchema(BaseModel):
        out_data: Differentiable[Array[(None, 3), Float32]]
        out_dict: dict[str, Differentiable[Array[(3, 3, 3), Float32]]]
        out_scalar: Differentiable[Float32]
        out_aux: str

    def apply(self, inputs: InputSchema) -> OutputSchema:
        return {
            "out_data": np.zeros_like(inputs.in_data),
            "out_dict": {
                key: np.zeros_like(value) for key, value in inputs.in_dict.items()
            },
            "out_scalar": np.zeros_like(inputs.in_scalar),
            "out_aux": inputs.in_aux,
        }

    def jacobian(
        self,
        inputs: InputSchema,
        jac_inputs: set[str],
        jac_outputs: set[str],
    ):
        outputs = self.apply(inputs)
        if self.correct_gradients:
            make_array = np.zeros
        else:
            make_array = np.ones

        return {
            key_out: {
                key_in: make_array(
                    (
                        *get_at_path(outputs, key_out).shape,
                        *get_at_path(inputs, key_in).shape,
                    )
                )
                for key_in in jac_inputs
            }
            for key_out in jac_outputs
        }

    def jacobian_vector_product(
        self,
        inputs: InputSchema,
        jvp_inputs: set[str],
        jvp_outputs: set[str],
        tangent_vector: Array[(None,), Float32],
    ):
        outputs = self.apply(inputs)
        if self.correct_gradients:
            make_array = np.zeros
        else:
            make_array = np.ones
        return {
            key_out: make_array(get_at_path(outputs, key_out).shape)
            for key_out in jvp_outputs
        }

    def vector_jacobian_product(
        self,
        inputs: InputSchema,
        vjp_inputs: set[str],
        vjp_outputs: set[str],
        cotangent_vector: Array[(None,), Float32],
    ):
        if self.correct_gradients:
            make_array = np.zeros
        else:
            make_array = np.ones
        return {
            key_in: make_array(get_at_path(inputs, key_in).shape)
            for key_in in vjp_inputs
        }


rng = np.random.default_rng(0)
input_data = {
    "in_data": rng.random((10, 3)),
    "in_dict": {"key": rng.random((3, 3, 3))},
    "in_scalar": rng.random(1)[0],
    "in_aux": "auxiliary",
}


@pytest.mark.parametrize("input_paths", [None, ["in_data"], ["in_dict.{key}"]])
@pytest.mark.parametrize("output_paths", [None, ["out_data", "out_dict.{key}"]])
@pytest.mark.parametrize("endpoints", [None, ["jacobian"]])
def test_check_gradients(input_paths, output_paths, endpoints):
    dummy_module_bad = DummyModule("dummy_module", correct_gradients=False)

    result_iter = check_gradients(
        dummy_module_bad,
        {"inputs": input_data},
        base_dir=None,
        input_paths=input_paths,
        output_paths=output_paths,
        endpoints=endpoints,
        max_evals=10,
    )

    run_endpoints = []
    for endpoint, failures, num_evals in result_iter:
        run_endpoints.append(endpoint)

        # everything should fail (all gradients are wrong)
        assert len(failures) == num_evals

        for failure in failures:
            assert not failure.exception

    # Now try again with correct gradients
    dummy_module_good = DummyModule("dummy_module", correct_gradients=True)
    result_iter = check_gradients(
        dummy_module_good,
        {"inputs": input_data},
        base_dir=None,
        input_paths=input_paths,
        output_paths=output_paths,
        endpoints=endpoints,
        max_evals=10,
    )

    for _, failures, _ in result_iter:
        assert not failures

    if endpoints is not None:
        assert run_endpoints == endpoints
    else:
        assert run_endpoints == [
            "jacobian",
            "jacobian_vector_product",
            "vector_jacobian_product",
        ]
