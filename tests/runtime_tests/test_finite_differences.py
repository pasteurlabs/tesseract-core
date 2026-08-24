from types import ModuleType

import numpy as np
import pytest
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.testing.finite_differences import (
    check_gradients,
    expand_path_pattern,
)
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


class TestExpandPathPatternOptionalFields:
    """Optional container fields must not abort path expansion.

    The pattern comes from the schema, so an optional field that was simply
    not supplied is a normal input rather than a bad path. Every branch of
    the walk raises on ``None`` though: ``[]`` and ``{}`` iterate it and a
    named part subscripts it, so a Tesseract with an optional container
    input used to fail before checking a single gradient.
    """

    @pytest.mark.parametrize(
        "pattern,inputs",
        [
            ("a.[].b", {"a": None}),
            ("a.b", {"a": None}),
            ("a.{}.b", {"a": None}),
        ],
        ids=["optional_list", "optional_submodel", "optional_dict"],
    )
    def test_absent_optional_container_expands_to_nothing(self, pattern, inputs):
        assert expand_path_pattern(pattern, inputs) == []

    def test_none_entry_inside_a_populated_list_is_skipped(self):
        """The present entries still expand; only the missing one drops out."""
        assert expand_path_pattern("a.[].b", {"a": [{"b": 1}, None]}) == ["a.[0].b"]

    @pytest.mark.parametrize(
        "pattern,inputs,expected",
        [
            ("a.[].b", {"a": [{"b": 1}, {"b": 2}]}, ["a.[0].b", "a.[1].b"]),
            ("a.{}.b", {"a": {"x": {"b": 1}}}, ["a.{x}.b"]),
            ("a.b", {"a": {"b": 1}}, ["a.b"]),
        ],
        ids=["list", "dict", "plain"],
    )
    def test_populated_paths_are_unchanged(self, pattern, inputs, expected):
        assert expand_path_pattern(pattern, inputs) == expected


class _OptionalExtra(BaseModel):
    w: Differentiable[Array[(3,), Float32]]


class OptionalContainerModule(ModuleType):
    """A schema with an optional sub-model, as real Tesseracts have.

    Optional initial conditions, boundary data or preconditioner state are
    ordinary inputs. The differentiable path ``extra.w`` is still declared
    when ``extra`` is absent, so the path walk meets ``None``.
    """

    class InputSchema(BaseModel):
        x: Differentiable[Array[(3,), Float32]]
        extra: _OptionalExtra | None = None

    class OutputSchema(BaseModel):
        y: Differentiable[Array[(3,), Float32]]

    def apply(self, inputs: InputSchema) -> OutputSchema:
        y = 2.0 * np.asarray(inputs.x, dtype=np.float32)
        if inputs.extra is not None:
            y = y + np.asarray(inputs.extra.w, dtype=np.float32)
        return {"y": y}

    def jacobian(
        self,
        inputs: InputSchema,
        jac_inputs: set[str],
        jac_outputs: set[str],
    ):
        return {
            "y": {
                p: (2.0 if p == "x" else 1.0) * np.eye(3, dtype=np.float32)
                for p in jac_inputs
            }
        }


def test_check_gradients_runs_with_an_absent_optional_container():
    """End-to-end: an absent optional input must not abort the whole check.

    ``extra.w`` stays in ``differentiable_arrays`` whether or not ``extra``
    was supplied, so the path walk meets ``None`` and every branch of it
    raises. On an unguarded tree this fails with ``TypeError: 'NoneType'
    object is not subscriptable`` before a single gradient is checked.
    """
    module = OptionalContainerModule("optional_container_module")

    num_evals_total = 0
    for _endpoint, failures, num_evals in check_gradients(
        module,
        {"inputs": {"x": np.ones(3, dtype=np.float32), "extra": None}},
        base_dir=None,
        endpoints=["jacobian"],
        max_evals=6,
        seed=0,
    ):
        num_evals_total += num_evals
        assert not failures

    assert num_evals_total > 0, "nothing was checked, so this proves nothing"
