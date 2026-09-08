from types import ModuleType
from typing import Any

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


class CountingVjpModule(DummyModule):
    """DummyModule that records how many times the VJP endpoint is called."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vjp_calls = 0

    def vector_jacobian_product(self, *args, **kwargs):
        self.vjp_calls += 1
        return super().vector_jacobian_product(*args, **kwargs)


def test_vjp_sweep_is_shared_across_sampled_indices():
    """The VJP path must not re-sweep the outputs once per sampled index.

    One VJP call with a one-hot cotangent already returns the gradient with
    respect to every element of the input path, so a single sweep answers for
    every sampled index of that path pair. Keeping only one element and
    re-running the sweep cost ``n_sampled x n_output_elements`` calls, where
    ``jacobian`` and ``jacobian_vector_product`` cost one per item.
    """
    module = CountingVjpModule("dummy_module", correct_gradients=True)

    num_evals_total = 0
    for _endpoint, failures, num_evals in check_gradients(
        module,
        {"inputs": input_data},
        base_dir=None,
        input_paths=["in_data"],
        output_paths=["out_dict.{key}"],
        endpoints=["vector_jacobian_product"],
        max_evals=10,
        seed=0,
    ):
        num_evals_total += num_evals
        assert not failures

    n_output_elements = int(np.prod(input_data["in_dict"]["key"].shape))
    # One sweep, reused by every sampled index, rather than one sweep each.
    assert module.vjp_calls == n_output_elements
    assert module.vjp_calls < num_evals_total * n_output_elements


class RecordingVjpModule(CountingVjpModule):
    """CountingVjpModule that additionally records cotangent one-hot coordinates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorded_cotangent_coords = []

    def vector_jacobian_product(
        self,
        inputs: DummyModule.InputSchema,
        vjp_inputs: set[str],
        vjp_outputs: set[str],
        cotangent_vector: dict[str, Any],
    ):
        for out_key in vjp_outputs:
            cot = np.asarray(cotangent_vector[out_key])
            nonzero = np.argwhere(cot != 0)
            for coord in nonzero:
                self.recorded_cotangent_coords.append(tuple(int(c) for c in coord))
        return super().vector_jacobian_product(
            inputs, vjp_inputs, vjp_outputs, cotangent_vector
        )


def test_vjp_output_sampling_bounds_calls():
    """Output sampling bounds VJP endpoint calls to min(n_output_elements, max_output_samples)."""
    module = CountingVjpModule("dummy_module", correct_gradients=True)
    num_evals_total = 0
    for _endpoint, failures, num_evals in check_gradients(
        module,
        {"inputs": input_data},
        base_dir=None,
        input_paths=["in_data"],
        output_paths=["out_dict.{key}"],
        endpoints=["vector_jacobian_product"],
        max_evals=10,
        max_output_samples=5,
        seed=0,
    ):
        num_evals_total += num_evals
        assert not failures

    assert num_evals_total > 0
    assert module.vjp_calls == 5


def test_vjp_output_sampling_shared_across_sampled_input_rows():
    """All sampled input rows for a path pair must share the same sampled VJP sweep."""
    module = CountingVjpModule("dummy_module", correct_gradients=True)
    num_evals_total = 0
    for _endpoint, failures, num_evals in check_gradients(
        module,
        {"inputs": input_data},
        base_dir=None,
        input_paths=["in_data"],
        output_paths=["out_dict.{key}"],
        endpoints=["vector_jacobian_product"],
        max_evals=20,
        max_output_samples=5,
        seed=0,
    ):
        num_evals_total += num_evals
        assert not failures

    assert num_evals_total > 1
    # Exactly 5 VJP calls shared across all sampled input rows, NOT num_evals * 5
    assert module.vjp_calls == 5


def test_vjp_output_sampling_cap_greater_than_output_size_stays_exhaustive():
    """When max_output_samples >= output elements, the check stays exhaustive without random sampling."""
    module = CountingVjpModule("dummy_module", correct_gradients=True)
    for _endpoint, failures, _ in check_gradients(
        module,
        {"inputs": input_data},
        base_dir=None,
        input_paths=["in_data"],
        output_paths=["out_dict.{key}"],
        endpoints=["vector_jacobian_product"],
        max_evals=10,
        max_output_samples=100,
        seed=0,
    ):
        assert not failures

    n_output_elements = int(np.prod(input_data["in_dict"]["key"].shape))
    assert module.vjp_calls == n_output_elements


def test_vjp_output_sampling_coordinates_are_unique():
    """Sampled output coordinates must be distinct (sampled without replacement)."""
    module = RecordingVjpModule("dummy_module", correct_gradients=True)
    for _endpoint, failures, _ in check_gradients(
        module,
        {"inputs": input_data},
        base_dir=None,
        input_paths=["in_data"],
        output_paths=["out_dict.{key}"],
        endpoints=["vector_jacobian_product"],
        max_evals=10,
        max_output_samples=7,
        seed=42,
    ):
        assert not failures

    assert len(module.recorded_cotangent_coords) == 7
    assert len(set(module.recorded_cotangent_coords)) == 7


def test_vjp_output_sampling_is_deterministic_with_seed():
    """Identical seeds must yield identical sampled output coordinates in the same order."""
    module1 = RecordingVjpModule("dummy_module", correct_gradients=True)
    for _endpoint, failures, _ in check_gradients(
        module1,
        {"inputs": input_data},
        base_dir=None,
        input_paths=["in_data"],
        output_paths=["out_dict.{key}"],
        endpoints=["vector_jacobian_product"],
        max_evals=10,
        max_output_samples=5,
        seed=123,
    ):
        assert not failures

    module2 = RecordingVjpModule("dummy_module", correct_gradients=True)
    for _endpoint, failures, _ in check_gradients(
        module2,
        {"inputs": input_data},
        base_dir=None,
        input_paths=["in_data"],
        output_paths=["out_dict.{key}"],
        endpoints=["vector_jacobian_product"],
        max_evals=10,
        max_output_samples=5,
        seed=123,
    ):
        assert not failures

    assert len(module1.recorded_cotangent_coords) == 5
    assert module1.recorded_cotangent_coords == module2.recorded_cotangent_coords


class NonzeroLinearModule(ModuleType):
    """Module where y = W @ x with non-zero, distinct Jacobian elements.

    This ensures that testing catches any incorrect coordinate alignment
    or zero-filling bug in the checker.
    """

    # Non-zero matrix of shape (6, 4)
    W = np.arange(1, 25, dtype=np.float32).reshape(6, 4)

    def __init__(self, *args, correct_gradients: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.correct_gradients = correct_gradients

    class InputSchema(BaseModel):
        x: Differentiable[Array[(4,), Float32]]

    class OutputSchema(BaseModel):
        y: Differentiable[Array[(6,), Float32]]

    def apply(self, inputs: InputSchema) -> OutputSchema:
        x = np.asarray(inputs.x, dtype=np.float32)
        return {"y": self.W @ x}

    def vector_jacobian_product(
        self,
        inputs: InputSchema,
        vjp_inputs: set[str],
        vjp_outputs: set[str],
        cotangent_vector: dict[str, Any],
    ):
        cot = np.asarray(cotangent_vector["y"], dtype=np.float32)
        if self.correct_gradients:
            grad_x = cot @ self.W
        else:
            grad_x = (cot @ self.W) + 50.0
        return {"x": grad_x}


def test_vjp_output_sampling_coordinate_alignment():
    """Finite differences and VJP values must be compared at exactly the same sampled coordinates.

    With a non-zero linear Jacobian, zero-filling unsampled positions would cause
    finite differences (which evaluate the full row with non-zero values) to mismatch
    the fake zeros. This test verifies that coordinate extraction is correctly aligned.
    """
    module = NonzeroLinearModule("nonzero_linear_module", correct_gradients=True)
    linear_inputs = {"x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}

    num_evals_total = 0
    for _endpoint, failures, num_evals in check_gradients(
        module,
        {"inputs": linear_inputs},
        base_dir=None,
        input_paths=["x"],
        output_paths=["y"],
        endpoints=["vector_jacobian_product"],
        max_evals=4,
        max_output_samples=2,
        seed=0,
    ):
        num_evals_total += num_evals
        assert not failures

    assert num_evals_total > 0


def test_vjp_output_sampling_wrong_gradients_fail():
    """Sampling must not mask real gradient errors: incorrect VJPs must still be caught."""
    bad_module = NonzeroLinearModule("nonzero_linear_module", correct_gradients=False)
    linear_inputs = {"x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}

    failures_found = []
    for _endpoint, failures, _num_evals in check_gradients(
        bad_module,
        {"inputs": linear_inputs},
        base_dir=None,
        input_paths=["x"],
        output_paths=["y"],
        endpoints=["vector_jacobian_product"],
        max_evals=4,
        max_output_samples=3,
        seed=0,
    ):
        failures_found.extend(failures)

    assert len(failures_found) > 0
    for failure in failures_found:
        assert failure.exception is None
        assert failure.ref_val is not None
        assert failure.grad_val is not None
        assert len(failure.ref_val) == 3
        assert len(failure.grad_val) == 3


def test_vjp_output_sampling_scalar_output():
    """Scalar outputs must produce exactly one VJP call with coordinate () regardless of cap."""
    module = CountingVjpModule("dummy_module", correct_gradients=True)
    num_evals_total = 0
    for _endpoint, failures, num_evals in check_gradients(
        module,
        {"inputs": input_data},
        base_dir=None,
        input_paths=["in_scalar"],
        output_paths=["out_scalar"],
        endpoints=["vector_jacobian_product"],
        max_evals=5,
        max_output_samples=10,
        seed=0,
    ):
        num_evals_total += num_evals
        assert not failures

    assert num_evals_total > 0
    assert module.vjp_calls == 1


@pytest.mark.parametrize("invalid_sample_count", [0, -1, -10])
def test_check_gradients_rejects_invalid_max_output_samples(invalid_sample_count):
    """max_output_samples <= 0 must be rejected with ValueError."""
    module = DummyModule("dummy_module", correct_gradients=True)
    with pytest.raises(ValueError, match="max_output_samples must be greater than 0"):
        list(
            check_gradients(
                module,
                {"inputs": input_data},
                endpoints=["vector_jacobian_product"],
                max_output_samples=invalid_sample_count,
            )
        )


def test_cli_max_output_samples_option(cli_runner):
    """CLI must expose --max-output-samples in check-gradients help."""
    from tesseract_core.runtime.cli import app

    result = cli_runner.invoke(app, ["check-gradients", "--help"])
    assert result.exit_code == 0
    assert "--max-output-samples" in result.stdout


class LargeOutputModule(ModuleType):
    """Module with 256x256 (65,536) elements to test asymptotic call count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vjp_calls = 0

    class InputSchema(BaseModel):
        x: Differentiable[Array[(2,), Float32]]

    class OutputSchema(BaseModel):
        y: Differentiable[Array[(256, 256), Float32]]

    def apply(self, inputs: InputSchema) -> OutputSchema:
        return {"y": np.zeros((256, 256), dtype=np.float32)}

    def vector_jacobian_product(
        self,
        inputs: InputSchema,
        vjp_inputs: set[str],
        vjp_outputs: set[str],
        cotangent_vector: dict[str, Any],
    ):
        self.vjp_calls += 1
        return {"x": np.zeros((2,), dtype=np.float32)}


def test_vjp_output_sampling_large_output_asymptotics():
    """65,536-element output runs exactly 10 VJP calls when max_output_samples=10."""
    module = LargeOutputModule("large_output_module")
    for _endpoint, failures, _num_evals in check_gradients(
        module,
        {"inputs": {"x": np.zeros(2, dtype=np.float32)}},
        base_dir=None,
        input_paths=["x"],
        output_paths=["y"],
        endpoints=["vector_jacobian_product"],
        max_evals=1,
        max_output_samples=10,
        seed=0,
    ):
        assert not failures

    assert module.vjp_calls == 10
