# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Ref, the sidecar-JSON encoding for nested models."""

import json

import numpy as np
import pytest
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float32, Float64
from tesseract_core.runtime.experimental import LazySequence, Ref
from tesseract_core.runtime.file_interactions import output_to_bytes
from tesseract_core.runtime.schema_generation import (
    create_abstract_eval_schema,
    create_apply_schema,
    create_gradient_schema,
    get_all_model_path_patterns,
)
from tesseract_core.runtime.schema_types import is_differentiable


class Frame(BaseModel):
    name: str
    displacement: Differentiable[Array[(None, 3), Float32]]
    pressure: Differentiable[Array[(None,), Float64]]

    def __ref_name__(self) -> str:
        return self.name


class InputSchema(BaseModel):
    scale: Differentiable[Array[(), Float32]]


class OutputSchema(BaseModel):
    result: list[Ref[Frame]]


def make_frame(idx: int, n: int = 2) -> Frame:
    return Frame(
        name=f"frame_{idx}",
        displacement=np.full((n, 3), float(idx), dtype="float32"),
        pressure=np.arange(n, dtype="float64") + idx,
    )


def make_output(count: int = 3) -> OutputSchema:
    return OutputSchema(result=[make_frame(i) for i in range(count)])


def test_binref_writes_sidecars_and_emits_paths(tmp_path):
    payload = json.loads(
        output_to_bytes(make_output(), "json+binref", base_dir=tmp_path)
    )
    assert payload == {"result": ["frame_0.json", "frame_1.json", "frame_2.json"]}

    sidecar = json.loads((tmp_path / "frame_1.json").read_text())
    assert sidecar["name"] == "frame_1"
    assert sidecar["displacement"]["data"]["encoding"] == "binref"
    assert sidecar["displacement"]["shape"] == [2, 3]
    assert sidecar["pressure"]["dtype"] == "float64"


def test_binref_arrays_share_one_buffer_across_sidecars(tmp_path):
    """The binref uuid accumulator must be threaded through refs, not reset per file."""
    output_to_bytes(make_output(), "json+binref", base_dir=tmp_path)
    assert len(list(tmp_path.glob("*.bin"))) == 1


def test_binref_dir_keeps_sidecars_next_to_buffers(tmp_path):
    payload = json.loads(
        output_to_bytes(
            make_output(1), "json+binref", base_dir=tmp_path, binref_dir="sub"
        )
    )
    assert payload == {"result": ["sub/frame_0.json"]}
    assert (tmp_path / "sub" / "frame_0.json").exists()

    sidecar = json.loads((tmp_path / "sub" / "frame_0.json").read_text())
    # binrefs inside the sidecar stay relative to base_dir, not to the sidecar
    assert sidecar["displacement"]["data"]["buffer"].startswith("sub/")


@pytest.mark.parametrize("fmt", ["json", "json+base64"])
def test_inline_when_no_base_dir(fmt):
    """Formats without an output directory must keep refs inline, not fail."""
    payload = json.loads(output_to_bytes(make_output(1), fmt))
    (frame,) = payload["result"]
    assert frame["name"] == "frame_0"
    expected_encoding = "base64" if fmt == "json+base64" else "json"
    assert frame["displacement"]["data"]["encoding"] == expected_encoding


def test_roundtrip_through_sidecars(tmp_path):
    original = make_output()
    payload = json.loads(output_to_bytes(original, "json+binref", base_dir=tmp_path))
    context = {"base_dir": str(tmp_path)}
    restored = OutputSchema.model_validate(payload, context=context)

    assert [f.name for f in restored.result] == ["frame_0", "frame_1", "frame_2"]
    for got, want in zip(restored.result, original.result, strict=True):
        np.testing.assert_allclose(got.displacement, want.displacement)
        np.testing.assert_allclose(got.pressure, want.pressure)


def test_validate_accepts_inline_objects():
    restored = OutputSchema.model_validate(
        {
            "result": [
                {"name": "a", "displacement": [[1.0, 2.0, 3.0]], "pressure": [1.0]}
            ]
        }
    )
    assert restored.result[0].displacement.shape == (1, 3)


def test_sidecar_arrays_are_shape_validated(tmp_path):
    """Arrays behind a ref go through the same validation as inline arrays."""
    (tmp_path / "bad.json").write_text(
        json.dumps({"name": "bad", "displacement": [[1.0, 2.0]], "pressure": [1.0]})
    )
    with pytest.raises(ValueError, match="shape"):
        OutputSchema.model_validate(
            {"result": ["bad.json"]}, context={"base_dir": str(tmp_path)}
        )


def test_path_without_base_dir_is_rejected():
    with pytest.raises(ValueError, match="no base_dir is set"):
        OutputSchema.model_validate({"result": ["frame_0.json"]})


def test_duplicate_filenames_are_rejected(tmp_path):
    duplicated = OutputSchema(result=[make_frame(0), make_frame(0)])
    with pytest.raises(Exception, match="Duplicate Ref filename"):
        output_to_bytes(duplicated, "json+binref", base_dir=tmp_path)


def test_ref_rejects_extra_type_parameters():
    with pytest.raises(ValueError, match="single parameter"):
        Ref[Frame, "name"]


def test_unsafe_filename_is_rejected(tmp_path):
    escaping = OutputSchema(result=[make_frame(0)])
    escaping.result[0].name = "../escape"
    with pytest.raises(Exception, match="as a Ref filename"):
        output_to_bytes(escaping, "json+binref", base_dir=tmp_path)


def test_uuid_names_by_default(tmp_path):
    class Anon(BaseModel):
        """Same fields as Frame but no __ref_name__, so sidecars get UUID names."""

        name: str
        displacement: Differentiable[Array[(None, 3), Float32]]
        pressure: Differentiable[Array[(None,), Float64]]

    class AnonOutput(BaseModel):
        result: list[Ref[Anon]]

    anon = Anon(**make_frame(0).model_dump())
    payload = json.loads(
        output_to_bytes(AnonOutput(result=[anon]), "json+binref", base_dir=tmp_path)
    )
    (name,) = payload["result"]
    assert name.endswith(".json") and name != "frame_0.json"


# ---------------------------------------------------------------------------
# Ref must stay transparent to schema generation
# ---------------------------------------------------------------------------


def test_diffable_paths_match_unwrapped_model():
    class Unwrapped(BaseModel):
        result: list[Frame]

    assert get_all_model_path_patterns(
        OutputSchema, is_differentiable
    ) == get_all_model_path_patterns(Unwrapped, is_differentiable)


def test_abstract_eval_replaces_arrays_inside_refs():
    _, AbstractOutput = create_abstract_eval_schema(InputSchema, OutputSchema)
    schema = AbstractOutput.model_json_schema()
    frame = schema["$defs"]["AbstractEval_Frame"]["properties"]
    assert frame["displacement"] == {"$ref": "#/$defs/ShapeDType"}
    assert frame["pressure"] == {"$ref": "#/$defs/ShapeDType"}
    assert frame["name"]["type"] == "string"


def test_apply_schema_rebuild_propagates_model_config():
    """The inner model must be rebuilt (Apply_Frame, extra=forbid), not captured stale."""
    _, ApplyOutput = create_apply_schema(InputSchema, OutputSchema)
    valid = {
        "result": [{"name": "a", "displacement": [[1.0, 2.0, 3.0]], "pressure": [1.0]}]
    }
    assert (
        type(ApplyOutput.model_validate(valid).root.result[0]).__name__ == "Apply_Frame"
    )

    with pytest.raises(ValueError, match=r"[Ee]xtra"):
        ApplyOutput.model_validate(
            {"result": [{**valid["result"][0], "unexpected": 1}]}
        )


@pytest.mark.parametrize("gradient_type", ["jacobian", "jvp", "vjp"])
def test_gradient_endpoints_see_through_refs(gradient_type):
    """Gradient payloads are flat path->array dicts, so Ref is inherently a no-op there."""
    GradInput, GradOutput = create_gradient_schema(
        InputSchema, OutputSchema, gradient_type
    )
    out_schema = GradOutput.model_json_schema()
    # Flat mapping of path -> encoded array (or path -> path -> array for jacobian).
    # Ref's string/path alternative appears nowhere in it, so gradients never
    # traverse a ref and need no special handling.
    assert '"format": "path"' not in json.dumps(out_schema)

    if gradient_type == "jacobian":
        patterns = GradInput.model_json_schema()["properties"]["jac_outputs"]["items"]
        assert {p["pattern"] for p in patterns["anyOf"]} == {
            r"result\.\[\d+\]\.displacement",
            r"result\.\[\d+\]\.pressure",
        }


def test_composes_with_lazy_sequence():
    class Composed(BaseModel):
        result: LazySequence[Ref[Frame]]

    restored = Composed.model_validate(
        {
            "result": [
                {"name": "a", "displacement": [[1.0, 2.0, 3.0]], "pressure": [1.0]}
            ]
        }
    )
    assert restored.result[0].name == "a"
