# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import orjson
import pytest
from pydantic import BaseModel, TypeAdapter
from pydantic_core import PydanticSerializationError

from tesseract_core.runtime.experimental import TesseractReference
from tesseract_core.runtime.file_interactions import output_to_bytes


class FakeTesseract:
    """Mock Tesseract instance simulating underlying engine behavior."""

    def __init__(self, kind: str, target: str) -> None:
        self.kind = kind
        self.target = target
        self.served = False

    @classmethod
    def from_url(cls, url: str) -> "FakeTesseract":
        # Simulate internal normalization (e.g. stripping trailing slashes)
        # to prove serialization returns the original ref rather than internal state.
        return cls("url", url.rstrip("/"))

    @classmethod
    def from_image(cls, image: str) -> "FakeTesseract":
        # Simulate internal image resolution
        return cls("image", f"resolved-image://{image}")

    @classmethod
    def from_tesseract_api(cls, path: str) -> "FakeTesseract":
        # Simulate internal path resolution
        return cls("api_path", f"/absolute/resolved/{path}")

    def serve(self) -> None:
        self.served = True

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"received": payload, "kind": self.kind}

    @property
    def custom_attribute(self) -> str:
        return f"custom_{self.kind}"


@pytest.fixture(autouse=True)
def patch_tesseract_class(monkeypatch):
    monkeypatch.setattr(
        TesseractReference,
        "_get_tesseract_class",
        classmethod(lambda cls: FakeTesseract),
    )


def test_type_adapter_round_trip():
    """Validate using TypeAdapter and serialize back to the exact original envelope."""
    envelope = {"type": "url", "ref": "http://solver-b:8000"}
    ta = TypeAdapter(TesseractReference)

    ref = ta.validate_python(envelope)
    dumped_python = ta.dump_python(ref)
    dumped_python_json = ta.dump_python(ref, mode="json")
    dumped_json = json.loads(ta.dump_json(ref))

    assert dumped_python == envelope
    assert dumped_python_json == envelope
    assert dumped_json == envelope


def test_base_model_model_dump():
    """OutputSchema.model_dump() contains the exact reference envelope."""

    class OutputSchema(BaseModel):
        best: TesseractReference

    envelope = {"type": "url", "ref": "http://solver-b:8000"}
    ta = TypeAdapter(TesseractReference)
    ref = ta.validate_python(envelope)

    model = OutputSchema(best=ref)
    dumped = model.model_dump()

    assert dumped == {"best": envelope}


def test_base_model_model_dump_json():
    """OutputSchema.model_dump_json() outputs JSON matching the exact envelope."""

    class OutputSchema(BaseModel):
        best: TesseractReference

    envelope = {"type": "url", "ref": "http://solver-b:8000"}
    ta = TypeAdapter(TesseractReference)
    ref = ta.validate_python(envelope)

    model = OutputSchema(best=ref)
    dumped_json = json.loads(model.model_dump_json())

    assert dumped_json == {"best": envelope}


def test_input_to_output_pass_through():
    """End-to-end pass-through from InputSchema to OutputSchema preserves the envelope."""

    class InputSchema(BaseModel):
        target: TesseractReference

    class OutputSchema(BaseModel):
        target: TesseractReference

    envelope = {"type": "url", "ref": "http://solver-b:8000"}
    inputs = InputSchema.model_validate({"target": envelope})
    output = OutputSchema(target=inputs.target)

    assert output.model_dump() == {"target": envelope}
    assert json.loads(output.model_dump_json()) == {"target": envelope}


def test_runtime_output_to_bytes():
    """Real unmocked output_to_bytes() serializes OutputSchema correctly."""

    class OutputSchema(BaseModel):
        target: TesseractReference

    envelope = {"type": "url", "ref": "http://solver-b:8000"}
    ta = TypeAdapter(TesseractReference)
    ref = ta.validate_python(envelope)
    output = OutputSchema(target=ref)

    raw_bytes = output_to_bytes(output, "json")
    parsed = orjson.loads(raw_bytes)

    assert parsed == {"target": envelope}


def test_delegation_remains_intact():
    """Delegation to underlying Tesseract via __getattr__ functions normally."""
    envelope = {"type": "url", "ref": "http://solver-b:8000"}
    ta = TypeAdapter(TesseractReference)
    ref = ta.validate_python(envelope)

    # Calling method on underlying object
    result = ref.apply({"payload_key": "payload_value"})
    assert result == {
        "received": {"payload_key": "payload_value"},
        "kind": "url",
    }

    # Accessing property on underlying object
    assert ref.custom_attribute == "custom_url"


@pytest.mark.parametrize(
    ("tesseract_type", "raw_ref"),
    [
        ("url", "http://solver-b:8000/"),  # Provenance: trailing slash preserved
        (
            "api_path",
            "./relative/custom/tesseract_api.py",
        ),  # Provenance: relative path preserved
        ("image", "my-registry.io/org/tesseract-solver:v1.2.3"),
    ],
)
def test_all_reference_kinds_and_provenance(tesseract_type: str, raw_ref: str):
    """All reference kinds preserve raw originating metadata and image calls serve()."""
    envelope = {"type": tesseract_type, "ref": raw_ref}
    ta = TypeAdapter(TesseractReference)
    ref = ta.validate_python(envelope)

    if tesseract_type == "image":
        assert ref._tesseract.served is True

    # Assert raw_ref is preserved exactly, not transformed into internal representation
    assert ref._tesseract.target != raw_ref
    dumped = ta.dump_python(ref)
    assert dumped == envelope
    assert json.loads(ta.dump_json(ref)) == envelope


def test_direct_constructor_compatibility():
    """Direct construction TesseractReference(tesseract) is supported, fails gracefully on serialization."""
    underlying = FakeTesseract("direct", "direct-target")
    ref = TesseractReference(underlying)

    # Delegation still works
    assert ref.apply({"step": 1}) == {
        "received": {"step": 1},
        "kind": "direct",
    }
    assert ref.custom_attribute == "custom_direct"

    # Serialization without originating metadata raises explicit error
    ta = TypeAdapter(TesseractReference)
    with pytest.raises(
        PydanticSerializationError, match="originating reference metadata"
    ):
        ta.dump_python(ref)

    with pytest.raises(
        PydanticSerializationError, match="originating reference metadata"
    ):
        ta.dump_json(ref)

    class OutputSchema(BaseModel):
        target: TesseractReference

    out = OutputSchema(target=ref)
    with pytest.raises(
        PydanticSerializationError, match="originating reference metadata"
    ):
        out.model_dump()

    with pytest.raises(
        PydanticSerializationError, match="originating reference metadata"
    ):
        out.model_dump_json()
