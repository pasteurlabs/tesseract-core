# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the json+nixl encoding's plumbing.

These cover the Python wiring around the NIXL transport -- the wire schema, the
experimental-flag gating, the format -> encoding-context mapping, and the
transport registration -- without a GPU or NIXL installed. The actual
cross-process transfer (which needs both) is covered by the GPU tests in
``test_nixl.py``.
"""

from __future__ import annotations

import pytest


def test_nixl_registered_as_transport():
    """The nixl backend is discoverable by name and advertises cross-host reach."""
    from tesseract_core.runtime.device_transport import DeviceTransport, get_transport

    transport = get_transport("nixl")
    assert transport.name == "nixl"
    assert transport.reach == "both"
    assert isinstance(transport, DeviceTransport)


def test_nixl_array_schema_roundtrips_a_valid_descriptor():
    from tesseract_core.runtime.array_encoding import NixlArrayData

    # <agent_meta_b64>:<descs_b64>:<device>
    data = NixlArrayData(buffer="YWJj:ZGVm:0", encoding="nixl")
    assert data.encoding == "nixl"


@pytest.mark.parametrize("bad", ["notpacked", "a:b", "a:b:c", ":ZGVm:0", "YWJj:ZGVm:x"])
def test_nixl_array_schema_rejects_malformed(bad):
    from pydantic import ValidationError

    from tesseract_core.runtime.array_encoding import NixlArrayData

    with pytest.raises(ValidationError):
        NixlArrayData(buffer=bad, encoding="nixl")


def test_json_nixl_gated_behind_experimental_flag():
    from tesseract_core.runtime import config
    from tesseract_core.runtime.file_interactions import available_formats

    config.update_config(enable_experimental_cuda_nixl=False)
    assert "json+nixl" not in available_formats()

    config.update_config(enable_experimental_cuda_nixl=True)
    assert "json+nixl" in available_formats()


def test_output_to_bytes_rejects_nixl_by_default():
    from tesseract_core.runtime import config, file_interactions

    config.update_config(enable_experimental_cuda_nixl=False)
    with pytest.raises(ValueError, match=r"Unsupported format json\+nixl"):
        file_interactions.output_to_bytes({"y": 1}, "json+nixl")


def test_output_to_bytes_nixl_context(monkeypatch):
    """json+nixl maps to the nixl array-encoding context (flag enabled)."""
    from tesseract_core.runtime import config, file_interactions

    config.update_config(enable_experimental_cuda_nixl=True)
    captured = {}

    class FakeAdapter:
        def __init__(self, _type):
            pass

        def dump_python(self, obj, mode, context, exclude_unset):
            captured["context"] = context
            return {}

    monkeypatch.setattr(file_interactions, "TypeAdapter", FakeAdapter)
    monkeypatch.setattr(file_interactions.orjson, "dumps", lambda d: b"{}")

    file_interactions.output_to_bytes({"y": 1}, "json+nixl")
    assert captured["context"] == {"array_encoding": "nixl"}


def test_client_import_device_transport_explains_missing_nixl(monkeypatch):
    """Without the nixl extra, using json+nixl points at the required extras."""
    import builtins

    from tesseract_core.sdk import tesseract as sdk

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tesseract_core.runtime.nixl_transport" or (
            name == "tesseract_core.runtime" and "nixl_transport" in (fromlist or ())
        ):
            raise ImportError("No module named 'nixl'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"tesseract-core\[runtime,nixl\]"):
        sdk._import_device_transport("nixl")
