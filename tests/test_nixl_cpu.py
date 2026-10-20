# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the nixl device transport.

These cover the wire schema, the ``gpu_transport`` gating, the transport
registration, and how encode, decode and release drive a NIXL agent, using a fake
agent and the ``mocked_cuda`` runtime instead of a GPU and NIXL. The GPU tests in
``test_nixl.py`` cover real transfers.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from tesseract_core.runtime import nixl_transport


class FakeCudaArray:
    """A GPU array's metadata surface without any device memory."""

    def __init__(self, shape, typestr="<f4", data_ptr=0x1000):
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": typestr,
            "data": (data_ptr, False),
            "strides": None,
            "version": 3,
        }


class FakeRegistration:
    def __init__(self, descs):
        self.descs = descs

    def trim(self):
        return ("xfer", tuple(self.descs))


class FakeNixlAgent:
    """Records how the transport drives a NIXL agent.

    Every remote agent it loads is named ``"producer"``, and loading it again
    while it is known fails, as in NIXL.
    """

    name = "local"

    def __init__(self, final_state="DONE"):
        self.final_state = final_state
        self.registered = []
        self.deregistered = []
        self.remote_agents = []
        self.removed_agents = []
        self.known_agents = set()
        self.xfers = []
        self.released_xfers = []

    def register_memory(self, descs, mem_type):
        assert mem_type == "VRAM"
        registration = FakeRegistration(descs)
        self.registered.append(registration)
        return registration

    def deregister_memory(self, registration):
        self.deregistered.append(registration)

    def get_agent_metadata(self):
        return b"agent-metadata"

    def add_remote_agent(self, metadata):
        if "producer" in self.known_agents:
            raise RuntimeError("NIXL_ERR_NOT_ALLOWED")
        self.known_agents.add("producer")
        self.remote_agents.append(metadata)
        # NIXL returns the name as bytes.
        return b"producer"

    def remove_remote_agent(self, name):
        self.known_agents.remove(name)
        self.removed_agents.append(name)

    def get_xfer_descs(self, descs, mem_type):
        assert mem_type == "VRAM"
        return ("remote", tuple(descs))

    def initialize_xfer(self, op, local, remote, peer, notif):
        handle = object()
        self.xfers.append((handle, op, local, remote, peer))
        return handle

    def transfer(self, handle):
        pass

    def check_xfer_state(self, handle):
        return self.final_state

    def release_xfer_handle(self, handle):
        self.released_xfers.append(handle)


@pytest.fixture
def fake_agent(monkeypatch):
    agent = FakeNixlAgent()
    monkeypatch.setattr(nixl_transport, "_NIXL_AGENT", agent)
    return agent


def test_nixl_registered_as_transport():
    """The nixl backend is discoverable by name and advertises cross-host reach."""
    from tesseract_core.runtime.device_transport import DeviceTransport, get_transport

    transport = get_transport("nixl")
    assert transport.name == "nixl"
    assert transport.reach == "both"
    assert isinstance(transport, DeviceTransport)


@pytest.mark.parametrize("buffer", ["tesseract-12-ab3f:YWJj:139872:0", ":::"])
def test_nixl_array_schema_accepts_valid_descriptors(buffer):
    from tesseract_core.runtime.array_encoding import NixlArrayData

    # <agent_name>:<agent_meta_b64>:<address>:<device>, or ":::" for an empty array
    data = NixlArrayData(buffer=buffer, encoding="nixl")
    assert data.encoding == "nixl"


@pytest.mark.parametrize(
    "bad",
    [
        "notpacked",
        "YWJj:12:0",
        "p:YWJj:12",
        "p:YWJj:ZGVm:0",
        "p:YWJj:12:x",
        ":YWJj:12:0",
        "p::12:0",
        "::",
        "a b:YWJj:1:0",
    ],
)
def test_nixl_array_schema_rejects_malformed(bad):
    from pydantic import ValidationError

    from tesseract_core.runtime.array_encoding import NixlArrayData

    with pytest.raises(ValidationError):
        NixlArrayData(buffer=bad, encoding="nixl")


def test_nixl_gated_behind_gpu_transport_config():
    """Nixl is a transport, not a format: it is offered only when configured."""
    from tesseract_core.runtime import config
    from tesseract_core.runtime.file_interactions import available_gpu_transports

    config.update_config(gpu_transport="none")
    assert "nixl" not in available_gpu_transports()

    config.update_config(gpu_transport="nixl")
    assert "nixl" in available_gpu_transports()


def test_output_to_bytes_rejects_nixl_by_default():
    from tesseract_core.runtime import config, file_interactions

    config.update_config(gpu_transport="none")
    with pytest.raises(ValueError, match=r"Unsupported GPU transport nixl"):
        file_interactions.output_to_bytes({"y": 1}, "json+base64", gpu_transport="nixl")


def test_export_group_is_falsy_until_it_holds_an_export(mocked_cuda, fake_agent):
    """A server keeps a response's session only while it holds exports."""
    from tesseract_core.runtime.device_transport import get_transport

    transport = get_transport("nixl")
    group = transport.new_exports()
    assert not group

    transport.register(FakeCudaArray((4,)), group)
    assert group

    transport.release(group)
    assert not group


def test_register_requires_a_session(mocked_cuda, fake_agent):
    from tesseract_core.runtime.device_transport import get_transport

    with pytest.raises(ValueError, match="new_exports"):
        get_transport("nixl").register(FakeCudaArray((4,)))


def test_dump_registers_syncs_and_keeps_the_array(mocked_cuda, fake_agent):
    import pybase64

    arr = FakeCudaArray((2, 3), data_ptr=0x4000)
    group = nixl_transport.NixlExportGroup()
    out = nixl_transport.dump_nixl_arraydict(arr, group)

    assert out["shape"] == [2, 3]
    assert out["dtype"] == "float32"
    assert out["data"]["encoding"] == "nixl"
    name, meta_b64, address, device = out["data"]["buffer"].split(":")
    assert name == "local"
    assert pybase64.b64decode(meta_b64) == b"agent-metadata"
    assert (int(address), int(device)) == (0x4000, 0)

    # The array's own bytes are registered, and the array is kept until release.
    [registration] = fake_agent.registered
    assert registration.descs == [(0x4000, 24, 0, "")]
    assert group.exports == [(registration, arr)]
    # Producer kernels must finish before another process reads the memory.
    assert mocked_cuda.calls["sync"]

    group.release()
    assert fake_agent.deregistered == [registration]
    assert not group.exports


def test_empty_array_crosses_without_nixl(mocked_cuda, monkeypatch):
    """An empty array has no memory to register, so NIXL is never touched."""

    def no_agent():
        raise AssertionError("an empty array must not create a NIXL agent")

    monkeypatch.setattr(nixl_transport, "_get_nixl_agent", no_agent)

    group = nixl_transport.NixlExportGroup()
    out = nixl_transport.dump_nixl_arraydict(FakeCudaArray((0, 3), data_ptr=0), group)
    assert out["data"]["buffer"] == ":::"
    assert not group

    arr = nixl_transport.load_nixl_arraydict(out)
    assert arr.shape == (0, 3)
    assert arr.nbytes == 0


def test_empty_descriptor_for_a_non_empty_array_is_rejected(mocked_cuda):
    val = {
        "object_type": "array",
        "shape": [3],
        "dtype": "float32",
        "data": {"buffer": ":::", "encoding": "nixl"},
    }
    with pytest.raises(ValueError, match="descriptor is empty"):
        nixl_transport.load_nixl_arraydict(val)


def test_load_reads_onto_the_active_device(mocked_cuda, fake_agent):
    """The consumer reads into its own buffer on its active device and cleans up."""
    import pybase64

    mocked_cuda.current_device = 2
    val = {
        "object_type": "array",
        "shape": [4],
        "dtype": "float64",
        "data": {
            "buffer": f"producer:{pybase64.b64encode_as_string(b'meta')}:{0x7000}:3",
            "encoding": "nixl",
        },
    }
    arr = nixl_transport.load_nixl_arraydict(val)

    assert arr.device == 2
    assert arr.shape == (4,)
    assert arr.dtype == np.float64
    assert mocked_cuda.calls["malloc"] == [32]
    assert fake_agent.remote_agents == [b"meta"]

    [(handle, op, _local, remote, peer)] = fake_agent.xfers
    assert op == "READ"
    # The producer's buffer, on the producer's device, with the array's size.
    assert remote == ("remote", ((0x7000, 32, 3),))
    assert peer == "producer"
    # The transfer handle, the local registration and the producer are all
    # released.
    assert fake_agent.released_xfers == [handle]
    assert fake_agent.deregistered == fake_agent.registered
    assert fake_agent.removed_agents == ["producer"]


def test_decoding_never_unpickles(mocked_cuda, fake_agent, monkeypatch):
    """A message from another process must not be able to run code here."""
    import pickle

    import pybase64

    def refuse(*args, **kwargs):
        raise AssertionError("decoding a nixl array must not unpickle")

    monkeypatch.setattr(pickle, "loads", refuse)
    monkeypatch.setattr(fake_agent, "deserialize_descs", refuse, raising=False)
    val = {
        "object_type": "array",
        "shape": [2],
        "dtype": "float32",
        "data": {
            "buffer": f"producer:{pybase64.b64encode_as_string(b'meta')}:{0x7000}:0",
            "encoding": "nixl",
        },
    }
    nixl_transport.load_nixl_arraydict(val)


def test_failed_transfer_frees_the_buffer(mocked_cuda, fake_agent):
    import pybase64

    fake_agent.final_state = "ERR"
    val = {
        "object_type": "array",
        "shape": [4],
        "dtype": "float32",
        "data": {
            "buffer": f"producer:{pybase64.b64encode_as_string(b'meta')}:{0x7000}:3",
            "encoding": "nixl",
        },
    }
    with pytest.raises(RuntimeError, match="nixl transfer failed"):
        nixl_transport.load_nixl_arraydict(val)

    assert len(mocked_cuda.calls["free"]) == 1
    assert len(fake_agent.released_xfers) == 1
    assert fake_agent.deregistered == fake_agent.registered
    assert fake_agent.removed_agents == ["producer"]


def test_concurrent_reads_from_one_producer_take_turns(fake_agent):
    """A second read from a producer loads its metadata only after the first ends."""
    second_loaded = threading.Event()

    def second_read():
        with nixl_transport._remote_agent(fake_agent, "producer", b"meta-2"):
            second_loaded.set()

    with nixl_transport._remote_agent(fake_agent, "producer", b"meta-1"):
        thread = threading.Thread(target=second_read)
        thread.start()
        assert not second_loaded.wait(timeout=0.2)
        assert fake_agent.remote_agents == [b"meta-1"]
    thread.join(timeout=5)

    assert second_loaded.is_set()
    assert fake_agent.remote_agents == [b"meta-1", b"meta-2"]
    assert fake_agent.removed_agents == ["producer", "producer"]
    assert not nixl_transport._PRODUCER_LOCKS
    assert not nixl_transport._PRODUCER_READS


def test_metadata_must_describe_the_named_producer(fake_agent):
    with (
        pytest.raises(ValueError, match="describes 'producer'"),
        nixl_transport._remote_agent(fake_agent, "someone-else", b"meta"),
    ):
        pass
    assert fake_agent.removed_agents == ["producer"]
    assert not nixl_transport._PRODUCER_LOCKS


def test_missing_nixl_points_at_the_extra(monkeypatch):
    def no_nixl():
        raise ModuleNotFoundError("No module named 'nixl'", name="nixl")

    monkeypatch.setattr(nixl_transport, "_NIXL_AGENT", None)
    monkeypatch.setattr(nixl_transport, "_import_nixl_api", no_nixl)

    with pytest.raises(RuntimeError, match=r"tesseract-core\[nixl\]"):
        nixl_transport._get_nixl_agent()
