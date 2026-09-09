# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.experimental import BinrefArray, BinrefWriter
from tesseract_core.runtime.schema_types import is_differentiable


class OneDModel(BaseModel):
    result: Array[(4,), Float64]


def _dump(ref, base_dir, encoding="binref"):
    return OneDModel(result=ref).model_dump(
        mode="json", context={"array_encoding": encoding, "base_dir": base_dir}
    )["result"]


def _decode_b64(arraydict):
    import pybase64

    buf = pybase64.b64decode(arraydict["data"]["buffer"])
    return np.frombuffer(buf, dtype=arraydict["dtype"]).reshape(arraydict["shape"])


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_write_writes_a_buffer(tmp_path):
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    ref = BinrefArray.write(arr, output_dir=tmp_path)

    assert isinstance(ref, BinrefArray)
    assert ref.shape == (2, 3)
    assert ref.dtype == "float64"
    binfiles = list(tmp_path.glob("*.bin"))
    assert len(binfiles) == 1
    # The bytes on disk are exactly the array's bytes.
    assert binfiles[0].read_bytes() == arr.tobytes()


def test_from_file_references_external_buffer(tmp_path):
    # Stand in for compiled code writing the buffer itself, no tesseract API.
    arr = np.arange(4, dtype=np.float64)
    (tmp_path / "ext.bin").write_bytes(arr.tobytes())

    ref = BinrefArray.from_file("ext.bin", shape=(4,), dtype="float64")
    assert ref.buffer == "ext.bin"
    np.testing.assert_array_equal(_decode_b64(_dump(ref, tmp_path, "base64")), arr)


def test_from_file_encodes_offset_and_compression():
    ref = BinrefArray.from_file("f.bin", shape=(4,), dtype="float64", offset=32)
    assert ref.buffer == "f.bin:32"

    ref = BinrefArray.from_file(
        "f.bin",
        shape=(4,),
        dtype="float64",
        offset=8,
        compression="lz4",
        compressed_size=20,
    )
    assert ref.buffer == "f.bin:8:20"
    assert ref.to_arraydict()["data"]["compression"] == "lz4"


def test_from_spec_takes_a_buffer_spec():
    ref = BinrefArray.from_spec("f.bin:0", (4,), "float64")
    assert ref.buffer == "f.bin:0"
    assert ref.shape == (4,)
    assert ref.dtype == "float64"


def test_direct_instantiation_raises():
    with pytest.raises(RuntimeError, match="named constructors"):
        BinrefArray("f.bin:0", (4,), "float64")
    with pytest.raises(RuntimeError, match="named constructors"):
        BinrefArray()


def test_from_spec_rejects_bad_inputs():
    with pytest.raises(ValueError, match="dtype"):
        BinrefArray.from_spec("f.bin", (4,), "float128")
    with pytest.raises(ValueError, match="non-empty"):
        BinrefArray.from_spec("", (4,), "float64")


def test_from_file_requires_compressed_size_when_compressed():
    with pytest.raises(ValueError, match="compressed_size"):
        BinrefArray.from_file("f.bin", (4,), "float64", compression="lz4")


# --------------------------------------------------------------------------- #
# Array-field behaviour
# --------------------------------------------------------------------------- #


def test_array_field_forwards_ref_verbatim(tmp_path):
    ref = BinrefArray.write(np.arange(4, dtype=np.float64), output_dir=tmp_path)
    n_before = len(list(tmp_path.glob("*.bin")))

    dumped = _dump(ref, tmp_path, "binref")
    # Emits a genuine binref dict pointing at the existing file; writes nothing new.
    assert dumped["data"]["encoding"] == "binref"
    assert dumped["shape"] == [4]
    assert dumped["data"]["buffer"] == ref.buffer
    assert len(list(tmp_path.glob("*.bin"))) == n_before


@pytest.mark.parametrize("encoding", ["json", "base64"])
def test_non_binref_output_loads_and_reencodes(tmp_path, encoding):
    arr = np.arange(4, dtype=np.float64) * 1.5
    ref = BinrefArray.write(arr, output_dir=tmp_path)

    dumped = _dump(ref, tmp_path, encoding)
    assert dumped["data"]["encoding"] == encoding
    assert dumped["shape"] == [4] and dumped["dtype"] == "float64"
    if encoding == "json":
        assert dumped["data"]["buffer"] == arr.tolist()


def test_python_roundtrip_preserves_reference(tmp_path):
    """The apply model_dump()/model_validate() round-trip must not load or flatten."""
    ref = BinrefArray.write(np.arange(4, dtype=np.float64), output_dir=tmp_path)

    dumped = OneDModel(result=ref).model_dump()  # python mode
    assert isinstance(dumped["result"], BinrefArray)

    revalidated = OneDModel.model_validate(dumped)
    assert isinstance(revalidated.result, BinrefArray)
    assert revalidated.result.buffer == ref.buffer


def test_array_field_still_accepts_plain_arrays_and_dicts():
    class M(BaseModel):
        result: Array[(None,), Float64]

    np.testing.assert_array_equal(
        M.model_validate({"result": [1.0, 2.0, 3.0]}).result, [1.0, 2.0, 3.0]
    )
    encoded = {
        "object_type": "array",
        "shape": [3],
        "dtype": "float64",
        "data": {"buffer": [1.0, 2.0, 3.0], "encoding": "json"},
    }
    np.testing.assert_array_equal(
        M.model_validate({"result": encoded}).result, [1.0, 2.0, 3.0]
    )


def test_ref_shape_mismatch_is_rejected(tmp_path):
    ref = BinrefArray.write(np.arange(3, dtype=np.float64), output_dir=tmp_path)
    with pytest.raises(ValidationError, match="shape"):
        OneDModel(result=ref)  # field expects shape (4,)


def test_ref_dtype_mismatch_is_rejected(tmp_path):
    ref = BinrefArray.write(np.arange(4, dtype=np.float32), output_dir=tmp_path)
    with pytest.raises(ValidationError, match="dtype"):
        OneDModel(result=ref)  # field expects float64


def test_output_matches_builtin_array_encoding(tmp_path):
    """A non-binref dump must be byte-identical to a normal Array of same data."""
    arr = np.arange(5, dtype=np.float64) * 2

    class M(BaseModel):
        result: Array[(5,), Float64]

    ref = BinrefArray.write(arr, output_dir=tmp_path)
    via_ref = M(result=ref).model_dump(
        mode="json", context={"array_encoding": "base64", "base_dir": tmp_path}
    )["result"]
    via_builtin = M(result=arr).model_dump(
        mode="json", context={"array_encoding": "base64"}
    )["result"]
    assert via_ref == via_builtin


def test_differentiable_array_accepts_ref(tmp_path):
    """The headline: Differentiable[Array[...]] composes with a binref ref."""

    class DiffModel(BaseModel):
        grad: Differentiable[Array[(4,), Float64]]

    assert is_differentiable(Differentiable[Array[(4,), Float64]])

    ref = BinrefArray.write(np.arange(4, dtype=np.float64), output_dir=tmp_path)
    model = DiffModel(grad=ref)
    assert isinstance(model.grad, BinrefArray)

    dumped = model.model_dump(
        mode="json", context={"array_encoding": "binref", "base_dir": tmp_path}
    )["grad"]
    assert dumped["data"]["encoding"] == "binref"
    assert dumped["data"]["buffer"] == ref.buffer


def test_compression_roundtrips(tmp_path):
    arr = np.arange(16, dtype=np.float64)
    ref = BinrefArray.write(arr, output_dir=tmp_path, compression="lz4")

    class M(BaseModel):
        result: Array[(16,), Float64]

    dumped = M(result=ref).model_dump(
        mode="json", context={"array_encoding": "binref", "base_dir": tmp_path}
    )["result"]
    assert dumped["data"]["compression"] == "lz4"

    via_json = M(result=ref).model_dump(
        mode="json", context={"array_encoding": "json", "base_dir": tmp_path}
    )["result"]
    np.testing.assert_array_equal(via_json["data"]["buffer"], arr.tolist())


def test_load_and_array_protocol(tmp_path):
    from tesseract_core.runtime.config import get_config, update_config

    update_config(output_path=str(tmp_path))
    output_path = get_config().output_path  # resolved (symlinks) form

    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    ref = BinrefArray.write(arr, output_dir=output_path)

    np.testing.assert_array_equal(ref.load(), arr)
    np.testing.assert_array_equal(np.asarray(ref), arr)
    assert np.asarray(ref).dtype == np.float64


# --------------------------------------------------------------------------- #
# BinrefWriter (packing)
# --------------------------------------------------------------------------- #


def test_write_writes_one_file_per_call(tmp_path):
    for _ in range(3):
        BinrefArray.write(np.arange(4, dtype=np.float64), output_dir=tmp_path)
    assert len(list(tmp_path.glob("*.bin"))) == 3


def test_writer_packs_into_one_buffer(tmp_path):
    arrays = [np.arange(4, dtype=np.float64) + i for i in range(5)]
    with BinrefWriter(tmp_path) as w:
        refs = [w.write(a) for a in arrays]

    assert all(isinstance(r, BinrefArray) for r in refs)
    binfiles = list(tmp_path.glob("*.bin"))
    assert len(binfiles) == 1
    offsets = [int(r.buffer.split(":")[1]) for r in refs]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0

    for ref, arr in zip(refs, arrays, strict=True):
        dumped = _dump(ref, tmp_path, "json")
        np.testing.assert_array_equal(dumped["data"]["buffer"], arr.tolist())


def test_writer_rotates_at_max_file_size(tmp_path):
    arr = np.arange(4, dtype=np.float64)  # 32 bytes
    with BinrefWriter(tmp_path, max_file_size=20) as w:
        refs = [w.write(arr) for _ in range(3)]

    binfiles = {r.buffer.split(":")[0] for r in refs}
    assert len(binfiles) > 1
    assert len(list(tmp_path.glob("*.bin"))) == len(binfiles)


def test_writer_and_write_produce_equivalent_data(tmp_path):
    arr = np.arange(6, dtype=np.float64) * 3

    class M(BaseModel):
        result: Array[(6,), Float64]

    ref_single = BinrefArray.write(arr, output_dir=tmp_path)
    with BinrefWriter(tmp_path) as w:
        ref_packed = w.write(arr)

    for ref in (ref_single, ref_packed):
        dumped = M(result=ref).model_dump(
            mode="json", context={"array_encoding": "base64", "base_dir": tmp_path}
        )["result"]
        np.testing.assert_array_equal(_decode_b64(dumped), arr)
