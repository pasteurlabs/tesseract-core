# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for array encoding and decoding via pydantic model serde.

Tests the performance of json, base64, and binref encoding methods
for arrays of varying sizes, exercising the full pydantic serialization
and validation codepath (model_dump_json / model_validate_json) rather
than calling encode_array / decode_array directly.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from conftest import create_test_array
from pydantic import BaseModel

from tesseract_core.runtime.schema_types import Array, Float64


class ArrayModel(BaseModel):
    """Minimal model with a single variable-length float64 array field."""

    data: Array[(None,), Float64]


ENCODINGS = ["json", "base64", "binref"]


def _encodings_for_size(size: int) -> list[str]:
    """Return applicable encodings for a given array size."""
    if size > 20_000_000:
        return [e for e in ENCODINGS if e != "json"]
    return ENCODINGS


def _encoding_params(sizes: list[int]) -> list[tuple[str, int]]:
    """Generate (encoding, size) pairs, skipping json for very large arrays."""
    return [(enc, size) for size in sizes for enc in _encodings_for_size(size)]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize tests based on --array-sizes."""
    if "encoding_and_size" in metafunc.fixturenames:
        raw = metafunc.config.getoption("--array-sizes", default=None)
        if raw:
            sizes = [int(s.strip()) for s in raw.split(",")]
        else:
            from conftest import DEFAULT_ARRAY_SIZES

            sizes = DEFAULT_ARRAY_SIZES

        params = _encoding_params(sizes)
        ids = [f"{enc}_{size:,}" for enc, size in params]
        metafunc.parametrize("encoding_and_size", params, ids=ids)


def _binref_rounds(size: int) -> int:
    """Scale rounds inversely with array size: more rounds for smaller, faster arrays."""
    if size <= 1_000:
        return 10_000
    if size <= 100_000:
        return 1000
    return 100


def _clear_dir(path: str) -> None:
    """Remove all files in a directory (but not the directory itself)."""
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))


def test_encoding(benchmark, encoding_and_size):
    encoding, size = encoding_and_size
    model = ArrayModel(data=create_test_array(size))

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = {"array_encoding": encoding}
        if encoding == "binref":
            ctx["base_dir"] = tmpdir

            def setup():
                _clear_dir(tmpdir)

            benchmark.pedantic(
                model.model_dump_json,
                kwargs={"context": ctx},
                setup=setup,
                rounds=_binref_rounds(size),
            )
        else:
            benchmark(model.model_dump_json, context=ctx)


def test_decoding(benchmark, encoding_and_size):
    encoding, size = encoding_and_size
    model = ArrayModel(data=create_test_array(size))

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx: dict[str, str] = {}
        if encoding == "binref":
            ctx["base_dir"] = tmpdir

        encoded = model.model_dump_json(context={"array_encoding": encoding, **ctx})

        if encoding == "binref":
            # binref filenames are random UUIDs, so we must re-encode in setup
            # and pass the fresh payload to the decode call via a mutable wrapper.
            payload = [encoded]

            def setup():
                _clear_dir(tmpdir)
                payload[0] = model.model_dump_json(
                    context={"array_encoding": encoding, **ctx}
                )

            def decode():
                ArrayModel.model_validate_json(payload[0], context=ctx)

            benchmark.pedantic(decode, setup=setup, rounds=_binref_rounds(size))
        else:
            benchmark(ArrayModel.model_validate_json, encoded, context=ctx)


def test_roundtrip(benchmark, encoding_and_size):
    encoding, size = encoding_and_size
    model = ArrayModel(data=create_test_array(size))

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx: dict[str, str] = {}
        if encoding == "binref":
            ctx["base_dir"] = tmpdir

        def roundtrip():
            enc = model.model_dump_json(context={"array_encoding": encoding, **ctx})
            ArrayModel.model_validate_json(enc, context=ctx)

        if encoding == "binref":

            def setup():
                _clear_dir(tmpdir)

            benchmark.pedantic(roundtrip, setup=setup, rounds=_binref_rounds(size))
        else:
            benchmark(roundtrip)
