# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for array encoding and decoding.

Tests the performance of json, base64, and binref encoding methods
for arrays of varying sizes, exercising the same codepaths used by the
runtime server (output_to_bytes for encoding, model_validate_json for
decoding).
"""

from __future__ import annotations

import os
import tempfile

import pytest
from conftest import create_test_array
from pydantic import BaseModel

from tesseract_core.runtime.file_interactions import (
    output_to_bytes,
    supported_format_type,
)
from tesseract_core.runtime.schema_types import Array, Float64


class ArrayModel(BaseModel):
    """Minimal model with a single variable-length float64 array field."""

    data: Array[(None,), Float64]


ENCODINGS = ["json", "base64", "binref"]

# Maps short encoding name to the format string used by output_to_bytes
_ENCODING_TO_FORMAT: dict[str, supported_format_type] = {
    "json": "json",
    "base64": "json+base64",
    "binref": "json+binref",
}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize tests based on --array-sizes."""
    if "encoding_and_size" in metafunc.fixturenames:
        raw = metafunc.config.getoption("--array-sizes", default=None)
        if raw:
            sizes = [int(s.strip()) for s in raw.split(",")]
        else:
            from conftest import DEFAULT_ARRAY_SIZES

            sizes = DEFAULT_ARRAY_SIZES

        params = [(enc, size) for size in sizes for enc in ENCODINGS]
        ids = [f"{enc}_{size:,}" for enc, size in params]
        metafunc.parametrize("encoding_and_size", params, ids=ids)


def _binref_rounds(size: int) -> int:
    """Scale rounds inversely with array size: more rounds for smaller, faster arrays."""
    return max(100, min(int(1e7 / size), 10_000))


def _clear_dir(path: str) -> None:
    """Remove all files in a directory (but not the directory itself).

    Required for binref benchmarks: each round writes a new file with a random
    UUID name. Without clearing, hundreds of MBs of stale files accumulate
    (e.g. 500 rounds x 100k arrays = ~380 MB). This can also increase jitter,
    though the effect is not consistently reproducible.
    """
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))


def test_encoding(benchmark, encoding_and_size):
    encoding, size = encoding_and_size
    model = ArrayModel(data=create_test_array(size))
    fmt = _ENCODING_TO_FORMAT[encoding]

    with tempfile.TemporaryDirectory() as tmpdir:
        if encoding == "binref":

            def setup():
                _clear_dir(tmpdir)

            benchmark.pedantic(
                output_to_bytes,
                args=(model, fmt),
                kwargs={"base_dir": tmpdir},
                setup=setup,
                rounds=_binref_rounds(size),
            )
        else:
            benchmark(output_to_bytes, model, fmt)


def test_decoding(benchmark, encoding_and_size):
    encoding, size = encoding_and_size
    model = ArrayModel(data=create_test_array(size))
    fmt = _ENCODING_TO_FORMAT[encoding]

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx: dict[str, str] = {}
        if encoding == "binref":
            ctx["base_dir"] = tmpdir

        encoded = output_to_bytes(model, fmt, base_dir=tmpdir)

        if encoding == "binref":
            # binref filenames are random UUIDs, so we must re-encode in setup
            # and pass the fresh payload to the decode call via a mutable wrapper.
            payload = [encoded]

            def setup():
                _clear_dir(tmpdir)
                payload[0] = output_to_bytes(model, fmt, base_dir=tmpdir)

            def decode():
                ArrayModel.model_validate_json(payload[0], context=ctx)

            benchmark.pedantic(decode, setup=setup, rounds=_binref_rounds(size))
        else:
            benchmark(ArrayModel.model_validate_json, encoded, context=ctx)


def test_roundtrip(benchmark, encoding_and_size):
    encoding, size = encoding_and_size
    model = ArrayModel(data=create_test_array(size))
    fmt = _ENCODING_TO_FORMAT[encoding]

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx: dict[str, str] = {}
        if encoding == "binref":
            ctx["base_dir"] = tmpdir

        def roundtrip():
            enc = output_to_bytes(model, fmt, base_dir=tmpdir)
            ArrayModel.model_validate_json(enc, context=ctx)

        if encoding == "binref":

            def setup():
                _clear_dir(tmpdir)

            benchmark.pedantic(roundtrip, setup=setup, rounds=_binref_rounds(size))
        else:
            benchmark(roundtrip)
