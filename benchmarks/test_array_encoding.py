# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for array encoding and decoding.

Times an encode + decode roundtrip of a single array for each encoding,
exercising the same codepaths used by the runtime server (output_to_bytes for
encoding, model_validate_json for decoding). The roundtrip is what a request
pays, so a regression in either half shows up here.

Compression is only benchmarked as base64+lz4, since lz4 runs the same code
whichever format wraps it.
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


ENCODINGS = ["json", "base64", "binref", "base64+lz4"]

# Maps short encoding name to the format string used by output_to_bytes
_ENCODING_TO_FORMAT: dict[str, supported_format_type] = {
    "json": "json",
    "base64": "json+base64",
    "binref": "json+binref",
    "base64+lz4": "json+base64",
}

# Maps short encoding name to extra kwargs passed to output_to_bytes
_ENCODING_TO_KWARGS: dict[str, dict] = {
    "json": {},
    "base64": {},
    "binref": {},
    "base64+lz4": {"compression": "lz4"},
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
    return max(20, min(int(1e8 / max(size, 1)), 1000))


def _clear_dir(path: str) -> None:
    """Remove all files in a directory (but not the directory itself).

    Required for binref benchmarks: each round writes a new file with a random
    UUID name. Without clearing, hundreds of MBs of stale files accumulate
    (e.g. 500 rounds x 100k arrays = ~380 MB). This can also increase jitter,
    though the effect is not consistently reproducible.
    """
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))


def test_roundtrip(benchmark, encoding_and_size):
    encoding, size = encoding_and_size
    model = ArrayModel(data=create_test_array(size))
    fmt = _ENCODING_TO_FORMAT[encoding]
    extra_kwargs = _ENCODING_TO_KWARGS[encoding]
    uses_binref = "binref" in encoding

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx: dict[str, str] = {}
        if uses_binref:
            ctx["base_dir"] = tmpdir

        def roundtrip():
            enc = output_to_bytes(model, fmt, base_dir=tmpdir, **extra_kwargs)
            ArrayModel.model_validate_json(enc, context=ctx)

        if uses_binref:

            def setup():
                _clear_dir(tmpdir)

            benchmark.pedantic(roundtrip, setup=setup, rounds=_binref_rounds(size))
        else:
            benchmark(roundtrip)
