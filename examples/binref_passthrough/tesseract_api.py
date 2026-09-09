# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward arrays written to disk during apply, without loading them back.

This Tesseract writes its results to disk in binref format *during* ``apply``
(standing in for a solver that streams results to disk to bound peak memory) and
returns lightweight references. With ``json+binref`` output the on-disk bytes
flow straight to the client, never round-tripped through memory on the server.

The output fields are ordinary ``Array`` types. An ``Array`` accepts a
``BinrefArray`` (a reference to an on-disk buffer) in place of a NumPy array and
forwards it verbatim for ``json+binref`` output, loading + re-encoding it only
for other formats. Because the fields are plain ``Array`` types,
``Differentiable[Array[...]]`` composes with the passthrough out of the box.

``BinrefArray`` can be built three ways:

* ``BinrefArray.write(arr)`` -- write a NumPy array to its own buffer.
* ``BinrefArray.from_file(path, shape, dtype)`` -- reference a buffer some other
  code (e.g. a compiled solver) already wrote.
* ``BinrefWriter`` -- pack many arrays into a few shared, rotating buffers.
"""

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.experimental import BinrefArray, BinrefWriter


class InputSchema(BaseModel):
    n: int = Field(description="Length of each array to generate.", default=8)
    scale: float = Field(
        description="Value to scale the generated arrays by.", default=1.0
    )
    parts: int = Field(description="How many array chunks to emit.", default=3)


class OutputSchema(BaseModel):
    result: Array[(None,), Float64] = Field(
        description="A single array forwarded as a binref (written via BinrefArray.write)."
    )
    # A plain Array field accepts a binref reference, so it composes with
    # Differentiable for free.
    grad: Differentiable[Array[(None,), Float64]] = Field(
        description="A differentiable array, also forwarded as a binref."
    )
    chunks: list[Array[(None,), Float64]] = Field(
        description="Several arrays packed into shared buffers (written via BinrefWriter)."
    )


def apply(inputs: InputSchema) -> OutputSchema:
    # A real component would produce these buffers as a side effect of its solve.
    # One-off: write a single array to its own file.
    result = BinrefArray.write(np.arange(inputs.n, dtype=np.float64) * inputs.scale)
    grad = BinrefArray.write(np.ones(inputs.n, dtype=np.float64) * inputs.scale)

    # Many small arrays: pack them into a few shared, rotating buffers rather
    # than one file each.
    with BinrefWriter() as writer:
        chunks = [
            writer.write(np.full(inputs.n, i, dtype=np.float64))
            for i in range(inputs.parts)
        ]

    # No np.load / np.frombuffer here: the bytes on disk are forwarded as-is.
    return OutputSchema(result=result, grad=grad, chunks=chunks)
