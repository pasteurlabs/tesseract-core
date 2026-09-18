# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack many on-disk arrays into a few shared binref buffers.

Some Tesseracts write arrays to disk in binref format *during* ``apply`` -- for
example a solver that streams results to disk to keep peak memory bounded. To
return such a buffer without reading it back into memory, hand a
:class:`~tesseract_core.runtime.array_encoding.BinrefArray` to an ordinary
:class:`~tesseract_core.runtime.Array` field: the field forwards the on-disk
buffer verbatim when the client negotiates ``json+binref`` output, and loads +
re-encodes it for any other format. Because the field is a plain ``Array``,
``Differentiable[Array[...]]`` composes with this out of the box.

For a single array, :meth:`BinrefArray.write` (write a NumPy array) or
:meth:`BinrefArray.from_file` (reference a buffer some other code wrote) is all
you need. This module adds :class:`BinrefWriter` for the case where a Tesseract
emits *many* small arrays and you want them packed into a few shared, rotating
buffers rather than one file each::

    from tesseract_core.runtime import Array, Float64
    from tesseract_core.runtime.experimental import BinrefWriter


    class OutputSchema(BaseModel):
        chunks: list[Array[(None,), Float64]]


    def apply(inputs):
        with BinrefWriter() as writer:
            chunks = [writer.write(a) for a in produce_chunks(inputs)]
        return OutputSchema(chunks=chunks)
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from tesseract_core.runtime.array_encoding import (
    MAX_BINREF_BUFFER_SIZE,
    ArrayLike,
    BinrefArray,
    _dump_binref_arraydict,
)
from tesseract_core.runtime.config import get_config


class BinrefWriter:
    """Write many arrays into shared, rotating binref buffers.

    Each :meth:`write` appends to the current ``.bin`` file and returns a
    :class:`~tesseract_core.runtime.array_encoding.BinrefArray` referencing the
    array's slice of it, rolling over to a fresh file once the current one
    exceeds ``max_file_size``. This packs many small arrays into a few files --
    the same layout the runtime's own binref serializer produces -- instead of
    the one-file-per-array behaviour of :meth:`BinrefArray.write`.

    The writer keeps only a small buffer identifier between calls, so it holds no
    array data; there is nothing to flush and the context manager is optional
    (it is provided for symmetry and readability).

    Args:
        output_dir: Directory to write buffers into. Defaults to the Tesseract's
            configured ``output_path``. The client resolves buffer paths relative
            to the served ``output_path``, so leave this as the default unless
            the files will end up under that directory.
        max_file_size: Roll over to a new buffer once the current one grows past
            this many bytes. Defaults to the runtime's binref buffer size.
        compression: Optional compression to apply (currently only ``"lz4"``).
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        max_file_size: int = MAX_BINREF_BUFFER_SIZE,
        compression: str | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._max_file_size = max_file_size
        self._compression = compression
        self._current_uuid = str(uuid4())

    def write(self, arr: ArrayLike) -> BinrefArray:
        """Append ``arr`` to the current buffer and return a reference to it."""
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = get_config().output_path
        arr = np.ascontiguousarray(arr)
        # subdir=None -> a path relative to output_dir, which the client resolves
        # as output_path / <path>. _dump_binref_arraydict appends to the current
        # buffer and hands back the (possibly rotated) uuid to reuse next time.
        arraydict, self._current_uuid = _dump_binref_arraydict(
            arr,
            base_dir=output_dir,
            subdir=None,
            current_binref_uuid=self._current_uuid,
            max_file_size=self._max_file_size,
            compression=self._compression,
        )
        return BinrefArray._from_arraydict(arraydict)

    def __enter__(self) -> BinrefWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        return None
