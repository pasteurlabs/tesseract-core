# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Return on-disk arrays through ordinary ``Array`` fields without reading them back.

Some Tesseracts write arrays to disk in binref format *during* ``apply`` -- for
example a solver that streams results to disk to keep peak memory bounded. To
return such a buffer without reading it back into memory, hand a
:class:`BinrefArray` to an ordinary :class:`~tesseract_core.runtime.Array` field:
the field forwards the on-disk buffer verbatim when the client negotiates
``json+binref`` output, and loads + re-encodes it for any other format. Because
the field is a plain ``Array``, ``Differentiable[Array[...]]`` composes with this
out of the box.

For a single array, :meth:`BinrefArray.write` (write a NumPy array) or
:meth:`BinrefArray.from_file` (reference a buffer some other code wrote) is all
you need. :class:`BinrefWriter` covers the case where a Tesseract emits *many*
small arrays and you want them packed into a few shared, rotating buffers rather
than one file each::

    from tesseract_core.runtime import Array, Float64
    from tesseract_core.runtime.experimental import BinrefWriter


    class OutputSchema(BaseModel):
        chunks: list[Array[(None,), Float64]]


    def apply(inputs):
        writer = BinrefWriter()
        chunks = [writer.write(a) for a in produce_chunks(inputs)]
        return OutputSchema(chunks=chunks)
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_args
from uuid import uuid4

import numpy as np

from tesseract_core.runtime.array_encoding import (
    MAX_BINREF_BUFFER_SIZE,
    AllowedDtypes,
    ArrayDict,
    ArrayLike,
    ShapeType,
    _check_uncast_shape_dtype,
    _dump_binref_arraydict,
    _load_binref_arraydict,
)
from tesseract_core.runtime.config import get_config


class BinrefArray:
    """An array whose data lives on disk in binref format, referenced by path.

    Hand one to an ordinary :class:`~tesseract_core.runtime.Array` field in
    place of a NumPy array and the field forwards it verbatim when the client
    negotiates ``json+binref`` output -- so a buffer written to disk during
    ``apply`` reaches the client
    without ever being read back into memory. For any other negotiated format
    (``json``, ``base64``) the buffer is loaded once and encoded like a normal
    array. This mirrors how the ``Array`` type already passes GPU arrays through
    validation untouched and materializes them only when needed (see
    :func:`~tesseract_core.runtime.array_encoding.validate_python_or_gpu_array`
    and :func:`~tesseract_core.runtime.array_encoding.encode_array`).

    Construct one via a named constructor (the plain constructor raises):

    * :meth:`write` -- write a NumPy array to disk and reference it.
    * :meth:`from_file` -- reference a buffer some other code already wrote (e.g.
      a compiled solver).

    Example (compiled code wrote ``mesh.bin`` itself)::

        def apply(inputs):
            run_solver(out="mesh.bin")  # writes into the output directory
            arr = BinrefArray.from_file("mesh.bin", shape=(1000, 1000), dtype="float64")
            return OutputSchema(result=arr)

    The buffer path is resolved by the client against the served
    ``output_path``, so it must be relative to that directory (a bare filename is
    resolved as ``output_path / filename``) or an absolute path / URL the
    decoder can reach. The data must be C-contiguous, row-major and match the
    declared ``shape`` and ``dtype``; a mismatch is caught when the field is
    validated.

    Deliberately a plain class (not a dataclass / ``BaseModel``) so Pydantic
    treats it as an opaque leaf and does not flatten it during the Python-mode
    ``model_dump()`` the runtime performs before serialization.
    """

    __slots__ = ("_arraydict",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "BinrefArray cannot be instantiated directly; use one of the named "
            "constructors: BinrefArray.write(arr) or "
            "BinrefArray.from_file(path, shape, dtype)."
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        shape: Sequence[int],
        dtype: str,
        *,
        offset: int = 0,
        compression: str | None = None,
        compressed_size: int | None = None,
    ) -> BinrefArray:
        """Reference a buffer already on disk (e.g. written by compiled code).

        Args:
            path: Buffer path, relative to the served ``output_path`` (or an
                absolute path / URL).
            shape: Shape of the array.
            dtype: NumPy dtype name (e.g. ``"float64"``).
            offset: Byte offset of the array within the file.
            compression: Compression applied to the buffer, if any (``"lz4"``).
            compressed_size: Number of compressed bytes; required when
                ``compression`` is set so the reader knows how much to read.
        """
        spec = str(path)
        if not spec:
            raise ValueError("BinrefArray path must be non-empty")
        allowed_dtypes = [d.lower() for d in get_args(AllowedDtypes)]
        if dtype not in allowed_dtypes:
            raise ValueError(
                f"BinrefArray dtype '{dtype}' is not supported; must be one of: "
                f"{', '.join(allowed_dtypes)}"
            )

        data: dict[str, Any] = {"encoding": "binref"}
        if compression is not None:
            if compressed_size is None:
                raise ValueError("compressed_size is required when compression is set")
            data["buffer"] = f"{spec}:{int(offset)}:{int(compressed_size)}"
            data["compression"] = compression
        elif offset:
            data["buffer"] = f"{spec}:{int(offset)}"
        else:
            data["buffer"] = spec
        return cls._from_arraydict(
            {
                "object_type": "array",
                "shape": [int(s) for s in shape],
                "dtype": dtype,
                "data": data,
            }
        )

    @classmethod
    def write(
        cls,
        arr: ArrayLike,
        *,
        output_dir: str | Path | None = None,
        compression: str | None = None,
    ) -> BinrefArray:
        """Write ``arr`` to its own binref buffer on disk and reference it.

        The buffer uses the same on-disk layout as the built-in binref encoder,
        so the result is indistinguishable from a normally-encoded array. Each
        call writes an independent file; to pack many arrays into a few shared,
        rotating buffers use :class:`BinrefWriter`.

        Args:
            arr: The array to write. Coerced to a contiguous NumPy array.
            output_dir: Directory to write into. Defaults to the configured
                ``output_path`` (see class docs on path resolution).
            compression: Optional compression to apply (currently only ``"lz4"``).
        """
        return BinrefWriter(output_dir, compression=compression).write(arr)

    @classmethod
    def _from_arraydict(cls, arraydict: ArrayDict) -> BinrefArray:
        """Wrap a pre-built binref ``ArrayDict`` (internal / writer use)."""
        obj = cls.__new__(cls)
        obj._arraydict = arraydict
        return obj

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the referenced array."""
        return tuple(self._arraydict["shape"])

    @property
    def dtype(self) -> str:
        """The dtype name of the referenced array."""
        return self._arraydict["dtype"]

    @property
    def buffer(self) -> str:
        """The ``<path>[:<offset>[:<compressed_size>]]`` buffer spec."""
        return self._arraydict["data"]["buffer"]

    def to_arraydict(self) -> ArrayDict:
        """The binref ``ArrayDict`` this reference serializes to."""
        return self._arraydict

    def load(self, context: dict[str, Any] | None = None) -> np.ndarray:
        """Read the referenced buffer into a NumPy array.

        This is the one operation that actually touches the buffer bytes; the
        rest of the reference stays lazy. Paths are resolved against the
        ``base_dir`` in ``context`` if given, else the configured
        ``output_path``.
        """
        base_dir = (context or {}).get("base_dir", get_config().output_path)
        return _load_binref_arraydict(self._arraydict, base_dir)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        # Lets NumPy (and any array-like consumer) materialize the reference on
        # demand via ``np.asarray(ref)``. Loading always produces a fresh array,
        # so ``copy`` needs no handling beyond being accepted.
        arr = self.load()
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    def __repr__(self) -> str:
        return (
            f"BinrefArray(buffer={self.buffer!r}, shape={self.shape!r}, "
            f"dtype={self.dtype!r})"
        )


def validate_binref_array(
    val: BinrefArray, expected_shape: ShapeType, expected_dtype: str | None
) -> BinrefArray:
    """Validate a :class:`BinrefArray`'s shape/dtype without reading the buffer.

    Returns the reference unchanged so it can later be forwarded verbatim (see
    :func:`~tesseract_core.runtime.array_encoding.encode_array`). Only the
    reference's declared ``shape``/``dtype`` are inspected -- no file is opened.
    Never casts: a cast would require reading and rewriting the buffer,
    defeating the point of a passthrough.
    """
    _check_uncast_shape_dtype(
        val.shape,
        val.dtype,
        expected_shape,
        expected_dtype,
        no_cast_reason="a passthrough binref is not cast",
    )
    return val


def load_for_inline_encoding(
    val: BinrefArray, array_encoding: str, context: dict[str, Any]
) -> np.ndarray:
    """Load a :class:`BinrefArray` so it can be serialized inline.

    Any encoding other than binref must inline the data, so the on-disk buffer
    has to be read into memory -- the exact cost a ``BinrefArray`` exists to
    avoid. Warn loudly (the values are still correct) so this is not a silent
    memory blow-up.
    """
    nbytes = int(np.prod(val.shape)) * np.dtype(val.dtype).itemsize
    warnings.warn(
        f"A BinrefArray ({nbytes / 1024**2:.1f} MiB) is being read into "
        f"memory to satisfy a '{array_encoding}' response; the on-disk buffer "
        "cannot be forwarded for this encoding. Request 'json+binref' output "
        "to stream it from disk without loading it.",
        RuntimeWarning,
        stacklevel=3,
    )
    return val.load(context)


class BinrefWriter:
    """Write many arrays into shared, rotating binref buffers.

    Each :meth:`write` appends to the current ``.bin`` file and returns a
    :class:`BinrefArray` referencing the array's slice of it, rolling over to a
    fresh file once the current one exceeds ``max_file_size``. This packs many
    small arrays into a few files -- the same layout the runtime's own binref
    serializer produces -- instead of the one-file-per-array behaviour of
    :meth:`BinrefArray.write`.

    The writer keeps only the current buffer's identifier between calls and holds
    no array data, so there is nothing to flush or close.

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
