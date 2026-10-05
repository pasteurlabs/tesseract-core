# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Return on-disk arrays through ordinary ``Array`` fields without reading them back.

A Tesseract that writes arrays to disk in binref format during ``apply``, for
example to keep a solver's peak memory bounded, can return them by assigning a
:class:`BinrefArray` to an ordinary :class:`~tesseract_core.runtime.Array` field.
When the client negotiates ``json+binref`` output, the field forwards the on-disk
buffer verbatim. For any other format it loads the buffer and encodes it like a
normal array. Because the field is a plain ``Array``, this also works with
``Differentiable[Array[...]]``.

Use :meth:`BinrefArray.write` or :meth:`BinrefArray.from_file` for single arrays,
and :class:`BinrefWriter` to pack many small arrays into a few shared buffers::

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
from typing import Any
from uuid import uuid4

import numpy as np

from tesseract_core.runtime.array_encoding import (
    ALLOWED_DTYPE_NAMES,
    MAX_BINREF_BUFFER_SIZE,
    ArrayDict,
    ArrayLike,
    ShapeType,
    check_shape_dtype_no_cast,
    dump_binref_arraydict,
    load_binref_arraydict,
)
from tesseract_core.runtime.config import get_config
from tesseract_core.runtime.file_interactions import is_url


def _resolve_in_output_path(path: str | Path) -> Path:
    """Resolve ``path`` against ``output_path`` and reject paths that leave it.

    Clients refuse to read binref buffers outside the served ``output_path``, so
    checking here surfaces the error in the Tesseract that produced the path.
    """
    # A drive or root makes the path absolute for our purposes, including
    # Windows paths like "/etc" that are rooted but not ``is_absolute()``.
    if is_url(path) or Path(path).anchor:
        raise ValueError(
            f"Binref path {str(path)!r} must be relative to the output path, "
            "not an absolute path or URL."
        )
    output_path = Path(get_config().output_path)
    full_path = (output_path / path).resolve()
    if not full_path.is_relative_to(output_path):
        raise ValueError(
            f"Binref path {str(path)!r} resolves outside the output path "
            f"({output_path})."
        )
    return full_path


class BinrefArray:
    """An array whose data lives on disk in binref format, referenced by path.

    Construct one with :meth:`write` (write a NumPy array to disk) or
    :meth:`from_file` (reference a buffer that other code, such as a compiled
    solver, already wrote). The plain constructor raises.

    Example::

        def apply(inputs):
            run_solver(out="mesh.bin")  # writes into the output directory
            arr = BinrefArray.from_file("mesh.bin", shape=(1000, 1000), dtype="float64")
            return OutputSchema(result=arr)

    The buffer path is relative to the served ``output_path`` and must resolve
    inside it. The bytes are forwarded unchecked, so they must be C-contiguous,
    row-major, and consistent with the declared ``shape`` and ``dtype``.

    This is a plain class rather than a dataclass or ``BaseModel`` so that the
    runtime's Python-mode ``model_dump()`` treats it as an opaque leaf instead of
    flattening it.
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
            path: Buffer path, relative to the served ``output_path``. The file
                must exist inside it and be large enough to hold the array.
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
        if dtype not in ALLOWED_DTYPE_NAMES:
            raise ValueError(
                f"BinrefArray dtype '{dtype}' is not supported; must be one of: "
                f"{', '.join(ALLOWED_DTYPE_NAMES)}"
            )
        if compression is not None and compressed_size is None:
            raise ValueError("compressed_size is required when compression is set")

        full_path = _resolve_in_output_path(spec)
        if not full_path.is_file():
            raise ValueError(f"BinrefArray buffer {full_path} does not exist")
        if compression is None:
            num_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        else:
            num_bytes = int(compressed_size)
        file_size = full_path.stat().st_size
        if file_size < offset + num_bytes:
            raise ValueError(
                f"BinrefArray buffer {full_path} is too small: expected {num_bytes} "
                f"bytes at offset {offset}, but the file is only {file_size} bytes."
            )

        data: dict[str, Any] = {"encoding": "binref"}
        if compression is not None:
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
        compression: str | None = None,
    ) -> BinrefArray:
        """Write ``arr`` to a new binref buffer on disk and reference it.

        Each call writes a separate file into the configured ``output_path``. Use
        :class:`BinrefWriter` to pack many arrays into a few shared buffers.

        Args:
            arr: The array to write. Coerced to a contiguous NumPy array.
            compression: Optional compression to apply (currently only ``"lz4"``).
        """
        return BinrefWriter(compression=compression).write(arr)

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

        Relative paths are resolved against ``context["base_dir"]`` if given,
        else the configured ``output_path``.
        """
        base_dir = (context or {}).get("base_dir", get_config().output_path)
        return load_binref_arraydict(self._arraydict, base_dir)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        # Loading always produces a fresh array, so ``copy`` can be ignored.
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
    """Validate a :class:`BinrefArray`'s declared shape/dtype without loading it.

    Returns the reference unchanged. It never casts, because casting would mean
    reading and rewriting the buffer.
    """
    check_shape_dtype_no_cast(
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
    """Load a :class:`BinrefArray` for a non-binref encoding, with a warning."""
    nbytes = int(np.prod(val.shape)) * np.dtype(val.dtype).itemsize
    warnings.warn(
        f"A BinrefArray ({nbytes / 1024**2:.1f} MiB) is being read into "
        f"memory because a '{array_encoding}' response cannot reference the "
        "on-disk buffer. Request 'json+binref' output to forward it without "
        "loading it.",
        RuntimeWarning,
        stacklevel=3,
    )
    return val.load(context)


class BinrefWriter:
    """Write many arrays into shared, rotating binref buffers.

    Each :meth:`write` appends to the current ``.bin`` file in the configured
    ``output_path`` and returns a :class:`BinrefArray` pointing at the array's
    slice of it. Once the file exceeds ``max_file_size``, the next write starts a
    new one. Data is written immediately, so there is nothing to flush or close.

    Args:
        max_file_size: Roll over to a new buffer once the current one grows past
            this many bytes. Defaults to the runtime's binref buffer size.
        compression: Optional compression to apply (currently only ``"lz4"``).
    """

    def __init__(
        self,
        *,
        max_file_size: int = MAX_BINREF_BUFFER_SIZE,
        compression: str | None = None,
    ) -> None:
        self._max_file_size = max_file_size
        self._compression = compression
        self._current_uuid = str(uuid4())

    def write(self, arr: ArrayLike) -> BinrefArray:
        """Append ``arr`` to the current buffer and return a reference to it."""
        arraydict, self._current_uuid = dump_binref_arraydict(
            np.ascontiguousarray(arr),
            base_dir=get_config().output_path,
            subdir=None,
            current_binref_uuid=self._current_uuid,
            max_file_size=self._max_file_size,
            compression=self._compression,
        )
        return BinrefArray._from_arraydict(arraydict)
