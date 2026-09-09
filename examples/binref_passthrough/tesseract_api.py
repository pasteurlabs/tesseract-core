# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Tesseract that forwards an already-written binref into its output verbatim.

The component writes an array to disk in binref format *during* ``apply`` (this
stands in for a real solver that streams results to disk) and returns a
lightweight reference to it. The reference type serialises to the exact same
binref ``ArrayDict`` the built-in ``Array`` type produces, but never loads the
buffer back into memory -- so the on-disk bytes flow straight through to the
client.

The serializer is *format-aware*: it honours the ``array_encoding`` the client
negotiated via the ``Accept`` header. When the client asks for ``json+binref``
the reference is forwarded untouched (zero-copy). For ``json`` / ``base64`` /
``cuda_ipc`` it loads the buffer once and delegates to the built-in
``encode_array`` so the output is byte-identical to a normal ``Array`` field.

See the ``BinrefRef`` docstring for how the passthrough survives the
``model_dump()`` / ``model_validate()`` round-trip the runtime performs before
serialisation.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

from tesseract_core.runtime.array_encoding import (
    ArrayDict,
    _load_binref_arraydict,
    encode_array,
)
from tesseract_core.runtime.config import get_config

# --------------------------------------------------------------------------- #
# The passthrough type
# --------------------------------------------------------------------------- #


class BinrefRef:
    """A reference to an already-written binref buffer on disk.

    Deliberately a plain class (not a dataclass / BaseModel) so Pydantic treats
    it as an opaque leaf and does not introspect it into a dict during the
    Python-mode ``model_dump()`` the runtime performs in ``runtime.core.apply``.
    That opacity is what lets the reference survive the round-trip intact instead
    of being flattened before re-validation.

    ``buffer`` follows the binref grammar ``<path>[:<offset>]`` and is resolved
    by the client against the served ``output_path`` (the mounted volume), so it
    must be *relative to that directory*. Here we write into ``output_path``
    directly and emit a bare filename.
    """

    __slots__ = ("buffer", "dtype", "shape")

    def __init__(self, buffer: str, shape: tuple[int, ...], dtype: str) -> None:
        self.buffer = buffer
        self.shape = tuple(shape)
        self.dtype = dtype

    def to_arraydict(self) -> ArrayDict:
        return {
            "object_type": "array",
            "shape": list(self.shape),
            "dtype": self.dtype,
            "data": {"buffer": self.buffer, "encoding": "binref"},
        }

    @classmethod
    def write(cls, arr: np.ndarray, output_dir: Path) -> BinrefRef:
        """Write ``arr`` to a fresh .bin file in ``output_dir`` and reference it.

        Stands in for a component that produces the buffer as a side effect of
        its real computation.
        """
        arr = np.ascontiguousarray(arr)
        filename = f"{uuid.uuid4()}.bin"
        (output_dir / filename).write_bytes(arr.tobytes())
        return cls(buffer=filename, shape=arr.shape, dtype=arr.dtype.name)


def _load_ref(val: BinrefRef, context: dict[str, Any]) -> np.ndarray:
    """Load the referenced buffer into a NumPy array (the one place we load).

    The buffer path is relative to the served ``output_path``; that same
    directory is where binref serialisation resolves relative paths from, so we
    reuse the runtime's own loader with it as ``base_dir``.
    """
    base_dir = context.get("base_dir", get_config().output_path)
    return _load_binref_arraydict(val.to_arraydict(), base_dir)


class _BinrefRefAnnotation:
    """Pydantic wiring: validate-passthrough + format-aware serialisation."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def _validate(val: Any, info: core_schema.ValidationInfo) -> BinrefRef:
            if isinstance(val, BinrefRef):
                return val  # passthrough: no load, no coercion
            raise ValueError(
                f"BinrefArray expects a BinrefRef, got {type(val).__name__}"
            )

        def _serialize(val: BinrefRef, info: core_schema.SerializationInfo) -> Any:
            # Python mode: pass the ref through untouched so it survives the
            # apply model_dump()/model_validate() round-trip.
            if not info.mode_is_json():
                return val

            context = info.context if info.context else {}
            array_encoding = context.get("array_encoding", "json")

            # The whole point: for binref output, forward the reference verbatim
            # without ever reading the buffer back into memory.
            if array_encoding == "binref":
                return val.to_arraydict()

            # Any other negotiated format (json, base64, cuda_ipc): load the
            # buffer once and hand it to the built-in encoder, so the field
            # behaves exactly like a normal Array of the same shape/dtype.
            # (cuda_ipc will raise from encode_array -- a host binref genuinely
            # can't be exported as an IPC handle without a device copy.)
            arr = _load_ref(val, context)
            return encode_array(
                arr, info, expected_shape=arr.shape, expected_dtype=None
            )

        return core_schema.with_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize, info_arg=True
            ),
        )


BinrefArray = Annotated[BinrefRef, _BinrefRefAnnotation]


# --------------------------------------------------------------------------- #
# The Tesseract
# --------------------------------------------------------------------------- #


class InputSchema(BaseModel):
    n: int = Field(description="Length of the array to generate.", default=8)
    scale: float = Field(
        description="Value to scale the generated array by.", default=1.0
    )


class OutputSchema(BaseModel):
    # Forwarded verbatim from disk, never loaded back into memory.
    result: BinrefArray = Field(
        description="An array produced on disk during apply and forwarded as a binref."
    )


def apply(inputs: InputSchema) -> OutputSchema:
    output_dir = Path(get_config().output_path)

    # A real component would produce this buffer as a side effect of its solve.
    # Here we synthesise it and write it straight to disk in binref layout.
    arr = np.arange(inputs.n, dtype=np.float64) * inputs.scale
    ref = BinrefRef.write(arr, output_dir)

    # No np.load / np.frombuffer here: the bytes on disk are forwarded as-is.
    return OutputSchema(result=ref)
