# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union, get_args
from uuid import uuid4

import numpy as np
import pybase64
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PositiveInt,
    ValidationInfo,
    create_model,
)

from tesseract_core.runtime.file_interactions import (
    get_filesize,
    is_absolute_path,
    is_url,
    join_paths,
    read_from_path,
    write_to_path,
)

AllowedDtypes = Literal[
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "bool",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "complex64",
    "complex128",
]
EllipsisType = type(Ellipsis)
ArrayLike = Union[np.ndarray, np.number, np.bool_]
ShapeType = Union[tuple[Optional[int], ...], EllipsisType]

MAX_BINREF_BUFFER_SIZE = 100 * 1024 * 1024  # 100 MB


# Base classes for the different array encodings
# The actual models are created dynamically based on the expected shape and dtype by get_array_model


class Base64ArrayData(BaseModel):
    """Data structure for base64 encoded binary buffers."""

    buffer: Annotated[
        str,
        Field(
            description="Base64 encoded binary buffer",
            examples=["<base64 encoded string>"],
        ),
    ]
    encoding: Literal["base64"]
    model_config = ConfigDict(extra="forbid")


class BinrefArrayData(BaseModel):
    """Data structure that dumps array data to binary file."""

    buffer: str = Field(pattern=r"^.+?(\:\d+)?$")
    encoding: Literal["binref"]
    model_config = ConfigDict(extra="forbid")


class JsonArrayData(BaseModel):
    """Data structure for json buffers (list of decimal numbers)."""

    buffer: JsonValue
    encoding: Literal["json"]
    model_config = ConfigDict(extra="forbid")


class EncodedArrayModel(BaseModel):
    """Base class for general encoded arrays.

    Allowed values for shape and dtype are enforced by subclasses.
    """

    object_type: Literal["array"]
    shape: tuple[PositiveInt, ...]
    dtype: AllowedDtypes
    data: Union[BinrefArrayData, Base64ArrayData, JsonArrayData]
    model_config = ConfigDict(extra="forbid")


def get_array_model(
    expected_shape: ShapeType, expected_dtype: Optional[str], flags: Sequence[str]
) -> type[EncodedArrayModel]:
    """Create a Pydantic model for an encoded array that does validation on the given expected shape and dtype."""
    if expected_dtype is None:
        dtype_type = AllowedDtypes
    else:
        # Only allow dtypes that can be cast to the expected dtype
        subdtypes = [
            dtype
            for dtype in get_args(AllowedDtypes)
            if np.can_cast(dtype, expected_dtype, casting="same_kind")
        ]
        dtype_type = Literal[tuple(subdtypes)]

    shape_kwargs = {}

    # Only allow shapes that can be broadcasted to the expected shape
    if expected_shape is Ellipsis:
        # No shape check
        shape_type = tuple[int, ...]
    else:
        # There are 3 cases for each dimension `n`:
        # - n=None: polymorphic dimension, can be any positive int
        # - n=1: fixed dimension, must be 1
        # - n=N: fixed dimension, must be N or 1 (triggers broadcasting to N)
        # Example: expected_shape=(None, 1, 3) -> allowed_vals=tuple[PositiveInt, Literal[1], Literal[1, 3]]
        allowed_vals = []
        for dim in expected_shape:
            if dim is None:
                allowed_vals.append(PositiveInt)
            elif dim == 1:
                allowed_vals.append(Literal[1])
            else:
                allowed_vals.append(Literal[1, dim])

        if not allowed_vals:
            # Scalar -> require empty tuple
            shape_type = tuple[int, ...]

        shape_type = tuple[tuple(allowed_vals)]
        shape_kwargs.update(
            examples=([1 if s is None else s for s in expected_shape],),
            # Dimensionality must match exactly
            min_length=len(expected_shape),
            max_length=len(expected_shape),
            # TODO: This is a hack to allow JSF to parse the JSON schema
            # see https://github.com/ghandic/jsf/issues/118
            json_schema_extra={"items": {"type": "integer"}},
        )

    # Add flags to the model config
    config = EncodedArrayModel.model_config
    config["json_schema_extra"] = {"array_flags": flags}

    fields = {
        "object_type": (
            Literal["array"],
            Field(
                description="Indicates that this dict can be parsed to an array.",
                default="array",
            ),
        ),
        "shape": (
            shape_type,
            Field(
                description="Shape of the array",
                **shape_kwargs,
            ),
        ),
        "dtype": (
            dtype_type,
            Field(
                description="Data type of the array",
                examples=[expected_dtype or "float64"],
            ),
        ),
        # Choose the appropriate data structure based on the encoding
        "data": (
            Union[BinrefArrayData, Base64ArrayData, JsonArrayData],
            Field(discriminator="encoding"),
        ),
        "model_config": (ConfigDict, config),
    }

    if expected_shape is Ellipsis:
        readable_shape = "anyrank"
    elif not expected_shape:
        readable_shape = "scalar"
    else:
        readable_shape = "_".join(
            str(s) if s is not None else "any" for s in expected_shape
        )

    readable_flags = "_".join(flags) if flags else "noflags"

    out = create_model(
        f"EncodedArrayModel__{readable_shape}__{expected_dtype}__{readable_flags}",
        **fields,
        __base__=EncodedArrayModel,
    )
    return out


def _dump_binref_arraydict(
    arr: Union[np.ndarray, np.number, np.bool_],
    base_dir: Union[Path, str],
    current_binref_uuid: str,
    max_file_size: int = MAX_BINREF_BUFFER_SIZE,
) -> tuple[dict[str, Union[str, dict[str, str]]], str]:
    """Dump array to json+binref encoded array dict.

    Writes a .bin file and returns json encoded data.
    """
    target_name = f"{current_binref_uuid}.bin"
    target_path = join_paths(base_dir, target_name)

    current_size = get_filesize(target_path)

    # if the current buffer is too large, use a new one
    if current_size > max_file_size:
        current_size = 0
        current_binref_uuid = str(uuid4())
        target_name = f"{current_binref_uuid}.bin"
        target_path = join_paths(base_dir, target_name)

    write_to_path(arr.tobytes(), target_path, append=True)
    offset = current_size

    data = {"buffer": f"{target_name}:{offset}", "encoding": "binref"}
    arraydict = {
        "object_type": "array",
        "shape": arr.shape,
        "dtype": arr.dtype.name,
        "data": data,
    }
    return arraydict, current_binref_uuid


def _dump_base64_arraydict(
    arr: Union[np.ndarray, np.number, np.bool_],
) -> dict[str, Union[str, dict[str, str]]]:
    """Dump array to json+base64 encoded array dict."""
    data = {
        "buffer": pybase64.b64encode(arr.tobytes()).decode(),
        "encoding": "base64",
    }
    arraydict = {
        "object_type": "array",
        "shape": arr.shape,
        "dtype": arr.dtype.name,
        "data": data,
    }
    return arraydict


def _dump_json_arraydict(
    arr: Union[np.ndarray, np.number, np.bool_],
) -> dict[str, Union[str, dict[str, str]]]:
    """Dump array to json encoded array dict."""
    data = {
        "buffer": arr.tolist(),
        "encoding": "json",
    }
    arraydict = {
        "object_type": "array",
        "shape": arr.shape,
        "dtype": arr.dtype.name,
        "data": data,
    }
    return arraydict


def _load_base64_arraydict(val: dict) -> np.ndarray:
    """Load array from json+base64 encoded array dict."""
    buffer = pybase64.b64decode(val["data"]["buffer"], validate=True)
    return np.frombuffer(buffer, dtype=val["dtype"]).reshape(val["shape"])


def _load_binref_arraydict(val: dict, base_dir: Union[str, Path, None]) -> np.ndarray:
    """Load array from json+binref encoded array dict."""
    path_match = re.match(r"^(?P<path>.+?)(\:(?P<offset>\d+))?$", val["data"]["buffer"])
    bufferpath = path_match.group("path")
    if path_match.group("offset") is None:
        offset = 0
    else:
        offset = int(path_match.group("offset"))

    uses_relative_path = not is_absolute_path(bufferpath) and not is_url(bufferpath)
    if uses_relative_path and base_dir is None:
        raise ValueError(
            "Array data is binref encoded with a relative path but no base_dir is provided. "
            "Invoke the Tesseract via file aliasing or make sure that paths are absolute."
        )

    dtype = np.dtype(val["dtype"])
    shape = val["shape"]
    size = 1 if len(shape) == 0 else np.prod(shape)
    num_bytes = int(size * dtype.itemsize)

    if base_dir is not None:
        bufferpath = join_paths(base_dir, bufferpath)

    buffer = read_from_path(bufferpath, offset=offset, length=num_bytes)
    return np.frombuffer(buffer, dtype=dtype).reshape(shape)


def _coerce_shape_dtype(
    arr: ArrayLike, expected_shape: ShapeType, expected_dtype: Optional[str]
) -> ArrayLike:
    """Coerce the shape and dtype of the passed array to the expected values."""
    if expected_shape is Ellipsis:
        # No shape check
        out_shape = arr.shape
    else:
        if len(arr.shape) != len(expected_shape):
            raise ValueError(
                f"Dimensionality mismatch: {len(arr.shape)}D array cannot be cast to {len(expected_shape)}D"
            )

        out_shape = tuple(
            # Polymorphic dims -> keep the passed shape
            arr.shape[i] if expected_shape[i] is None else expected_shape[i]
            for i in range(len(expected_shape))
        )

    # Broadcast the arr to the expected shape and dtype
    try:
        arr = np.broadcast_to(arr, out_shape)
    except ValueError:
        raise ValueError(
            f"Shape mismatch: {arr.shape} cannot be cast to {out_shape}"
        ) from None

    if expected_dtype is not None:
        if not np.can_cast(arr.dtype, expected_dtype, casting="same_kind"):
            raise ValueError(
                f"Dtype mismatch: {arr.dtype} cannot be cast to {expected_dtype}"
            )
        arr = arr.astype(expected_dtype)

    allowed_dtypes = [dtype.lower() for dtype in get_args(AllowedDtypes)]
    if arr.dtype.name not in allowed_dtypes:
        raise ValueError(
            f"Got invalid dtype: {arr.dtype.name}, expected one of {allowed_dtypes}"
        )

    if not out_shape:
        # Cast to a scalar type
        return arr.dtype.type(arr)

    return arr


def python_to_array(
    val: Any, expected_shape: ShapeType, expected_dtype: Optional[str]
) -> ArrayLike:
    """Convert a Python object to a NumPy array."""
    try:
        arr = np.asarray(val, dtype=expected_dtype, order="C")
    except TypeError as exc:
        raise ValueError(f"Could not convert {val} to NumPy array") from exc
    return _coerce_shape_dtype(arr, expected_shape, expected_dtype)


def decode_array(
    val: EncodedArrayModel,
    info: ValidationInfo,
    expected_shape: ShapeType,
    expected_dtype: Optional[str],
) -> ArrayLike:
    """Decode an EncodedArrayModel to a NumPy array."""
    context = info.context if info.context else {}

    try:
        if val.data.encoding == "base64":
            data = _load_base64_arraydict(val.model_dump())

        elif val.data.encoding == "binref":
            data = _load_binref_arraydict(val.model_dump(), context.get("base_dir"))

        # keep checking for "raw" for backwards compat
        elif val.data.encoding in {"json", "raw"}:
            data = np.asarray(val.data.buffer, dtype=val.dtype).reshape(val.shape)

        else:
            # Unreachable
            raise AssertionError(f"Unsupported encoding: {val.data.encoding}")

    except Exception as e:
        raise ValueError(f"Failed to decode buffer as {val.data.encoding}: {e}") from e

    data = _coerce_shape_dtype(data, expected_shape, expected_dtype)
    return data


def encode_array(
    arr: ArrayLike, info: Any, expected_shape: ShapeType, expected_dtype: Optional[str]
) -> Union[EncodedArrayModel, ArrayLike]:
    """Encode a NumPy array as an EncodedArrayModel."""
    # Convert to a NumPy array if necessary
    arr = python_to_array(arr, expected_shape, expected_dtype)

    context = info.context if info.context else {}

    # Python mode -> just return the array without encoding
    if not info.mode_is_json():
        return arr

    array_encoding = context.get("array_encoding", "json")
    if array_encoding == "base64":
        data = _dump_base64_arraydict(arr)
    elif array_encoding == "binref":
        if not context.get("base_dir"):
            raise ValueError(
                "To write data with binref encoding you have to provide a 'base_dir' to "
                "store array data like so: "
                "`context={'array_encoding':'binref', 'base_dir': 'path/to/arraydata'}"
            )
        data, new_binref_uuid = _dump_binref_arraydict(
            arr,
            base_dir=context.get("base_dir"),
            current_binref_uuid=context.get("__binref_uuid", str(uuid4())),
            max_file_size=context.get("max_file_size", MAX_BINREF_BUFFER_SIZE),
        )
        context["__binref_uuid"] = new_binref_uuid
    elif array_encoding == "json":
        data = _dump_json_arraydict(arr)
    else:
        # Unreachable
        raise AssertionError(f"Unsupported encoding: {array_encoding}")

    return EncodedArrayModel.model_validate(data)
