# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta
from enum import IntEnum
from functools import partial
from typing import (
    Annotated,
    Any,
    Optional,
    Union,
    get_args,
    get_origin,
)

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from tesseract_core.runtime.array_encoding import (
    AllowedDtypes,
    decode_array,
    encode_array,
    get_array_model,
    python_to_array,
)

AnnotatedType = type(Annotated[Any, Any])
EllipsisType = type(Ellipsis)


def _ensure_valid_shapedtype(expected_shape: Any, expected_dtype: Any) -> tuple:
    if not isinstance(expected_shape, (tuple, EllipsisType)):
        raise ValueError(
            "Shape in Array[<shape>, <dtype>] must be a tuple or '...' (ellipsis)"
        )

    if isinstance(expected_shape, tuple):
        for dim in expected_shape:
            if dim is not None and not isinstance(dim, int):
                raise ValueError(
                    "Shape values in Array[<shape>, <dtype>] must be integers or None"
                )

    if is_array_annotation(expected_dtype):
        expected_dtype = expected_dtype.__metadata__[0].expected_dtype

    allowed_dtypes = get_args(AllowedDtypes)

    if expected_dtype not in allowed_dtypes and expected_dtype is not None:
        raise ValueError(
            f"Invalid dtype in Array[<shape>, <dtype>]: {expected_dtype} "
            f"(must be one of {allowed_dtypes} or a scalar Array type like, Array[(), Int32])"
        )
    return expected_shape, expected_dtype


class ShapeDType(BaseModel):
    """Data structure describing an array's shape and data type."""

    shape: tuple[int, ...]
    dtype: AllowedDtypes
    # Ignore extra fields in the model, to allow encoded arrays to be passed
    model_config = ConfigDict(extra="ignore")

    def __class_getitem__(
        cls,
        key: tuple[
            Union[tuple[Optional[int], ...], EllipsisType],
            Union[AnnotatedType, str, None],
        ],
    ) -> AnnotatedType:
        expected_shape, expected_dtype = _ensure_valid_shapedtype(*key)

        def validate(shapedtype: ShapeDType) -> ShapeDType:
            """Validator to check if the shape and dtype match the expected values."""
            if isinstance(shapedtype, ShapeDType):
                shape = shapedtype.shape
                if expected_shape is Ellipsis:
                    return shapedtype
                for actual, expected in zip(shape, expected_shape, strict=True):
                    if expected is not None and actual != expected:
                        raise ValueError(
                            f"Expected shape: {expected_shape}. Found: {shape}."
                        )
            return shapedtype

        return Annotated[ShapeDType, AfterValidator(validate)]

    @classmethod
    def from_array_annotation(cls, obj: AnnotatedType) -> AnnotatedType:
        """Create a ShapeDType from an array annotation."""
        shape = obj.__metadata__[0].expected_shape
        dtype = obj.__metadata__[0].expected_dtype
        return cls[shape, dtype]


class ArrayFlags(IntEnum):
    """Custom flags for array annotations."""

    DIFFERENTIABLE = 1


class ArrayAnnotationType(ABCMeta):
    """Metaclass for Array type annotation to enforce repr on created types based on class variables.

    Example:
        >>> class MyArray(metaclass=ArrayAnnotationType):
        ...     expected_shape = (2, 3)
        ...     expected_dtype = "float32"

        >>> MyArray
        MyArray[(2, 3), 'float32']
    """

    def __repr__(cls) -> str:
        return f"{cls.__name__}[{cls.expected_shape!r}, {cls.expected_dtype!r}]"


def safe_issubclass(obj: Any, baseclass: type[object]) -> bool:
    """Check if obj is a subclass of baseclass in a way that never raises.

    (This is useful when obj is not guaranteed to be a type.)
    """
    try:
        return issubclass(obj, baseclass)
    except TypeError:
        return False


def _is_annotated(obj: Any) -> bool:
    """Check if an object is typing.Annotated or typing_extensions.Annotated."""
    return get_origin(obj) is Annotated


class PydanticArrayAnnotation(metaclass=ArrayAnnotationType):
    """Base class for Pydantic annotations for NumPy array types.

    This class provides Pydantic support for arrays with a fixed / polymorphic shape and dtype,
    with proper validation and serialization.

    See https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types

    When serializing or validating pydantic models that contain this annotation
    you can customize the array encoding to: plain json, base64, or binref. For
    more details see the docstring of 'Array'.
    """

    # These are class attributes that must be set when the class is created
    expected_shape: Union[tuple[int, ...], EllipsisType]
    expected_dtype: str
    flags: tuple[ArrayFlags]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"{self.__class__.__name__} cannot be instantiated")

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """This method is called by Pydantic to get the core schema for the annotated type.

        Does most of the heavy lifting for validation and serialization.
        """
        # Create a Pydantic model for the encoded array, for easier validation
        array_schema = _handler(
            get_array_model(
                cls.expected_shape,
                cls.expected_dtype,
                [flag.name for flag in cls.flags],
            )
        )

        python_to_array_ = partial(
            python_to_array,
            expected_shape=cls.expected_shape,
            expected_dtype=cls.expected_dtype,
        )
        encode_array_ = partial(
            encode_array,
            expected_shape=cls.expected_shape,
            expected_dtype=cls.expected_dtype,
        )
        decode_array_ = partial(
            decode_array,
            expected_shape=cls.expected_shape,
            expected_dtype=cls.expected_dtype,
        )

        load_from_dict_schema = core_schema.chain_schema(
            # first load / validate JSON, then decode into a NumPy array
            [
                array_schema,
                core_schema.with_info_plain_validator_function(
                    decode_array_,
                    serialization=core_schema.plain_serializer_function_ser_schema(
                        encode_array_,
                        info_arg=True,
                        return_schema=array_schema,
                    ),
                ),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=load_from_dict_schema,
            python_schema=core_schema.union_schema(
                [
                    load_from_dict_schema,
                    # when loading from Python, we also allow any array-like object
                    core_schema.no_info_plain_validator_function(python_to_array_),
                ],
                mode="left_to_right",
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                encode_array_,
                info_arg=True,
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """This method is called by Pydantic to get the JSON schema for the annotated type."""
        return handler(_core_schema)


class Array:
    """Generic Pydantic type annotation for a multi-dimensional array with a fixed shape and dtype.

    Arrays will be broadcasted to the expected shape and dtype during validation,
    but dimensions must match exactly.

    Polymorphic dimensions are supported by using `None` in the shape tuple.
    To indicate a scalar, use an empty tuple.

    Arrays of any shape and rank can be represented by using `...` (ellipsis) as the shape.

    Example:
        >>> class MyModel(BaseModel):
        ...     int_array: Array[(2, 3), Int32]
        ...     float_array: Array[(None, 3), Float64]
        ...     scalar_int: Array[(), Int16]
        ...     any_shape_array: Array[..., Float32]


    You can serialize to (and validate from) different array encodings.

        >>> model = MyModel(
        ...     int_array=np.array([[1, 2, 3], [4, 5, 6]]),
        ...     float_array=np.array([[1.0, 2.0, 3.0]]),
        ...     scalar_int=np.int32(42),
        ...     any_shape_array=np.array([True, False, True]).reshape(1, 1, 3),
        ... )

        >>> model.model_dump_json(context={"array_encoding": "json"})

        >>> model.model_dump_json(context={"array_encoding": "base64"})

    or to binref:

        >>> model.model_dump_json(
        ...     context={
        ...         "array_encoding": "binref",
        ...         "base_dir": "path/to/base",
        ...         "max_file_size": 10**8,
        ...     }
        ... )

    In the 'binref' case you have to provide a base_dir to save/load binary
    (.bin) files. The .bin file(s) are written to `context['base_dir'] /
    f"{context['__binref_uuid']}.bin"`.  The '__binref_uuid' is considered an internal
    variable and should not be modified manually!  You can set a 'max_file_size'
    for the binary files.  When this file size (in bytes) is reached, a new
    __binref_uuid (i.e. a new .bin) is created to append array data to.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        clsname = self.__class__.__name__
        raise RuntimeError(
            f"{clsname} cannot be instantiated directly, perhaps you meant to use `{clsname}[(shape), dtype]`?"
        )

    def __class_getitem__(
        cls,
        key: tuple[
            Union[tuple[Optional[int], ...], EllipsisType],
            Union[AnnotatedType, str, None],
        ],
    ) -> AnnotatedType:
        """Create a new type annotation based on the given shape and dtype."""
        expected_shape, expected_dtype = _ensure_valid_shapedtype(*key)
        classvars = {
            "expected_shape": expected_shape,
            "expected_dtype": expected_dtype,
            "flags": (),
            "__module__": cls.__module__,
        }
        model = type(cls.__name__, (PydanticArrayAnnotation,), classvars)
        return Annotated[np.ndarray, model]


def is_array_annotation(obj: Any) -> bool:
    """Check if an object is a Pydantic array type annotation."""
    if _is_annotated(obj):
        metadata = obj.__metadata__[0]
        if safe_issubclass(metadata, PydanticArrayAnnotation):
            return True
    return False


class Differentiable:
    """Type annotation for a differentiable array.

    Example:
        >>> class MyModel(BaseModel):
        ...     array: Differentiable[Array[(None, 3), Float64]]
    """

    def __class_getitem__(cls, key: Any) -> AnnotatedType:
        """Mark wrapped array type as differentiable."""
        if not is_array_annotation(key):
            raise ValueError("Differentiable can only be applied to Array types")

        arr = key.__metadata__[0]
        # Create a new array type with the DIFFERENTIABLE flag, to not modify the original type in-place
        newarr = type(
            arr.__name__, (arr,), {"flags": (*arr.flags, ArrayFlags.DIFFERENTIABLE)}
        )
        return Annotated[key.__origin__, newarr]


def is_differentiable(obj: Any) -> bool:
    """Check if an object is a Differentiable array type annotation."""
    if is_array_annotation(obj):
        return ArrayFlags.DIFFERENTIABLE in obj.__metadata__[0].flags

    return False


# Export concrete scalar types
Float16 = Array[(), "float16"]
Float32 = Array[(), "float32"]
Float64 = Array[(), "float64"]
Int8 = Array[(), "int8"]
Int16 = Array[(), "int16"]
Int32 = Array[(), "int32"]
Int64 = Array[(), "int64"]
Bool = Array[(), "bool"]
UInt8 = Array[(), "uint8"]
UInt16 = Array[(), "uint16"]
UInt32 = Array[(), "uint32"]
UInt64 = Array[(), "uint64"]
Complex64 = Array[(), "complex64"]
Complex128 = Array[(), "complex128"]
