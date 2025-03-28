# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator, Sequence
from typing import (
    Annotated,
    Any,
    Callable,
    Union,
    get_args,
    get_origin,
)

from pydantic import (
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    TypeAdapter,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import SchemaSerializer, SchemaValidator, core_schema

from tesseract_core.runtime.file_interactions import parent_path
from tesseract_core.runtime.schema_types import safe_issubclass


class LazySequence(Sequence):
    """Lazy sequence type that loads items from a file handle on access.

    This allows users to define a sequence of objects that are lazily loaded from a data source,
    and validated when accessed.

    When used as a Pydantic annotation, lazy sequences accept either a list of objects or a
    glob pattern to load objects from a file path.

    Example:
        >>> class MyModel(BaseModel):
        ...     objects: LazySequence[str]
        >>> model = MyModel.model_validate({"objects": ["item1", "item2"]})
        >>> model.objects[0]
        'item1'
        >>> model = MyModel.model_validate({"objects": "@/path/to/data/*.json"})
        >>> model.objects[1]
        'item2'
    """

    def __init__(self, keys: Sequence[Any], getter: Callable[[Any], Any]) -> None:
        """Initialize a LazySequence with the given keys and getter function.

        Args:
            keys: Sequence of keys to load items from.
            getter: Function that loads an item from a key.

        Example:
            >>> items = LazySequence(["item1", "item2"], lambda key: f"Loaded {key}")
            >>> items[0]
            'Loaded item1'
        """
        self.keys = keys
        self.getter = getter

    def __class_getitem__(cls, base_type: type) -> type:
        """Create a new type annotation based on the given wrapped type."""
        # Support for LazySequence[MyObject] syntax
        return Annotated[Sequence[base_type], PydanticLazySequenceAnnotation]

    @classmethod
    def __get_pydantic_core_schema__(cls, *args: Any, **kwargs: Any) -> None:
        # Raise if LazySequence is accidentally used as Pyedantic annotation without a wrapped type
        raise NotImplementedError(
            f"Generic {cls.__name__} objects do not support Pydantic schema generation. "
            f"Did you mean to use {cls.__name__}[MyObject]?"
        )

    def __getitem__(self, key: int) -> Any:
        if not isinstance(key, int):
            raise TypeError("LazySequence indices must be integers")
        return self.getter(self.keys[key])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(keys={self.keys})"

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self) -> Iterator[Any]:
        return (self.__getitem__(idx) for idx in range(len(self)))


class PydanticLazySequenceAnnotation:
    """Pydantic annotation for lazy sequences."""

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

        def create_sequence(maybe_path: Union[str, Sequence[Any]]) -> LazySequence:
            """Expand a glob pattern into a LazySequence if needed."""
            validator = SchemaValidator(item_schema)

            if not isinstance(maybe_path, str) or not maybe_path.startswith("@"):
                items = maybe_path
                getter = validator.validate_python
                return LazySequence(items, getter)

            # We know that the path is a glob pattern, so we need to load items from files
            from .file_interactions import (
                expand_glob,
                guess_format_from_path,
                load_bytes,
                read_from_path,
            )

            maybe_path = maybe_path[1:]
            items = expand_glob(maybe_path)

            def load_item(key: str) -> Any:
                buffer = read_from_path(key)
                format = guess_format_from_path(key)
                obj = load_bytes(buffer, format)
                context = {"base_dir": parent_path(key)}
                return validator.validate_python(obj, context=context)

            return LazySequence(items, load_item)

        def serialize(obj: LazySequence, __info: Any) -> Any:
            """When serializing, convert the LazySequence to a list of items.

            This is not an encouraged use case, but it is supported for completeness.
            """
            materialized_sequence = list(obj)
            serializer = SchemaSerializer(sequence_schema)

            return serializer.to_python(materialized_sequence, **__info.__dict__)

        origin = get_origin(_source_type)
        if not safe_issubclass(origin, Sequence):
            # should never happen, since we always use Annotated[Sequence[...], PydanticLazySequenceAnnotation]
            raise ValueError(
                f"LazySequence can only be used with Sequence types, not {origin}"
            )

        # This is a Sequence, so args is a single type
        args = get_args(_source_type)
        assert len(args) == 1

        # Wrap in TypeAdapter so we don't need conditional logic for Python types vs. Pydantic models
        item_schema = TypeAdapter(args[0]).core_schema
        sequence_schema = _handler(_source_type)

        obj_or_path = core_schema.union_schema(
            [sequence_schema, core_schema.str_schema(pattern="^@")]
        )
        load_schema = core_schema.chain_schema(
            # first load data, then validate it with the wrapped schema
            [
                obj_or_path,
                core_schema.no_info_plain_validator_function(
                    create_sequence,
                    serialization=core_schema.plain_serializer_function_ser_schema(
                        serialize,
                        info_arg=True,
                        return_schema=sequence_schema,
                    ),
                ),
            ]
        )
        return core_schema.json_or_python_schema(
            json_schema=load_schema,
            python_schema=load_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize,
                info_arg=True,
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """This method is called by Pydantic to get the JSON schema for the annotated type."""
        return handler(_core_schema)
