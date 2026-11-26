# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import re
import types
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import copy
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    create_model,
    field_validator,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.dataclasses import is_pydantic_dataclass
from pydantic_core import PydanticSerializationError
from typing_extensions import NotRequired, Required, is_typeddict
from typing_extensions import TypedDict as ExtTypedDict

from .schema_types import (
    Array,
    PydanticArrayAnnotation,
    ShapeDType,
    is_differentiable,
    safe_issubclass,
)
from .tree_transforms import get_at_path

# Constants to mark sequence and dict indexing in pytree paths
SEQ_INDEX_SENTINEL = object()
DICT_INDEX_SENTINEL = object()

T = TypeVar("T")

# Python has funnily enough two union types now. See https://github.com/python/cpython/issues/105499
# We check against both for compatibility with older versions of Python.
UNION_TYPES = [Union, types.UnionType]


# Building a TypeAdapter generates a pydantic core schema, which for an array
# annotation means creating a new model class. That costs far more than the
# validation itself, so gradient endpoints must not build adapters per request.
@functools.lru_cache(maxsize=1024)
def _type_adapter(annotation: Any) -> TypeAdapter:
    """TypeAdapter for an annotation taken from a Tesseract's schema."""
    return TypeAdapter(annotation)


@functools.lru_cache(maxsize=1024)
def _exact_array_adapter(shape: tuple, dtype: str | None) -> TypeAdapter:
    """TypeAdapter for ``Array[shape, dtype]``, cached by ``(shape, dtype)``.

    ``Array[...]`` returns a new class per call, so it cannot be the cache key.
    """
    return TypeAdapter(Array[shape, dtype])


def _construct_annotated(obj: Any, metadata: Iterable[Any]) -> Any:
    """Construct an Annotated type with the given metadata."""
    out = obj
    # Exploit that repeated Annotated applications are flattened
    for meta in metadata:
        out = Annotated[out, meta]
    return out


def _is_annotated(obj: Any) -> bool:
    """Check if an object is typing.Annotated or typing_extensions.Annotated."""
    return get_origin(obj) is Annotated


def _is_namedtuple(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, tuple) and hasattr(obj, "_fields")


def _has_fields(obj: Any) -> bool:
    """Whether ``obj`` is a NamedTuple, dataclass or TypedDict class.

    These have a fixed set of named fields, unlike generics such as
    ``list[...]``, whose children come from type arguments. Pydantic models
    have fields too but are handled separately. Instances return False.
    """
    return isinstance(obj, type) and (
        _is_namedtuple(obj) or dataclasses.is_dataclass(obj) or is_typeddict(obj)
    )


def _without_required(annotation: Any) -> Any:
    """Strip a ``Required[...]`` / ``NotRequired[...]`` wrapper from a TypedDict field."""
    if get_origin(annotation) in (Required, NotRequired):
        return get_args(annotation)[0]
    return annotation


def _container_hints(cls: type) -> dict[str, Any] | None:
    """The field annotations of container ``cls``, keeping ``Annotated`` metadata.

    Returns None, with a warning, if they can't be resolved, and the container
    is then treated as a single leaf. This happens for some forward references
    that pydantic resolves but ``get_type_hints`` can't, such as names defined
    inside a function.
    """
    try:
        hints = get_type_hints(cls, include_extras=True)
    except (NameError, TypeError) as exc:
        warnings.warn(
            f"Cannot resolve the annotations of {cls.__qualname__} ({exc}), so "
            "arrays inside it are not treated as arrays, e.g. not differentiable.",
            stacklevel=2,
        )
        return None
    if _is_namedtuple(cls):
        return {name: hints[name] for name in cls._fields}
    if dataclasses.is_dataclass(cls):
        return {f.name: hints[f.name] for f in dataclasses.fields(cls)}
    return {key: _without_required(t) for key, t in hints.items()}


def _rebuild_container(
    cls: type, hints: dict[str, Any], new_types: dict[str, Any], name: str
) -> type:
    """A container like ``cls`` with new field types, or ``cls`` itself if none changed."""
    if new_types == hints:
        return cls
    if _is_namedtuple(cls):
        return _rebuild_namedtuple(cls, new_types, name)
    if dataclasses.is_dataclass(cls):
        return _rebuild_dataclass(cls, hints, new_types, name)
    return _rebuild_typeddict(cls, new_types, name)


def _rebuild_namedtuple(cls: type, new_types: dict[str, Any], name: str) -> type:
    new = NamedTuple(name, list(new_types.items()))
    new.__new__.__defaults__ = cls.__new__.__defaults__
    new._field_defaults = dict(cls._field_defaults)
    new.__module__ = cls.__module__
    return new


def _rebuild_dataclass(
    cls: type, hints: dict[str, Any], new_types: dict[str, Any], name: str
) -> type:
    """A subclass of dataclass ``cls`` that re-declares the fields whose types changed.

    Subclassing keeps the original's methods and ``isinstance`` checks, as for
    pydantic models.
    """
    changed = {k: t for k, t in new_types.items() if t != hints[k]}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    namespace: dict[str, Any] = {
        "__annotations__": changed,
        "__module__": cls.__module__,
        # The dataclass decorator re-reads each copied field's name and type
        **{key: copy(fields[key]) for key in changed},
    }
    new = types.new_class(name, (cls,), exec_body=lambda ns: ns.update(namespace))
    if is_pydantic_dataclass(cls):
        return pydantic_dataclass(new, config=cls.__pydantic_config__)
    if "__dataclass_fields__" not in new.__dict__:
        # A plain subclass isn't a dataclass yet; one whose base makes its
        # subclasses dataclasses (e.g. via a metaclass) already is.
        params = cls.__dataclass_params__
        new = dataclasses.dataclass(
            eq=params.eq,
            order=params.order,
            unsafe_hash=params.unsafe_hash,
            frozen=params.frozen,
        )(new)
    return new


def _rebuild_typeddict(cls: type, new_types: dict[str, Any], name: str) -> type:
    fields = {
        k: Required[t] if k in cls.__required_keys__ else NotRequired[t]
        for k, t in new_types.items()
    }
    # pydantic needs typing_extensions' TypedDict before Python 3.12
    new = ExtTypedDict(name, fields)
    new.__module__ = cls.__module__
    return new


def _wrap_default_factory(
    factory: Callable[..., Any], convert: Callable[[Any], Any]
) -> Callable[..., Any]:
    """Wrap a default_factory so its output goes through ``convert``.

    ``functools.wraps`` keeps the factory's signature visible to pydantic, which
    inspects it to decide whether to pass the validated data.
    """

    @functools.wraps(factory)
    def wrapped(*args: Any) -> Any:
        return convert(factory(*args))

    return wrapped


@dataclasses.dataclass(frozen=True)
class _RebuildOptions:
    """How ``apply_function_to_model_tree`` rebuilds models and containers."""

    model_prefix: str
    default_model_config: dict[str, Any]
    make_default_converter: Callable[[Any, Any], Callable[[Any], Any] | None] | None


@dataclasses.dataclass(frozen=True)
class _SplitAnnotation:
    """An annotation split into the annotations it contains.

    Each child comes with the path segments that lead to it.
    ``rebuild`` makes the annotation again from new children, in the same order.
    """

    children: list[tuple[tuple, Any]]
    rebuild: Callable[[list[Any], _RebuildOptions], Any]


def _split_model(annotation: Any) -> _SplitAnnotation | None:
    if not safe_issubclass(annotation, BaseModel):
        return None
    fields = annotation.model_fields
    old_types = [
        _construct_annotated(field.annotation, field.metadata)
        for field in fields.values()
    ]

    def rebuild(new_types: list[Any], opts: _RebuildOptions) -> type[BaseModel]:
        new_fields = {}
        for (name, field), old_type, new_type in zip(
            fields.items(), old_types, new_types, strict=True
        ):
            new_field = copy(field)
            # Need to strip off metadata to trigger re-evaluation of the field
            new_field.metadata = []
            convert = None
            if opts.make_default_converter is not None and not field.is_required():
                convert = opts.make_default_converter(old_type, new_type)
            if convert is not None:
                if field.default_factory is not None:
                    new_field.default_factory = _wrap_default_factory(
                        field.default_factory, convert
                    )
                else:
                    new_field.default = convert(field.default)
            new_fields[name] = (new_type, new_field)

        # Only override model_config if it is not already present
        # in the pydantic model definition
        model_config = ConfigDict(**opts.default_model_config)
        model_config.update(annotation.model_config)

        # Make sure that reconstructed RootModels don't contain unsupported 'extra' config.
        if safe_issubclass(annotation, RootModel):
            model_config.pop("extra", None)

        return create_model(
            f"{opts.model_prefix}{annotation.__name__}",
            **new_fields,
            model_config=(ConfigDict, model_config),
            __base__=annotation,
        )

    return _SplitAnnotation(
        [((name,), t) for name, t in zip(fields, old_types, strict=True)], rebuild
    )


def _split_fields(annotation: Any) -> _SplitAnnotation | None:
    if not _has_fields(annotation):
        return None
    hints = _container_hints(annotation)
    if hints is None:
        return None

    def rebuild(new_types: list[Any], opts: _RebuildOptions) -> type:
        return _rebuild_container(
            annotation,
            hints,
            dict(zip(hints, new_types, strict=True)),
            f"{opts.model_prefix}{annotation.__name__}",
        )

    # A NamedTuple is sent and received as a list, so its items are pathed by
    # position ("nt.[0]"), like a tuple's. The fields of dataclasses and
    # TypedDicts are pathed by name ("dc.a"), like a model's.
    by_position = _is_namedtuple(annotation)
    return _SplitAnnotation(
        [
            ((SEQ_INDEX_SENTINEL,) if by_position else (key,), t)
            for key, t in hints.items()
        ],
        rebuild,
    )


def _split_annotated(annotation: Any) -> _SplitAnnotation | None:
    if not _is_annotated(annotation):
        return None
    # __origin__ and __metadata__ instead of get_args for compatibility with
    # typing_extensions.Annotated
    metadata = annotation.__metadata__
    return _SplitAnnotation(
        [((), annotation.__origin__)],
        lambda new, _: _construct_annotated(new[0], metadata),
    )


def _split_union(annotation: Any) -> _SplitAnnotation | None:
    # Optional[X] is a Union too
    if not any(get_origin(annotation) is t for t in UNION_TYPES):
        return None
    return _SplitAnnotation(
        [((), arg) for arg in get_args(annotation)],
        lambda new, _: Union[tuple(new)],  # noqa: UP007
    )


def _split_mapping(annotation: Any) -> _SplitAnnotation | None:
    origin = get_origin(annotation)
    if not safe_issubclass(origin, Mapping) or not get_args(annotation):
        return None
    key_type, value_type = get_args(annotation)
    return _SplitAnnotation(
        [((DICT_INDEX_SENTINEL,), value_type)],
        lambda new, _: origin[key_type, new[0]],
    )


def _split_iterable(annotation: Any) -> _SplitAnnotation | None:
    # list, set, tuple and the like
    origin = get_origin(annotation)
    args = get_args(annotation)
    if not safe_issubclass(origin, Iterable) or not args:
        return None
    # The ... in tuple[X, ...] marks a variable-length tuple and is not an item type
    item_types = [arg for arg in args if arg is not Ellipsis]

    def rebuild(new_types: list[Any], _: _RebuildOptions) -> Any:
        new_items = iter(new_types)
        return origin[
            tuple(arg if arg is Ellipsis else next(new_items) for arg in args)
        ]

    return _SplitAnnotation([((SEQ_INDEX_SENTINEL,), t) for t in item_types], rebuild)


# Tried in order until one returns a node. Annotations that none of them split
# are leaves.
_SPLITTERS: list[Callable[[Any], _SplitAnnotation | None]] = [
    _split_model,
    _split_fields,
    _split_annotated,
    _split_union,
    _split_mapping,
    _split_iterable,
]


def _split(
    annotation: Any, path: tuple, enclosing_ids: frozenset[int]
) -> _SplitAnnotation | None:
    """Split ``annotation`` into its children, or return None for a leaf.

    ``enclosing_ids`` holds the ids of the annotations that contain this one.
    """
    if id(annotation) in enclosing_ids:
        if safe_issubclass(annotation, BaseModel):
            raise ValueError(
                f"Recursive model definition detected at path {list(path)}: "
                f"{annotation.__name__}. Recursive models are not supported as "
                "Tesseract schemas."
            )
        # A container nested inside itself is a leaf where it reappears
        return None
    for splitter in _SPLITTERS:
        node = splitter(annotation)
        if node is not None:
            return node
    return None


def _iter_tree(
    annotation: Any, path: tuple = (), enclosing_ids: frozenset[int] = frozenset()
) -> Iterable[tuple[tuple, Any, _SplitAnnotation | None]]:
    """Yield ``(path, annotation, node)`` for every annotation in the tree, parents first.

    ``node`` is None for leaves.
    """
    node = _split(annotation, path, enclosing_ids)
    yield path, annotation, node
    if node is not None:
        inner = enclosing_ids | {id(annotation)}
        for segments, child in node.children:
            yield from _iter_tree(child, (*path, *segments), inner)


def apply_function_to_model_tree(
    Schema: type[BaseModel],
    func: Callable[[Any, tuple], Any],
    model_prefix: str = "",
    default_model_config: dict[str, Any] | None = None,
    make_default_converter: Callable[[Any, Any], Callable[[Any], Any] | None]
    | None = None,
) -> type[BaseModel]:
    """Apply a function to all leaves of a Pydantic model, recursing into containers + nested models.

    A leaf is any annotation that isn't a Pydantic model or a container. ``func`` is
    called as ``func(annotation, path)`` and returns the leaf's new annotation. A path
    is a tuple of field names and sentinels. For example, the ``int`` in
    ``a: list[dict[str, int]]`` has path ``("a", SEQ_INDEX_SENTINEL, DICT_INDEX_SENTINEL)``.

    If given, make_default_converter is called as
    ``make_default_converter(old_type, new_type)`` for every field that has a default,
    when the model is built. It returns a function mapping the old default to the new
    one, or None to keep the default as is. A ``default=`` is converted right away,
    and a ``default_factory=`` is wrapped so that each value it produces is converted.
    """
    opts = _RebuildOptions(
        model_prefix, default_model_config or {}, make_default_converter
    )

    def visit(annotation: Any, path: tuple, enclosing_ids: frozenset[int]) -> Any:
        node = _split(annotation, path, enclosing_ids)
        if node is None:
            return func(annotation, path)
        inner = enclosing_ids | {id(annotation)}
        new_children = [
            visit(child, (*path, *segments), inner) for segments, child in node.children
        ]
        return node.rebuild(new_children, opts)

    return visit(Schema, (), frozenset())


def _path_tuple_to_str(pathtuple: tuple) -> str:
    """Convert a path tuple to a dotted string, replacing sentinels with [] / {} indexing."""
    str_parts = []
    for part in pathtuple:
        if part is SEQ_INDEX_SENTINEL:
            str_parts.append("[]")
        elif part is DICT_INDEX_SENTINEL:
            str_parts.append("{}")
        else:
            str_parts.append(part)
    return ".".join(str_parts)


def _serialize_diffable_arrays(
    obj: dict[tuple, Any],
) -> dict[str, dict[str, Any]]:
    """Convert a dict {path_tuple: array_type} to a dict {path_str: {shape, dtype}}.

    path_str is a string representation of the path, with SEQ_INDEX_SENTINEL and DICT_INDEX_SENTINEL
    replaced by indexing syntax (e.g. `foo.[].{}`).
    """
    serialized = {}
    for pathtuple, value in obj.items():
        shape = value.expected_shape
        dtype = value.expected_dtype

        # Ensure shape is JSON serializable
        if shape is Ellipsis:
            shape = None
        else:
            shape = tuple(shape)

        serialized[_path_tuple_to_str(pathtuple)] = {
            "shape": shape,
            "dtype": dtype,
        }

    return serialized


def _serialize_field_order(obj: dict[tuple, list[str]]) -> dict[str, list[str]]:
    """Convert a dict {path_tuple: [field_names]} to a dict {path_str: [field_names]}."""
    return {_path_tuple_to_str(p): fields for p, fields in obj.items()}


def create_apply_schema(
    InputSchema: type[BaseModel], OutputSchema: type[BaseModel]
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Create the input / output schemas for the /apply endpoint."""
    # We add metadata to the input and output schemas to indicate which fields are differentiable,
    # what their paths are, and which expected shape / dtype they have.
    # This is used internally and by some official clients, but not advertised as part of the public API,
    # so people should not rely on it.
    diffable_input_paths = get_all_model_path_patterns(
        InputSchema, filter_fn=is_differentiable
    )
    diffable_output_paths = get_all_model_path_patterns(
        OutputSchema, filter_fn=is_differentiable
    )

    # Snapshot field declaration order for every (nested) model, since JSON object key order is
    # not guaranteed to be preserved by downstream clients (e.g. postgres).
    input_field_order = get_field_order(InputSchema)
    output_field_order = get_field_order(OutputSchema)

    InputSchema = apply_function_to_model_tree(
        InputSchema,
        lambda x, _: x,
        model_prefix="Apply_",
        default_model_config=dict(extra="forbid"),
    )
    OutputSchema = apply_function_to_model_tree(
        OutputSchema,
        lambda x, _: x,
        model_prefix="Apply_",
        default_model_config=dict(extra="forbid"),
    )

    class ApplyInputSchema(BaseModel):
        inputs: InputSchema = Field(
            ..., description="The input data to apply the Tesseract to."
        )

        differentiable_arrays: ClassVar[dict[str, dict[str, Any]]] = (
            _serialize_diffable_arrays(diffable_input_paths)
        )
        field_order: ClassVar[dict[str, list[str]]] = _serialize_field_order(
            input_field_order
        )
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={
                "differentiable_arrays": differentiable_arrays,
                "field_order": field_order,
            },
        )

    class ApplyOutputSchema(RootModel):
        root: OutputSchema = Field(
            ..., description="The output data from applying the Tesseract."
        )
        differentiable_arrays: ClassVar[dict[str, dict[str, Any]]] = (
            _serialize_diffable_arrays(diffable_output_paths)
        )
        field_order: ClassVar[dict[str, list[str]]] = _serialize_field_order(
            output_field_order
        )
        model_config = ConfigDict(
            json_schema_extra={
                "differentiable_arrays": differentiable_arrays,
                "field_order": field_order,
            },
        )

    return ApplyInputSchema, ApplyOutputSchema


# Validation context key that lets the abstract schema's array fields accept real
# arrays. Only passed when converting defaults, so abstract_eval still rejects them.
_CONVERT_DEFAULT_KEY = "tesseract_convert_abstract_default"


def _array_default_to_shapedtype(value: Any, info: ValidationInfo) -> Any:
    """Before-validator replacing a real array by its ShapeDType, when converting a default."""
    converting = (info.context or {}).get(_CONVERT_DEFAULT_KEY, False)
    # Defaults reach here validated against the real Array annotation, which turns
    # any array-like (JAX, PyTorch, ...) into a NumPy array, so NumPy is all we see.
    if converting and isinstance(value, np.ndarray | np.generic):
        return ShapeDType(shape=value.shape, dtype=str(value.dtype))
    return value


def _abstract_default_converter(
    old_type: Any, new_type: Any
) -> Callable[[Any], Any] | None:
    """Return a function turning a default of ``old_type`` into one of ``new_type``.

    The default is validated against the original annotation, so it is cast to the
    declared dtype (``[1, 2]`` on a ``Float64`` field becomes float64), and dumped
    back to plain data, which turns model instances into dicts but keeps arrays.
    Validating that against the abstract annotation, with the context that lets its
    array fields accept arrays, replaces each array by its ShapeDType.
    A default that fails validation is returned unchanged.

    Returns None if ``new_type`` has no arrays to replace.
    """
    if new_type == old_type:
        return None

    # Built once per field, not cached: annotations may be unhashable
    real_adapter = TypeAdapter(old_type)
    abstract_adapter = TypeAdapter(new_type)

    def convert(default: Any) -> Any:
        try:
            real = real_adapter.validate_python(default)
            plain = real_adapter.dump_python(real, by_alias=True)
            return abstract_adapter.validate_python(
                plain, context={_CONVERT_DEFAULT_KEY: True}
            )
        except (ValidationError, PydanticSerializationError):
            return default

    return convert


def create_abstract_eval_schema(
    InputSchema: type[BaseModel], OutputSchema: type[BaseModel]
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Create new Pydantic models that represent abstract versions of the given schemas.

    The returned schemas will have the same structure as the input schemas, but with array
    fields replaced by ShapeDType objects.
    """

    def replace_array_with_shapedtype(obj: T, _: Any) -> T | type[ShapeDType]:
        if safe_issubclass(obj, PydanticArrayAnnotation):
            return Annotated[
                ShapeDType.from_array_type(obj),
                BeforeValidator(_array_default_to_shapedtype),
            ]
        return obj

    GeneratedInputSchema = apply_function_to_model_tree(
        InputSchema,
        replace_array_with_shapedtype,
        model_prefix="AbstractEval_",
        default_model_config=dict(extra="forbid"),
        make_default_converter=_abstract_default_converter,
    )

    GeneratedOutputSchema = apply_function_to_model_tree(
        OutputSchema,
        replace_array_with_shapedtype,
        model_prefix="AbstractEval_",
        default_model_config=dict(extra="forbid"),
        make_default_converter=_abstract_default_converter,
    )

    class AbstractInputSchema(BaseModel):
        inputs: GeneratedInputSchema = Field(
            ...,
            description=(
                "The abstract input data to evaluate the Tesseract on. Has the same structure as InputSchema, "
                "but with array fields replaced by ShapeDType."
            ),
        )
        model_config = ConfigDict(extra="forbid")

    class AbstractOutputSchema(RootModel):
        root: GeneratedOutputSchema = Field(
            ...,
            description=(
                "Abstract outputs with the same structure as OutputSchema, but with array fields "
                "replaced by ShapeDType."
            ),
        )

    return AbstractInputSchema, AbstractOutputSchema


def get_all_model_path_patterns(
    schema: type[BaseModel], filter_fn: Callable[[type], bool] | None = None
) -> dict[tuple, type]:
    """Return a dictionary mapping the path of every leaf in the schema to its type.

    Paths are as described in ``apply_function_to_model_tree``.

    Args:
        schema: The Pydantic model schema to traverse.
        filter_fn: Optional predicate to filter which paths to include.
                   If None, includes all paths. If provided, only includes
                   paths where filter_fn(type) returns True.
    """
    return {
        path: annotation
        for path, annotation, node in _iter_tree(schema)
        if node is None and (filter_fn is None or filter_fn(annotation))
    }


def get_field_order(schema: type[BaseModel]) -> dict[tuple, list[str]]:
    """Return a mapping {path: [field_names_in_declaration_order]} for every object in the tree.

    Objects are models, dataclasses and TypedDicts, i.e. everything sent as a JSON
    object with fixed keys. The root model is keyed by the empty tuple. Objects at
    the same path (e.g. the members of a Union) share one entry.
    """
    field_order_by_path: dict[tuple, dict[str, None]] = {}
    for path, _, node in _iter_tree(schema):
        if node is None:
            continue
        names = [seg[0] for seg, _ in node.children if seg and isinstance(seg[0], str)]
        if names:
            field_order_by_path.setdefault(path, {}).update(dict.fromkeys(names))
    return {p: list(fields) for p, fields in field_order_by_path.items()}


def _path_to_pattern(path: Sequence[str | object]) -> str:
    """Return a type describing valid paths for all passed paths."""
    # Check if path includes sequence or dict indexing -- in this case, we can't use the
    # path as a literal and need to use a regex pattern instead.
    final_path = []
    is_literal = True
    for part in path:
        if part is SEQ_INDEX_SENTINEL:
            is_literal = False
            part = r"\[\d+\]"
        elif part is DICT_INDEX_SENTINEL:
            is_literal = False
            part = r"\{(?:[^{}\\]|\\.)+\}"
        final_path.append(part)

    if is_literal:
        return ".".join(final_path)

    return r"\.".join(final_path)


def _pattern_to_type(pattern: str) -> type:
    """Convert a string pattern (which may be a literal or regex) to a type that can be used for validation."""
    if _is_regex_pattern(pattern):
        # Anchored, since a pattern is searched for and not matched in full.
        return Annotated[str, Field(pattern=f"^{pattern}$")]
    else:
        return Literal[pattern]


def _is_regex_pattern(pattern: str) -> bool:
    # Poor man's regex detection, but it should work for our purposes
    return "\\" in pattern


def input_path_validator(path: str, info: ValidationInfo) -> str:
    """Validate that the given path points to a valid input key."""
    if "[" in path or "{" in path:
        try:
            get_at_path(info.data["inputs"], path)
        except (LookupError, AttributeError) as exc:
            raise ValueError(
                f"Could not find {info.field_name} path {path} in inputs."
            ) from exc
    return path


def create_gradient_schema(
    InputSchema: type[BaseModel],
    OutputSchema: type[BaseModel],
    gradient_type: Literal["jacobian", "jvp", "vjp"],
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Generate the input / outputs schemas for gradient endpoints like /jacobian, /jacobian_vector_product, etc.

    Returns a tuple (InputSchema, OutputSchema) with the generated schemas.
    """
    diffable_input_paths = get_all_model_path_patterns(
        InputSchema, filter_fn=is_differentiable
    )
    diffable_output_paths = get_all_model_path_patterns(
        OutputSchema, filter_fn=is_differentiable
    )

    if not diffable_input_paths:
        raise RuntimeError("No differentiable inputs found in the input schema")

    if not diffable_output_paths:
        raise RuntimeError("No differentiable outputs found in the output schema")

    diffable_input_patterns = {
        _path_to_pattern(path): obj for path, obj in diffable_input_paths.items()
    }
    diffable_output_patterns = {
        _path_to_pattern(path): obj for path, obj in diffable_output_paths.items()
    }

    diffable_input_type = Union[  # noqa: UP007
        tuple(_pattern_to_type(p) for p in diffable_input_patterns)
    ]
    diffable_output_type = Union[  # noqa: UP007
        tuple(_pattern_to_type(p) for p in diffable_output_patterns)
    ]

    def _find_annotation_from_path(path_patterns: dict, concrete_path: str) -> type:
        """Find the PydanticArrayAnnotation for a given concrete path."""
        for path_pattern, array_type in path_patterns.items():
            if _is_regex_pattern(path_pattern):
                path_matches = bool(re.fullmatch(path_pattern, concrete_path))
            else:
                path_matches = path_pattern == concrete_path

            if path_matches:
                return array_type

        raise ValueError(f"Invalid path: {concrete_path}")

    def _find_shape_from_path(
        path_patterns: dict, concrete_path: str
    ) -> tuple | types.EllipsisType:
        return _find_annotation_from_path(path_patterns, concrete_path).expected_shape

    InputSchema = apply_function_to_model_tree(
        InputSchema,
        lambda x, _: x,
        model_prefix=f"{gradient_type.title()}_",
        default_model_config=dict(extra="forbid"),
    )

    def result_validator(
        result: dict[str, Any], info: ValidationInfo
    ) -> dict[str, Any]:
        """Validate the result of a gradient endpoint computation.

        Since the structure of the result is already validated in core.py, we only need to check the shapes
        to ensure they match what's expected from the schema.
        """
        if gradient_type == "jacobian":
            if set(info.context["output_keys"]) != set(result.keys()):
                raise ValueError(
                    "Error when validating output of jacobian:\n"
                    f"Expected keys {info.context['output_keys']} in output; got {set(result.keys())}"
                )
            for output_path, subout in result.items():
                if set(info.context["input_keys"]) != set(subout.keys()):
                    raise ValueError(
                        "Error when validating output of jacobian:\n"
                        "Expected output with structure "
                        f"{{{tuple(info.context['output_keys'])}: {{{tuple(info.context['input_keys'])}: ...}}}}, "
                        f"got {set(subout.keys())} for output key {output_path}."
                    )
                output_shape = _find_shape_from_path(
                    diffable_output_patterns, output_path
                )
                for input_path, arr in subout.items():
                    input_shape = _find_shape_from_path(
                        diffable_input_patterns, input_path
                    )

                    if output_shape is Ellipsis and input_shape is Ellipsis:
                        # Everything goes
                        continue
                    elif output_shape is Ellipsis:
                        expected_shape = (
                            *arr.shape[: len(arr.shape) - len(input_shape)],
                            *input_shape,
                        )
                    elif input_shape is Ellipsis:
                        expected_shape = (
                            *output_shape,
                            *arr.shape[len(output_shape) :],
                        )
                    else:
                        expected_shape = (*output_shape, *input_shape)

                    # No dtype check: GPU arrays pass through validation unconverted,
                    # and their dtype (e.g. torch.float32) need not be a NumPy name.
                    try:
                        result[output_path][input_path] = _exact_array_adapter(
                            expected_shape, None
                        ).validate_python(arr, context=info.context)
                    except ValidationError as e:
                        raise ValueError(
                            f"Jacobian result [{output_path}][{input_path}]: {e}"
                        ) from e

        elif gradient_type == "jvp":
            if set(info.context["output_keys"]) != set(result.keys()):
                raise ValueError(
                    f"Expected keys {info.context['output_keys']} in output; got {set(result.keys())}"
                )
            for output_path, arr in result.items():
                expected_annotation = _find_annotation_from_path(
                    diffable_output_patterns, output_path
                )
                try:
                    result[output_path] = _type_adapter(
                        expected_annotation
                    ).validate_python(arr, context=info.context)
                except ValidationError as e:
                    raise ValueError(f"JVP result [{output_path}]: {e}") from e

        elif gradient_type == "vjp":
            if set(info.context["input_keys"]) != set(result.keys()):
                raise ValueError(
                    f"Expected keys {info.context['input_keys']} in output; got {set(result.keys())}"
                )
            for input_path, arr in result.items():
                expected_annotation = _find_annotation_from_path(
                    diffable_input_patterns, input_path
                )
                try:
                    result[input_path] = _type_adapter(
                        expected_annotation
                    ).validate_python(arr, context=info.context)
                except ValidationError as e:
                    raise ValueError(f"VJP result [{input_path}]: {e}") from e

        return result

    # NOTE (dh): Literal[somevar] is marked as an error by Pylance, but this doesn't throw a runtime error, so I'd
    # prefer to keep this instead of introducing more arcane hacks to create Literal types at runtime.

    if gradient_type == "jacobian":

        class JacobianInputSchema(BaseModel):
            inputs: InputSchema = Field(
                ..., description="The input data to compute the Jacobian at."
            )
            jac_inputs: set[
                Annotated[diffable_input_type, AfterValidator(input_path_validator)]
            ] = Field(
                ...,
                description="The set of differentiable inputs to compute the Jacobian with respect to.",
                min_length=1,
            )
            jac_outputs: set[diffable_output_type] = Field(
                ...,
                description="The set of differentiable outputs to compute the Jacobian of.",
                min_length=1,
            )

            model_config = ConfigDict(extra="forbid")

        class JacobianOutputSchema(RootModel):
            root: Annotated[
                dict[diffable_output_type, dict[diffable_input_type, Array[..., None]]],
                AfterValidator(result_validator),
            ] = Field(
                ...,
                description=(
                    "Container for the results of Jacobian computations. "
                    "The result represents a nested structure of the Jacobian matrix "
                    "as a mapping with structure ``{jac_outputs: {jac_inputs: array}}``. "
                    "The shape of each array is the concatenation of the shapes of the output and input arrays, "
                    "i.e. ``(*output_array.shape, *input_array.shape)``."
                ),
            )

        InputSchema = JacobianInputSchema
        OutputSchema = JacobianOutputSchema

    elif gradient_type == "jvp":

        class JVPInputSchema(BaseModel):
            inputs: InputSchema = Field(
                ..., description="The input data to compute the JVP at."
            )
            jvp_inputs: set[
                Annotated[diffable_input_type, AfterValidator(input_path_validator)]
            ] = Field(
                ...,
                description="The set of differentiable inputs to compute the JVP with respect to.",
                min_length=1,
            )
            jvp_outputs: set[diffable_output_type] = Field(
                ...,
                description="The set of differentiable outputs to compute the JVP of.",
                min_length=1,
            )
            tangent_vector: dict[diffable_input_type, Array[..., None]] = Field(
                ...,
                description=(
                    "Tangent vector to multiply the Jacobian with. "
                    "Expected to be a mapping with structure ``{jvp_inputs: array}``. "
                    "The shape of each array is the same as the shape of the corresponding input array."
                ),
            )
            model_config = ConfigDict(extra="forbid")

            @field_validator("tangent_vector", mode="after")
            @classmethod
            def validate_tangent_vector(
                cls, tangent_vector: dict, info: ValidationInfo
            ) -> dict:
                """Validate tangent vector keys, shapes, and dtypes."""
                try:
                    if set(tangent_vector.keys()) != info.data["jvp_inputs"]:
                        raise ValueError(
                            f"Expected tangent vector with keys conforming to jvp_inputs: {info.data['jvp_inputs']}, "
                            f"got {set(tangent_vector.keys())}."
                        )
                except KeyError as e:
                    raise ValueError(
                        "Unable to validate tangent vector as jvp_inputs either missing or invalid"
                    ) from e

                if "inputs" not in info.data:
                    raise ValueError(
                        "Unable to validate tangent vector as inputs either missing or invalid"
                    )

                validated = {}
                for path, arr in tangent_vector.items():
                    try:
                        ref_arr = get_at_path(info.data["inputs"], path)
                    except (LookupError, AttributeError) as e:
                        raise ValueError(
                            f"Could not find tangent_vector path {path} in inputs."
                        ) from e
                    if hasattr(ref_arr, "shape"):
                        ref_shape = ref_arr.shape
                    elif isinstance(ref_arr, dict):
                        ref_shape = tuple(ref_arr["shape"])
                    else:
                        # Scalar (e.g. Differentiable[Float32])
                        ref_shape = ()
                    annotation = _find_annotation_from_path(
                        diffable_input_patterns, path
                    )
                    try:
                        validated[path] = _exact_array_adapter(
                            tuple(ref_shape), annotation.expected_dtype
                        ).validate_python(arr, context=info.context)
                    except ValidationError as e:
                        raise ValueError(f"Tangent vector '{path}': {e}") from e
                return validated

        class JVPOutputSchema(RootModel):
            root: Annotated[
                dict[diffable_output_type, Array[..., None]],
                AfterValidator(result_validator),
            ] = Field(
                ...,
                description=(
                    "Container for the results of Jacobian-vector products. "
                    "The result is a mapping with structure ``{jvp_outputs: array}``. "
                    "The shape of each array is the same as the shape of the corresponding output array."
                ),
            )

        InputSchema = JVPInputSchema
        OutputSchema = JVPOutputSchema

    elif gradient_type == "vjp":

        class VJPInputSchema(BaseModel):
            inputs: InputSchema = Field(
                ..., description="The input data to compute the VJP at."
            )
            vjp_inputs: set[
                Annotated[diffable_input_type, AfterValidator(input_path_validator)]
            ] = Field(
                ...,
                description="The set of differentiable inputs to compute the VJP with respect to.",
                min_length=1,
            )
            vjp_outputs: set[diffable_output_type] = Field(
                ...,
                description="The set of differentiable outputs to compute the VJP of.",
                min_length=1,
            )
            cotangent_vector: dict[diffable_output_type, Array[..., None]] = Field(
                ...,
                description=(
                    "Cotangent vector to multiply the Jacobian with. "
                    "Expected to be a mapping with structure ``{vjp_outputs: array}``. "
                    "The shape of each array is the same as the shape of the corresponding output array."
                ),
            )
            model_config = ConfigDict(extra="forbid")

            @field_validator("cotangent_vector", mode="after")
            @classmethod
            def validate_cotangent_vector(
                cls, cotangent_vector: dict, info: ValidationInfo
            ) -> dict:
                """Validate cotangent vector keys, shapes, and dtypes."""
                try:
                    if set(cotangent_vector.keys()) != info.data["vjp_outputs"]:
                        raise ValueError(
                            "Expected cotangent vector with keys conforming to vjp_outputs: "
                            f"{info.data['vjp_outputs']}, got {set(cotangent_vector.keys())}."
                        )
                except KeyError as e:
                    raise ValueError(
                        "Unable to validate cotangent vector as vjp_outputs either missing or invalid"
                    ) from e

                validated = {}
                for path, arr in cotangent_vector.items():
                    annotation = _find_annotation_from_path(
                        diffable_output_patterns, path
                    )
                    try:
                        validated[path] = _type_adapter(annotation).validate_python(
                            arr, context=info.context
                        )
                    except ValidationError as e:
                        raise ValueError(f"Cotangent vector '{path}': {e}") from e
                return validated

        class VJPOutputSchema(RootModel):
            root: Annotated[
                dict[diffable_input_type, Array[..., None]],
                AfterValidator(result_validator),
            ] = Field(
                ...,
                description=(
                    "Container for the results of vector-Jacobian products. "
                    "The result is a mapping with structure ``{vjp_inputs: array}``. "
                    "The shape of each array is the same as the shape of the corresponding input array."
                ),
            )

        InputSchema = VJPInputSchema
        OutputSchema = VJPOutputSchema

    return InputSchema, OutputSchema
