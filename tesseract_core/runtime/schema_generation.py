# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import sys
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import copy
from typing import (
    Annotated,
    Any,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    create_model,
)

from .schema_types import (
    Array,
    PydanticArrayAnnotation,
    ShapeDType,
    _is_annotated,
    is_array_annotation,
    is_differentiable,
    safe_issubclass,
)

# Constants to mark sequence and dict indexing in pytree paths
SEQ_INDEX_SENTINEL = object()
DICT_INDEX_SENTINEL = object()

T = TypeVar("T")

# Python has funnily enough two union types now. See https://github.com/python/cpython/issues/105499
# We check against both for compatibility with older versions of Python.
UNION_TYPES = [Union]
if sys.version_info >= (3, 10):
    UNION_TYPES += [types.UnionType]


def _construct_annotated(obj: Any, metadata: Iterable[Any]) -> Any:
    """Construct an Annotated type with the given metadata."""
    out = obj
    # Exploit that repeated Annotated applications are flattened
    for meta in metadata:
        out = Annotated[out, meta]
    return out


def apply_function_to_model_tree(
    Schema: type[BaseModel],
    func: Callable[[type, tuple], type],
    model_prefix: str = "",
    model_kwargs: Any = None,
) -> type[BaseModel]:
    """Apply a function to all leaves of a Pydantic model, recursing into containers + nested models.

    A leaf is any type annotation that is not a Pydantic model or a container type.

    The given function should take two arguments: the type annotation and the path to it
    in the model tree (as a list of field names). It should return the new type annotation
    for the field, or None to remove the field from the model.

    The notion of path is used to handle nested models and containers. For example, given
    a model like this:

    class MyModel(BaseModel):
        a: list[dict[str, int]]

    The path to the field "a" would be ["a"], and the path to the int type would be
    ["a", SEQ_INDEX_SENTINEL, DICT_INDEX_SENTINEL].
    """
    # Annotation types that should be treated as leaves and not recursed into
    annotated_types_as_leaves = (PydanticArrayAnnotation,)

    if model_kwargs is None:
        model_kwargs = {}

    seen_models = set()

    def _recurse_over_model_tree(treeobj: Any, path: list[str]) -> Any:
        # Get the origin type of the annotation, e.g. List for List[int]
        origin_type = get_origin(treeobj)
        deprecated_types = ["List", "Dict", "Set", "FrozenSet", "Tuple"]

        if origin_type is not None and origin_type.__name__ in deprecated_types:
            # Raise error if the user tries to use a deprecated type
            raise TypeError(
                f"'{origin_type}' annotation deprecated. Use '{origin_type.lower()}' instead."
            )

        if safe_issubclass(treeobj, BaseModel):
            # Recurse into (nested) pydantic models

            if id(treeobj) in seen_models:
                raise ValueError(
                    f"Recursive model definition detected at path {path}: {treeobj.__name__}. "
                    "Recursive models are not supported as Tesseract schemas."
                )
            seen_models.add(id(treeobj))

            new_fields = {}

            for field_name, field in treeobj.model_fields.items():
                field_type = field.annotation
                if field.metadata:
                    field_type = _construct_annotated(field_type, field.metadata)

                field_path = [*path, field_name]
                new_type = _recurse_over_model_tree(field_type, field_path)
                if new_type is None:
                    continue

                new_field = copy(field)
                # Need to strip off metadata to trigger re-evaluation of the field
                new_field.metadata = []

                new_fields[field_name] = (new_type, new_field)

            # We only forbid encountering the same model twice if it is within the same subtree
            seen_models.remove(id(treeobj))

            return create_model(
                f"{model_prefix}{treeobj.__name__}",
                **new_fields,
                **model_kwargs,
                __base__=treeobj,
            )

        elif _is_annotated(treeobj):
            # Recurse into Annotated types by stripping off right-most (most recent) metadata

            # Using __origin__ and __metadata__ instead of get_args for compatibility with typing_extensions.Annotated
            inner_type = treeobj.__origin__
            *other_meta, current_meta = treeobj.__metadata__

            is_leaf = False
            for t in annotated_types_as_leaves:
                if safe_issubclass(current_meta, t):
                    is_leaf = True
                    break

            if is_leaf:
                # Reached a leaf -> apply func to object as-is, then re-attach add'l metadata
                current_annotation = Annotated[inner_type, current_meta]
                out = func(current_annotation, tuple(path))
                out = _construct_annotated(out, other_meta)
                return out

            # Not a leaf -> recurse into Annotated[inner_type, *other_meta] and re-attach current_meta to the result
            inner_type = _construct_annotated(inner_type, other_meta)
            new_type = _recurse_over_model_tree(inner_type, path)

            if new_type is None:
                return None

            return Annotated[new_type, current_meta]

        elif any(origin_type is t for t in UNION_TYPES):
            # Recurse into Union and Optional (which is a subclass of Union)
            args = get_args(treeobj)
            newargs = [_recurse_over_model_tree(arg, path) for arg in args]
            newargs = [arg for arg in newargs if arg is not None]
            if not newargs:
                return None

            return Union[tuple(newargs)]

        elif safe_issubclass(origin_type, Mapping):
            # Recurse into dict-likes
            args = get_args(treeobj)
            if not args:
                return origin_type

            key_type, value_type = args

            # Append sentinel to path to indicate dict indexing occurred
            new_path = [*path, DICT_INDEX_SENTINEL]

            new_type = _recurse_over_model_tree(value_type, new_path)
            if new_type is None:
                return None

            return origin_type[key_type, new_type]

        elif safe_issubclass(origin_type, Iterable):
            # Recurse into List, Set, Tuple
            value_type = get_args(treeobj)
            if not value_type:
                return origin_type

            # Append sentinel to path to indicate sequence indexing occurred
            new_path = [*path, SEQ_INDEX_SENTINEL]

            new_types = [_recurse_over_model_tree(vt, new_path) for vt in value_type]
            new_types = [nt for nt in new_types if nt is not None]
            if not new_types:
                return None

            return origin_type[tuple(new_types)]

        # Reached a leaf -> apply func
        return func(treeobj, tuple(path))

    return _recurse_over_model_tree(Schema, [])


def create_apply_schema(
    InputSchema: type[BaseModel], OutputSchema: type[BaseModel]
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Create the input / output schemas for the /apply endpoint."""
    InputSchema = apply_function_to_model_tree(
        InputSchema,
        lambda x, _: x,
        model_prefix="Apply_",
        model_kwargs={"model_config": ConfigDict(extra="forbid")},
    )
    OutputSchema = apply_function_to_model_tree(
        OutputSchema,
        lambda x, _: x,
        model_prefix="Apply_",
        model_kwargs={"model_config": ConfigDict(extra="forbid")},
    )

    class ApplyInputSchema(BaseModel):
        inputs: InputSchema = Field(
            ..., description="The input data to apply the Tesseract to."
        )

        model_config = ConfigDict(extra="forbid")

    class ApplyOutputSchema(RootModel):
        root: OutputSchema = Field(
            ..., description="The output data from applying the Tesseract."
        )

    return ApplyInputSchema, ApplyOutputSchema


def create_abstract_eval_schema(
    InputSchema: type[BaseModel], OutputSchema: type[BaseModel]
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Create new Pydantic models that represent abstract versions of the given schemas.

    The returned schemas will have the same structure as the input schemas, but with array
    fields replaced by ShapeDType objects.
    """

    def replace_array_with_shapedtype(obj: T, _: Any) -> Union[T, type[ShapeDType]]:
        if is_array_annotation(obj):
            return ShapeDType.from_array_annotation(obj)
        return obj

    GeneratedInputSchema = apply_function_to_model_tree(
        InputSchema,
        replace_array_with_shapedtype,
        model_prefix="AbstractEval_",
        model_kwargs={"model_config": ConfigDict(extra="forbid")},
    )

    GeneratedOutputSchema = apply_function_to_model_tree(
        OutputSchema,
        replace_array_with_shapedtype,
        model_prefix="AbstractEval_",
        model_kwargs={"model_config": ConfigDict(extra="forbid")},
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


def _get_diffable_arrays(schema: type[BaseModel]) -> dict[str, Any]:
    """Return a dictionary mapping path patterns of differentiable arrays to their types."""
    diffable_paths = {}

    def add_to_dict_if_diffable(obj: T, path: tuple) -> T:
        if is_differentiable(obj):
            pattern = _path_to_pattern(path)
            diffable_paths[pattern] = obj
        return obj

    apply_function_to_model_tree(schema, add_to_dict_if_diffable)
    return diffable_paths


def _path_to_pattern(path: Sequence[Union[str, object]]) -> str:
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
            part = r"\{.*?\}"
        final_path.append(part)

    if is_literal:
        return ".".join(final_path)

    return r"\.".join(final_path)


def _pattern_to_type(pattern: str) -> type:
    """Convert a string pattern (which may be a literal or regex) to a type that can be used for validation."""
    if _is_regex_pattern(pattern):
        return Annotated[str, Field(pattern=pattern)]

    return Literal[pattern]


def _is_regex_pattern(pattern: str) -> bool:
    # Poor man's regex detection, but it should work for our purposes
    return "\\" in pattern


def create_autodiff_schema(
    InputSchema: type[BaseModel],
    OutputSchema: type[BaseModel],
    ad_flavor: Literal["jacobian", "jvp", "vjp"],
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Generate the input / outputs schemas for AD endpoints like /jacobian, /jacobian_vector_product, etc.

    Returns a tuple (InputSchema, OutputSchema) with the generated schemas.
    """
    diffable_inputs = _get_diffable_arrays(InputSchema)
    diffable_outputs = _get_diffable_arrays(OutputSchema)

    if not diffable_inputs:
        raise RuntimeError("No differentiable inputs found in the input schema")

    if not diffable_outputs:
        raise RuntimeError("No differentiable outputs found in the output schema")

    diffable_input_type = Union[
        tuple(_pattern_to_type(p) for p in diffable_inputs.keys())
    ]
    diffable_output_type = Union[
        tuple(_pattern_to_type(p) for p in diffable_outputs.keys())
    ]

    def _find_shape_from_path(container: dict, concrete_path: str) -> tuple:
        for path_pattern, array_type in container.items():
            if _is_regex_pattern(path_pattern):
                path_matches = bool(re.match(path_pattern, concrete_path))
            else:
                path_matches = path_pattern == concrete_path

            if path_matches:
                return array_type.__metadata__[0].expected_shape

        raise ValueError(f"Invalid path: {concrete_path}")

    InputSchema = apply_function_to_model_tree(
        InputSchema,
        lambda x, _: x,
        model_prefix=f"{ad_flavor.title()}_",
        model_kwargs={"model_config": ConfigDict(extra="forbid")},
    )

    def result_validator(
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate the result of a Jacobian computation.

        Since the structure of the result is already validated in core.py, we only need to check the shapes
        to ensure they match what's expected from the schema.
        """
        if ad_flavor == "jacobian":
            for output_path, subout in result.items():
                output_shape = _find_shape_from_path(diffable_outputs, output_path)
                for input_path, arr in subout.items():
                    input_shape = _find_shape_from_path(diffable_inputs, input_path)

                    if output_shape is Ellipsis and input_shape is Ellipsis:
                        # Everything goes
                        continue
                    elif output_shape is Ellipsis:
                        expected_shape = (
                            *input_shape,
                            arr.shape[-len(input_shape) :],
                        )
                    elif input_shape is Ellipsis:
                        expected_shape = (
                            *output_shape,
                            *arr.shape[: len(output_shape)],
                        )
                    else:
                        expected_shape = (*output_shape, *input_shape)

                    # We allow both encoded and decoded arrays as schema output
                    if hasattr(arr, "shape"):
                        got_shape = arr.shape
                    else:
                        got_shape = arr["shape"]

                    if len(got_shape) != len(expected_shape):
                        raise ValueError(
                            f"Jacobian result [{output_path}][{input_path}]: "
                            f"Expected shape {expected_shape}, got {got_shape}"
                        )

                    if any(
                        s1 != s2
                        for s1, s2 in zip(got_shape, expected_shape)
                        if s2 is not None
                    ):
                        raise ValueError(
                            f"Jacobian result [{output_path}][{input_path}]: "
                            f"Expected shape {expected_shape}, got {got_shape}"
                        )

        elif ad_flavor == "jvp":
            for output_path, arr in result.items():
                output_shape = _find_shape_from_path(diffable_outputs, output_path)
                if output_shape is Ellipsis:
                    # Everything goes
                    continue

                expected_shape = output_shape

                if len(arr.shape) != len(expected_shape):
                    raise ValueError(
                        f"JVP result [{output_path}]: Expected shape {expected_shape}, got {arr.shape}"
                    )

                if any(
                    s1 != s2
                    for s1, s2 in zip(arr.shape, expected_shape)
                    if s2 is not None
                ):
                    raise ValueError(
                        f"JVP result [{output_path}]: Expected shape {expected_shape}, got {arr.shape}"
                    )

        elif ad_flavor == "vjp":
            for input_path, arr in result.items():
                input_shape = _find_shape_from_path(diffable_inputs, input_path)
                if input_shape is Ellipsis:
                    # Everything goes
                    continue

                expected_shape = input_shape

                if len(arr.shape) != len(expected_shape):
                    raise ValueError(
                        f"VJP result [{input_path}]: Expected shape {expected_shape}, got {arr.shape}"
                    )

                if any(
                    s1 != s2
                    for s1, s2 in zip(arr.shape, expected_shape)
                    if s2 is not None
                ):
                    raise ValueError(
                        f"VJP result [{input_path}]: Expected shape {expected_shape}, got {arr.shape}"
                    )

        return result

    # NOTE (dh): Literal[somevar] is marked as an error by Pylance, but this doesn't throw a runtime error, so I'd
    # prefer to keep this instead of introducing more arcane hacks to create Literal types at runtime.

    if ad_flavor == "jacobian":

        class JacobianInputSchema(BaseModel):
            inputs: InputSchema = Field(
                ..., description="The input data to compute the Jacobian at."
            )
            jac_inputs: set[diffable_input_type] = Field(
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

    elif ad_flavor == "jvp":

        class JVPInputSchema(BaseModel):
            inputs: InputSchema = Field(
                ..., description="The input data to compute the JVP at."
            )
            jvp_inputs: set[diffable_input_type] = Field(
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

    elif ad_flavor == "vjp":

        class VJPInputSchema(BaseModel):
            inputs: InputSchema = Field(
                ..., description="The input data to compute the VJP at."
            )
            vjp_inputs: set[diffable_input_type] = Field(
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
