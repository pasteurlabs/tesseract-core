# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import copy
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationInfo,
    create_model,
    field_validator,
)

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


def apply_function_to_model_tree(
    Schema: type[BaseModel],
    func: Callable[[type, tuple], type],
    model_prefix: str = "",
    default_model_config: dict[str, Any] | None = None,
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
    if default_model_config is None:
        default_model_config = {}

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

            # Only override model_config if it is not already present
            # in the pydantic model definition
            model_config = ConfigDict(**default_model_config)
            model_config.update(treeobj.model_config)

            # Make sure that reconstructed RootModels don't contain unsupported 'extra' config.
            if safe_issubclass(treeobj, RootModel):
                model_config.pop("extra", None)

            return create_model(
                f"{model_prefix}{treeobj.__name__}",
                **new_fields,
                model_config=(ConfigDict, model_config),
                __base__=treeobj,
            )

        elif _is_annotated(treeobj):
            # Recurse into Annotated types by stripping off right-most (most recent) metadata

            # Using __origin__ and __metadata__ instead of get_args for compatibility with typing_extensions.Annotated
            inner_type = treeobj.__origin__
            *other_meta, current_meta = treeobj.__metadata__

            # Recurse into Annotated[inner_type, *other_meta] and re-attach current_meta to the result
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

            return Union[tuple(newargs)]  # noqa: UP007

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

        # Replace sentinel values with indexing syntax
        str_parts = []
        for part in pathtuple:
            if part is SEQ_INDEX_SENTINEL:
                str_parts.append("[]")
            elif part is DICT_INDEX_SENTINEL:
                str_parts.append("{}")
            else:
                str_parts.append(part)

        serialized[".".join(str_parts)] = {
            "shape": shape,
            "dtype": dtype,
        }

    return serialized


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
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"differentiable_arrays": differentiable_arrays},
        )

    class ApplyOutputSchema(RootModel):
        root: OutputSchema = Field(
            ..., description="The output data from applying the Tesseract."
        )
        differentiable_arrays: ClassVar[dict[str, dict[str, Any]]] = (
            _serialize_diffable_arrays(diffable_output_paths)
        )
        model_config = ConfigDict(
            json_schema_extra={"differentiable_arrays": differentiable_arrays}
        )

    return ApplyInputSchema, ApplyOutputSchema


def create_abstract_eval_schema(
    InputSchema: type[BaseModel], OutputSchema: type[BaseModel]
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Create new Pydantic models that represent abstract versions of the given schemas.

    The returned schemas will have the same structure as the input schemas, but with array
    fields replaced by ShapeDType objects.
    """

    def replace_array_with_shapedtype(obj: T, _: Any) -> T | type[ShapeDType]:
        if safe_issubclass(obj, PydanticArrayAnnotation):
            return ShapeDType.from_array_type(obj)
        return obj

    GeneratedInputSchema = apply_function_to_model_tree(
        InputSchema,
        replace_array_with_shapedtype,
        model_prefix="AbstractEval_",
        default_model_config=dict(extra="forbid"),
    )

    GeneratedOutputSchema = apply_function_to_model_tree(
        OutputSchema,
        replace_array_with_shapedtype,
        model_prefix="AbstractEval_",
        default_model_config=dict(extra="forbid"),
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
    """Return a dictionary mapping path patterns to their types.

    For containers (dict, list), the path will include sentinels and map to the container's inner type.
    For model fields, maps to the field's type.
    For leaves (primitives, arrays), maps to the actual type.

    Args:
        schema: The Pydantic model schema to traverse.
        filter_fn: Optional predicate to filter which paths to include.
                   If None, includes all paths. If provided, only includes
                   paths where filter_fn(type) returns True.
    """
    path_to_type = {}

    def add_path(obj: T, path: tuple) -> T:
        if filter_fn is None or filter_fn(obj):
            path_to_type[path] = obj
        return obj

    apply_function_to_model_tree(schema, add_path)
    return path_to_type


def _path_to_pattern(path: Sequence[str | object]) -> str:
    """Return a type describing valid paths for all passed paths."""
    # Check if path includes sequence or dict indexing -- in this case, we can't use the
    # path as a literal and need to use a regex pattern instead.
    final_path = []
    is_literal = True
    for part in path:
        if part is SEQ_INDEX_SENTINEL:
            is_literal = False
            part = r"\[-?\d+\]"
        elif part is DICT_INDEX_SENTINEL:
            is_literal = False
            part = r"\{[\w \-]+\}"
        final_path.append(part)

    if is_literal:
        return ".".join(final_path)

    return r"\.".join(final_path)


def _pattern_to_type(pattern: str) -> type:
    """Convert a string pattern (which may be a literal or regex) to a type that can be used for validation."""
    if _is_regex_pattern(pattern):
        return Annotated[str, Field(pattern=pattern)]
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


def create_autodiff_schema(
    InputSchema: type[BaseModel],
    OutputSchema: type[BaseModel],
    ad_flavor: Literal["jacobian", "jvp", "vjp"],
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Generate the input / outputs schemas for AD endpoints like /jacobian, /jacobian_vector_product, etc.

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

    def _find_shape_dtype_from_path(path_patterns: dict, concrete_path: str):
        """Find the expected shape and dtype for a given concrete path from path patterns.

        Returns:
            A tuple of (expected_shape, expected_dtype) where expected_shape can be
            a tuple or Ellipsis, and expected_dtype can be a string or None.
        """
        for path_pattern, array_type in path_patterns.items():
            if _is_regex_pattern(path_pattern):
                path_matches = bool(re.match(path_pattern, concrete_path))
            else:
                path_matches = path_pattern == concrete_path

            if path_matches:
                return array_type.expected_shape, array_type.expected_dtype

        raise ValueError(f"Invalid path: {concrete_path}")

    def _find_shape_from_path(path_patterns: dict, concrete_path: str) -> tuple:
        shape, _ = _find_shape_dtype_from_path(path_patterns, concrete_path)
        return shape

    def _validate_and_cast_vector(
        vector: dict,
        reference_data: Any,
        path_patterns: dict,
        vector_name: str,
    ) -> dict:
        """Validate shape and cast dtype for tangent/cotangent vectors.

        Args:
            vector: The tangent or cotangent vector dict (key -> array).
            reference_data: The reference data to compare shapes against (inputs model or dict).
            path_patterns: The path patterns dict with expected shapes/dtypes from the schema.
            vector_name: Name of the vector for error messages ("tangent" or "cotangent").

        Returns:
            The vector dict with dtypes cast to float if needed.
        """
        validated_vector = {}
        for path, arr in vector.items():
            # Get reference array from inputs to compare shapes
            try:
                ref_arr = get_at_path(reference_data, path)
            except (LookupError, AttributeError) as exc:
                raise ValueError(
                    f"Could not find {vector_name} vector path '{path}' in reference data."
                ) from exc

            # Handle both numpy arrays and encoded array dicts
            if hasattr(ref_arr, "shape"):
                ref_shape = ref_arr.shape
                ref_dtype = ref_arr.dtype
            else:
                # Encoded array dict
                ref_shape = tuple(ref_arr["shape"])
                ref_dtype = np.dtype(ref_arr["dtype"])

            if hasattr(arr, "shape"):
                arr_shape = arr.shape
                arr_dtype = arr.dtype
            else:
                arr_shape = tuple(arr["shape"])
                arr_dtype = np.dtype(arr["dtype"])

            # Validate shape matches
            if arr_shape != ref_shape:
                raise ValueError(
                    f"{vector_name.title()} vector '{path}' has shape {arr_shape}, "
                    f"but expected shape {ref_shape} (matching the corresponding array)."
                )

            # Cast integer dtypes to float for AD compatibility
            # JAX/AD frameworks require float dtypes for differentiation
            if np.issubdtype(arr_dtype, np.integer):
                target_dtype = (
                    ref_dtype if np.issubdtype(ref_dtype, np.floating) else np.float64
                )
                if hasattr(arr, "astype"):
                    arr = arr.astype(target_dtype)
                else:
                    # For encoded arrays, update the dtype field
                    arr = dict(arr)
                    arr["dtype"] = str(np.dtype(target_dtype))
            # Also cast if dtype doesn't match reference (and both are floating)
            elif np.issubdtype(arr_dtype, np.floating) and np.issubdtype(
                ref_dtype, np.floating
            ):
                if arr_dtype != ref_dtype:
                    if hasattr(arr, "astype"):
                        arr = arr.astype(ref_dtype)
                    else:
                        arr = dict(arr)
                        arr["dtype"] = str(ref_dtype)

            validated_vector[path] = arr

        return validated_vector

    def _validate_and_cast_vector_from_schema(
        vector: dict,
        path_patterns: dict,
        vector_name: str,
    ) -> dict:
        """Validate shape and cast dtype for cotangent vectors using schema patterns.

        This is used for VJP cotangent vectors where we don't have actual output data
        to compare against, only the schema-defined shapes and dtypes.

        Args:
            vector: The cotangent vector dict (key -> array).
            path_patterns: The path patterns dict with expected shapes/dtypes from the schema.
            vector_name: Name of the vector for error messages ("cotangent").

        Returns:
            The vector dict with dtypes cast to float if needed.
        """
        validated_vector = {}
        for path, arr in vector.items():
            # Get expected shape and dtype from schema patterns
            expected_shape, expected_dtype = _find_shape_dtype_from_path(
                path_patterns, path
            )

            if hasattr(arr, "shape"):
                arr_shape = arr.shape
                arr_dtype = arr.dtype
            else:
                arr_shape = tuple(arr["shape"])
                arr_dtype = np.dtype(arr["dtype"])

            # Validate shape matches (if not Ellipsis)
            if expected_shape is not Ellipsis:
                # Check dimensionality
                if len(arr_shape) != len(expected_shape):
                    raise ValueError(
                        f"{vector_name.title()} vector '{path}' has {len(arr_shape)} dimensions, "
                        f"but expected {len(expected_shape)} dimensions."
                    )
                # Check each dimension (None means polymorphic/any size)
                for actual, expected in zip(arr_shape, expected_shape, strict=True):
                    if expected is not None and actual != expected:
                        raise ValueError(
                            f"{vector_name.title()} vector '{path}' has shape {arr_shape}, "
                            f"but expected shape {expected_shape}."
                        )

            # Cast integer dtypes to float for AD compatibility
            if np.issubdtype(arr_dtype, np.integer):
                if expected_dtype is not None and np.issubdtype(
                    np.dtype(expected_dtype), np.floating
                ):
                    target_dtype = np.dtype(expected_dtype)
                else:
                    target_dtype = np.float64
                if hasattr(arr, "astype"):
                    arr = arr.astype(target_dtype)
                else:
                    arr = dict(arr)
                    arr["dtype"] = str(target_dtype)
            # Cast to expected dtype if specified and different
            elif expected_dtype is not None:
                expected_np_dtype = np.dtype(expected_dtype)
                if arr_dtype != expected_np_dtype and np.issubdtype(
                    arr_dtype, np.floating
                ):
                    if hasattr(arr, "astype"):
                        arr = arr.astype(expected_np_dtype)
                    else:
                        arr = dict(arr)
                        arr["dtype"] = expected_dtype

            validated_vector[path] = arr

        return validated_vector

    InputSchema = apply_function_to_model_tree(
        InputSchema,
        lambda x, _: x,
        model_prefix=f"{ad_flavor.title()}_",
        default_model_config=dict(extra="forbid"),
    )

    def result_validator(
        result: dict[str, Any], info: ValidationInfo
    ) -> dict[str, Any]:
        """Validate the result of a Jacobian computation.

        Since the structure of the result is already validated in core.py, we only need to check the shapes
        to ensure they match what's expected from the schema.
        """
        if ad_flavor == "jacobian":
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
                        for s1, s2 in zip(got_shape, expected_shape, strict=True)
                        if s2 is not None
                    ):
                        raise ValueError(
                            f"Jacobian result [{output_path}][{input_path}]: "
                            f"Expected shape {expected_shape}, got {got_shape}"
                        )

        elif ad_flavor == "jvp":
            if set(info.context["output_keys"]) != set(result.keys()):
                raise ValueError(
                    f"Expected keys {info.context['output_keys']} in output; got {set(result.keys())}"
                )
            for output_path, arr in result.items():
                output_shape = _find_shape_from_path(
                    diffable_output_patterns, output_path
                )
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
                    for s1, s2 in zip(arr.shape, expected_shape, strict=True)
                    if s2 is not None
                ):
                    raise ValueError(
                        f"JVP result [{output_path}]: Expected shape {expected_shape}, got {arr.shape}"
                    )

        elif ad_flavor == "vjp":
            if set(info.context["input_keys"]) != set(result.keys()):
                raise ValueError(
                    f"Expected keys {info.context['input_keys']} in output; got {set(result.keys())}"
                )
            for input_path, arr in result.items():
                input_shape = _find_shape_from_path(diffable_input_patterns, input_path)
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
                    for s1, s2 in zip(arr.shape, expected_shape, strict=True)
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

    elif ad_flavor == "jvp":

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
                # Tangent vector needs same keys as jvp_inputs
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

                # Validate shapes and cast dtypes
                return _validate_and_cast_vector(
                    tangent_vector,
                    info.data["inputs"],
                    diffable_input_patterns,
                    "tangent",
                )

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
                            "Expected cotangent vector with keys conforming to vjp_outputs:",
                            f"{info.data['vjp_outputs']}, got {set(cotangent_vector.keys())}.",
                        )
                except KeyError as e:
                    raise ValueError(
                        "Unable to validate cotangent vector as vjp_outputs either missing or invalid"
                    ) from e

                # Validate shapes and cast dtypes using schema patterns
                # (we don't have actual outputs to compare against)
                return _validate_and_cast_vector_from_schema(
                    cotangent_vector,
                    diffable_output_patterns,
                    "cotangent",
                )

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
