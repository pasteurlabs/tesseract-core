# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import (
    Annotated,
    Any,
    get_args,
    get_origin,
)

import numpy as np
from pydantic import (
    AfterValidator,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    TypeAdapter,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator, core_schema

from tesseract_core.runtime.file_interactions import PathLike, parent_path
from tesseract_core.runtime.mpa import (
    log_artifact,
    log_metric,
    log_parameter,
)
from tesseract_core.runtime.schema_types import safe_issubclass

# Finite difference utilities for automatic differentiation
# These provide a simple way to make any Tesseract differentiable without
# implementing analytical gradients. Note: These are experimental and the API
# may change in future releases.
from tesseract_core.runtime.testing.finite_differences import (
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
)
from tesseract_core.runtime.tree_transforms import get_at_path

# Autodiff fallback utilities for deriving missing autodiff endpoints from existing ones.
# These are experimental and the API may change in future releases.


def vjp_from_jacobian(
    jacobian_fn: Callable,
    inputs: Any,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
    diagonal: bool = False,
) -> dict[str, Any]:
    """Compute VJP as v^T @ J using the full Jacobian.

    Args:
        jacobian_fn: The api_module.jacobian callable.
        inputs: Validated InputSchema instance.
        vjp_inputs: set[str] of input path strings to differentiate w.r.t.
        vjp_outputs: set[str] of output path strings to differentiate.
        cotangent_vector: dict mapping output paths to cotangent arrays.
        diagonal: If True, assume each Jacobian block is diagonal (i.e. all
            off-diagonal entries are zero) and compute the product using an
            elementwise multiply against the diagonal instead of a full
            tensordot. This is the correct choice when the primal computation
            consists entirely of elementwise or in-place operations, so that
            each output element depends only on the corresponding input element.
            The caller is responsible for ensuring the assumption holds;
            no validation is performed. Requires dy_shape == dx_shape.

    Returns:
        dict mapping input paths to gradient arrays.
    """
    jac = jacobian_fn(inputs=inputs, jac_inputs=vjp_inputs, jac_outputs=vjp_outputs)
    out = {}
    for dx in vjp_inputs:
        grad = None
        for dy in vjp_outputs:
            J = np.asarray(jac[dy][dx])  # shape: (*dy_shape, *dx_shape)
            v = np.asarray(cotangent_vector[dy])  # shape: (*dy_shape)
            if diagonal:
                diag = np.diag(J.reshape(v.size, v.size))
                term = diag.reshape(v.shape) * v
            else:
                term = np.tensordot(v, J, axes=v.ndim)
            grad = term if grad is None else grad + term
        out[dx] = grad
    return out


def jvp_from_jacobian(
    jacobian_fn: Callable,
    inputs: Any,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
    diagonal: bool = False,
) -> dict[str, Any]:
    """Compute JVP as J @ t using the full Jacobian.

    Args:
        jacobian_fn: The api_module.jacobian callable.
        inputs: Validated InputSchema instance.
        jvp_inputs: set[str] of input path strings.
        jvp_outputs: set[str] of output path strings.
        tangent_vector: dict mapping input paths to tangent arrays.
        diagonal: If True, assume each Jacobian block is diagonal (i.e. all
            off-diagonal entries are zero) and compute the product using an
            elementwise multiply against the diagonal instead of a full
            tensordot. This is the correct choice when the primal computation
            consists entirely of elementwise or in-place operations, so that
            each output element depends only on the corresponding input element.
            The caller is responsible for ensuring the assumption holds;
            no validation is performed. Requires dy_shape == dx_shape.

    Returns:
        dict mapping output paths to JVP result arrays.
    """
    jac = jacobian_fn(inputs=inputs, jac_inputs=jvp_inputs, jac_outputs=jvp_outputs)
    out = {}
    for dy in jvp_outputs:
        result = None
        for dx in jvp_inputs:
            J = np.asarray(jac[dy][dx])  # shape: (*dy_shape, *dx_shape)
            t = np.asarray(tangent_vector[dx])  # shape: (*dx_shape)
            if diagonal:
                diag = np.diag(J.reshape(t.size, t.size))
                term = diag.reshape(t.shape) * t
            else:
                term = np.tensordot(J, t, axes=t.ndim)
            result = term if result is None else result + term
        out[dy] = result
    return out


def jacobian_from_vjp(
    vjp_fn: Callable,
    apply_fn: Callable,
    inputs: Any,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> dict[str, dict[str, Any]]:
    """Compute the Jacobian by calling VJP with one-hot cotangent vectors.

    Requires M calls to VJP, where M is the total number of output elements.

    Args:
        vjp_fn: The api_module.vector_jacobian_product callable.
        apply_fn: The api_module.apply callable (used to determine output shapes).
        inputs: Validated InputSchema instance.
        jac_inputs: set[str] of input path strings.
        jac_outputs: set[str] of output path strings.

    Returns:
        dict[str, dict[str, np.ndarray]] with structure {output_path: {input_path: array}}
        where each array has shape ``(*output_shape, *input_shape)``.
    """
    raw_outputs = apply_fn(inputs=inputs)
    outputs_dict = (
        raw_outputs.model_dump() if hasattr(raw_outputs, "model_dump") else raw_outputs
    )

    jac = {}
    output_vals = {}
    for dy in jac_outputs:
        dy_val = np.asarray(get_at_path(outputs_dict, dy))
        output_vals[dy] = dy_val
        jac[dy] = {}
        for dx in jac_inputs:
            dx_val = np.asarray(get_at_path(inputs, dx))
            jac[dy][dx] = np.zeros((*dy_val.shape, *dx_val.shape), dtype=dy_val.dtype)

    for dy in jac_outputs:
        dy_val = output_vals[dy]
        dy_shape = dy_val.shape
        for nd_idx in np.ndindex(*dy_shape) if dy_shape else [()]:
            cotangent = {dy: np.zeros_like(dy_val)}
            if dy_shape:
                cotangent[dy][nd_idx] = 1.0
            else:
                cotangent[dy] = np.array(1.0, dtype=dy_val.dtype)
            grad = vjp_fn(
                inputs=inputs,
                vjp_inputs=jac_inputs,
                vjp_outputs={dy},
                cotangent_vector=cotangent,
            )
            for dx in jac_inputs:
                if dy_shape:
                    jac[dy][dx][nd_idx] = np.asarray(grad[dx])
                else:
                    jac[dy][dx] = np.asarray(grad[dx])
    return jac


def jacobian_from_jvp(
    jvp_fn: Callable,
    apply_fn: Callable,
    inputs: Any,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> dict[str, dict[str, Any]]:
    """Compute the Jacobian by calling JVP with one-hot tangent vectors.

    Requires N calls to JVP, where N is the total number of input elements.

    Args:
        jvp_fn: The api_module.jacobian_vector_product callable.
        apply_fn: The api_module.apply callable (used to determine output shapes).
        inputs: Validated InputSchema instance.
        jac_inputs: set[str] of input path strings.
        jac_outputs: set[str] of output path strings.

    Returns:
        dict[str, dict[str, np.ndarray]] with structure {output_path: {input_path: array}}
        where each array has shape ``(*output_shape, *input_shape)``.
    """
    raw_outputs = apply_fn(inputs=inputs)
    outputs_dict = (
        raw_outputs.model_dump() if hasattr(raw_outputs, "model_dump") else raw_outputs
    )

    jac = {}
    for dy in jac_outputs:
        dy_val = np.asarray(get_at_path(outputs_dict, dy))
        jac[dy] = {}
        for dx in jac_inputs:
            dx_val = np.asarray(get_at_path(inputs, dx))
            jac[dy][dx] = np.zeros((*dy_val.shape, *dx_val.shape), dtype=dy_val.dtype)

    for dx in jac_inputs:
        dx_val = np.asarray(get_at_path(inputs, dx))
        dx_shape = dx_val.shape
        for nd_idx in np.ndindex(*dx_shape) if dx_shape else [()]:
            tangent = {dx: np.zeros_like(dx_val)}
            if dx_shape:
                tangent[dx][nd_idx] = 1.0
            else:
                tangent[dx] = np.array(1.0, dtype=dx_val.dtype)
            result = jvp_fn(
                inputs=inputs,
                jvp_inputs={dx},
                jvp_outputs=jac_outputs,
                tangent_vector=tangent,
            )
            for dy in jac_outputs:
                dy_result = np.asarray(result[dy])
                if dx_shape:
                    jac[dy][dx][(..., *nd_idx)] = dy_result
                else:
                    jac[dy][dx] = dy_result
    return jac


# Flag is modified by runtime.cli based on arguments or during build time
SKIP_REQUIRED_FILE_CHECK = False


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

        def create_sequence(maybe_path: str | Sequence[Any]) -> LazySequence:
            """Expand a glob pattern into a LazySequence if needed."""
            validator = SchemaValidator(item_schema)

            if not isinstance(maybe_path, str) or not maybe_path.startswith("@"):
                items = maybe_path
                getter = validator.validate_python
                return LazySequence(items, getter)

            # We know that the path is a glob pattern, so we need to load items from files
            from .file_interactions import (
                expand_glob,
                read_from_path,
            )

            maybe_path = maybe_path[1:]
            items = expand_glob(maybe_path)

            def load_item(key: str) -> Any:
                buffer = read_from_path(key)
                obj = json.loads(buffer.decode("utf-8"))
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


def _resolve_input_path(path: Path) -> Path:
    from tesseract_core.runtime.config import get_config

    input_path = get_config().input_path
    tess_path = (input_path / path).resolve()
    if str(input_path) not in str(tess_path):
        raise ValueError(
            f"Invalid input file reference: {path}. "
            f"Expected path to be relative to {input_path}, but got {tess_path}. "
            "File references have to be relative to --input-path."
        )
    if not tess_path.exists():
        raise FileNotFoundError(f"Input path {tess_path} does not exist.")
    if not tess_path.is_file():
        raise ValueError(f"Input path {tess_path} is not a file.")
    return tess_path


def _strip_output_path(path: Path) -> Path:
    from tesseract_core.runtime.config import get_config

    output_path = get_config().output_path
    if path.is_relative_to(output_path):
        return path.relative_to(output_path)
    else:
        return path


InputFileReference = Annotated[Path, AfterValidator(_resolve_input_path)]
OutputFileReference = Annotated[Path, AfterValidator(_strip_output_path)]


def require_file(file_path: PathLike) -> Path:
    """Designate a file which is required to be present at runtime.

    Args:
        file_path: Path to required file. Must be relative to `input_path` assigned in `tesseract run`.
    """
    if SKIP_REQUIRED_FILE_CHECK:
        return Path(file_path)

    file_path = _resolve_input_path(Path(file_path))

    if not file_path.is_file():
        raise FileNotFoundError(f"Required file not found: {file_path}")

    return file_path


class TesseractReference:
    """Allows passing a reference to another Tesseract as input."""

    def __init__(self, tesseract: Any) -> None:
        self._tesseract = tesseract

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying Tesseract instance."""
        return getattr(self._tesseract, name)

    @classmethod
    def _get_tesseract_class(cls) -> type:
        """Lazy import of Tesseract class. Avoids hard dependency of Tesseract runtime on Tesseract SDK."""
        try:
            from tesseract_core import Tesseract

            return Tesseract
        except ImportError:
            raise ImportError(
                "Tesseract class not found. Ensure tesseract_core is installed and configured correctly."
            ) from ImportError

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for TesseractReference."""

        def validate_tesseract_reference(v: Any) -> "TesseractReference":
            if isinstance(v, cls):
                return v

            if not (isinstance(v, dict) and "type" in v and "ref" in v):
                raise ValueError(
                    f"Expected dict with 'type' and 'ref' keys, got {type(v)}"
                )

            tesseract_type = v["type"]
            ref = v["ref"]

            if tesseract_type not in ("api_path", "image", "url"):
                raise ValueError(
                    f"Invalid tesseract type '{tesseract_type}'. Expected 'api_path', 'image' or 'url'."
                )

            Tesseract = cls._get_tesseract_class()
            if tesseract_type == "api_path":
                tesseract = Tesseract.from_tesseract_api(ref)
            elif tesseract_type == "image":
                tesseract = Tesseract.from_image(ref)
                tesseract.serve()
            elif tesseract_type == "url":
                tesseract = Tesseract.from_url(ref)

            return cls(tesseract)

        return core_schema.no_info_plain_validator_function(
            validate_tesseract_reference
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Generate JSON schema for OpenAPI."""
        return {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["api_path", "image", "url"],
                    "description": "Type of tesseract reference",
                },
                "ref": {
                    "type": "string",
                    "description": "URL or file path to the tesseract",
                },
            },
            "required": ["type", "ref"],
            "additionalProperties": False,
        }


__all__ = [
    "InputFileReference",
    "LazySequence",
    "OutputFileReference",
    "PydanticLazySequenceAnnotation",
    "TesseractReference",
    "finite_difference_jacobian",
    "finite_difference_jvp",
    "finite_difference_vjp",
    "jacobian_from_jvp",
    "jacobian_from_vjp",
    "jvp_from_jacobian",
    "log_artifact",
    "log_metric",
    "log_parameter",
    "require_file",
    "vjp_from_jacobian",
]
