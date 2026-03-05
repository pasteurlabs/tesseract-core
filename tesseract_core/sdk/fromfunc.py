# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-generate tesseract_api.py from a type-hinted Python function."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

# --- Constants ---

_TYPING_NAMES = {
    "Any",
    "Optional",
    "Union",
    "Literal",
    "ClassVar",
    "List",
    "Dict",
    "Tuple",
    "Set",
    "FrozenSet",
    "Sequence",
    "Mapping",
    "Iterable",
    "Iterator",
    "Callable",
    "Type",
    "Deque",
    "DefaultDict",
    "OrderedDict",
    "Counter",
    "NamedTuple",
    "TypedDict",
}

_BUILTIN_TYPE_NAMES = {
    "int",
    "float",
    "str",
    "bool",
    "bytes",
    "list",
    "dict",
    "tuple",
    "set",
    "frozenset",
    "None",
    "complex",
    "object",
    "type",
    "bytearray",
    "memoryview",
    "range",
    "slice",
    "property",
}

_ARRAY_BARE_NAMES = {"NDArray", "Tensor"}


# --- Data classes ---


@dataclass
class ParameterInfo:
    """Information about a function parameter."""

    name: str
    annotation: str
    default: str | None = None
    description: str | None = None


@dataclass
class OutputFieldInfo:
    """Information about an output field."""

    name: str
    annotation: str
    description: str | None = None


@dataclass
class FunctionAnalysis:
    """Result of analyzing a Python function."""

    func_name: str
    parameters: list[ParameterInfo]
    output_fields: list[OutputFieldInfo]
    detected_framework: str | None = None  # "jax", "pytorch", "numpy", or None
    has_x64: bool = False
    source_file: Path = field(default_factory=lambda: Path("."))
    module_name: str = ""
    _ast_tree: ast.Module | None = field(default=None, repr=False)
    _func_node: ast.FunctionDef | None = field(default=None, repr=False)


@dataclass
class NameOrigin:
    """Tracks where a name was defined in the source file."""

    name: str
    origin: str  # "defined" or "imported"
    class_node: ast.ClassDef | None = None


@dataclass
class _FlattenInfo:
    """Info for flattening a structured type into schema fields."""

    class_name: str
    # (name, annotation, default_or_None, description_or_None)
    fields: list[tuple[str, str, str | None, str | None]]


# --- Docstring parsing ---


def _parse_docstring(
    func_node: ast.FunctionDef,
) -> tuple[dict[str, str], dict[str, str]]:
    """Parse a function's docstring to extract parameter and return descriptions.

    Supports Google, NumPy, and Sphinx/reST docstring formats.
    Returns (param_descriptions, return_descriptions), both {name: description} dicts.
    Silently returns ({}, {}) on failure or unrecognized format.
    """
    try:
        docstring = ast.get_docstring(func_node)
        if not docstring:
            return {}, {}

        # Try each format in order
        for parser in (_parse_google, _parse_numpy, _parse_sphinx):
            result = parser(docstring)
            if result is not None:
                return result

        return {}, {}
    except Exception:
        return {}, {}


def _parse_google(docstring: str) -> tuple[dict[str, str], dict[str, str]] | None:
    """Parse Google-style docstring. Returns None if not Google style."""
    lines = docstring.split("\n")

    # Detect Google style: look for "Args:" or "Arguments:" or "Returns:" sections
    has_args = any(line.strip() in ("Args:", "Arguments:") for line in lines)
    has_returns = any(line.strip() == "Returns:" for line in lines)

    if not has_args and not has_returns:
        return None

    param_descs = _parse_google_section(lines, ("Args:", "Arguments:"))
    return_descs = _parse_google_section(lines, ("Returns:",))

    return param_descs, return_descs


def _parse_google_section(
    lines: list[str], section_headers: tuple[str, ...]
) -> dict[str, str]:
    """Parse a Google-style section (Args or Returns) into {name: description}."""
    result: dict[str, str] = {}
    in_section = False
    section_indent: int | None = None
    current_name: str | None = None
    current_desc_parts: list[str] = []

    for line in lines:
        stripped = line.strip()

        if stripped in section_headers:
            in_section = True
            section_indent = None
            current_name = None
            current_desc_parts = []
            continue

        if not in_section:
            continue

        # Empty line within section is okay (continuation gap)
        if not stripped:
            if current_name:
                current_desc_parts.append("")
            continue

        # Determine indent level
        indent = len(line) - len(line.lstrip())

        # If we hit a line at same or lesser indent as section header,
        # or another section header, end this section
        if section_indent is not None and indent <= (section_indent - 4) and stripped:
            break
        # Check if this is a new section header (e.g., "Returns:", "Raises:", etc.)
        if stripped.endswith(":") and indent == 0:
            break
        if re.match(r"^[A-Z]\w*:$", stripped) and indent <= 4:
            break

        # First content line sets the base indent
        if section_indent is None:
            section_indent = indent

        # Line at base indent with "name:" pattern -> new field
        if indent == section_indent:
            match = re.match(r"^(\w+)\s*(?:\(.*?\))?\s*:\s*(.*)", stripped)
            if match:
                # Save previous
                if current_name:
                    result[current_name] = _join_desc(current_desc_parts)
                current_name = match.group(1)
                current_desc_parts = [match.group(2)] if match.group(2) else []
                continue

        # Continuation line (deeper indent or same indent without "name:")
        if current_name:
            current_desc_parts.append(stripped)

    # Save last entry
    if current_name:
        result[current_name] = _join_desc(current_desc_parts)

    return result


def _parse_numpy(docstring: str) -> tuple[dict[str, str], dict[str, str]] | None:
    """Parse NumPy-style docstring. Returns None if not NumPy style."""
    lines = docstring.split("\n")

    # Detect NumPy style: look for section underlines (---+)
    has_underline = any(re.match(r"^\s*-{3,}\s*$", line) for line in lines)
    if not has_underline:
        return None

    param_descs = _parse_numpy_section(lines, ("Parameters", "Params"))
    return_descs = _parse_numpy_section(lines, ("Returns", "Return"))

    return param_descs, return_descs


def _parse_numpy_section(
    lines: list[str], section_headers: tuple[str, ...]
) -> dict[str, str]:
    """Parse a NumPy-style section into {name: description}."""
    result: dict[str, str] = {}
    in_section = False
    past_underline = False
    section_indent: int | None = None
    current_name: str | None = None
    current_desc_parts: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Look for section header
        if stripped in section_headers:
            # Check next line is underline
            if i + 1 < len(lines) and re.match(r"^\s*-{3,}\s*$", lines[i + 1]):
                in_section = True
                past_underline = False
                section_indent = None
                current_name = None
                current_desc_parts = []
                continue

        if not in_section:
            continue

        # Skip underline
        if re.match(r"^\s*-{3,}\s*$", line):
            past_underline = True
            continue

        if not past_underline:
            continue

        # Empty line
        if not stripped:
            if current_name:
                current_desc_parts.append("")
            continue

        indent = len(line) - len(line.lstrip())

        # Check for new section header (word followed by underline)
        if i + 1 < len(lines) and re.match(r"^\s*-{3,}\s*$", lines[i + 1]):
            break

        # First content line sets base indent
        if section_indent is None:
            section_indent = indent

        # Line at base indent with "name : type" or "name" pattern -> new field
        if indent == section_indent:
            # NumPy style: "name : type" or just "name"
            match = re.match(r"^(\w+)\s*(?::.*)?$", stripped)
            if match:
                if current_name:
                    result[current_name] = _join_desc(current_desc_parts)
                current_name = match.group(1)
                current_desc_parts = []
                continue

        # Description lines (deeper indent)
        if current_name and indent > section_indent:
            current_desc_parts.append(stripped)

    if current_name:
        result[current_name] = _join_desc(current_desc_parts)

    return result


def _parse_sphinx(docstring: str) -> tuple[dict[str, str], dict[str, str]] | None:
    """Parse Sphinx/reST-style docstring. Returns None if not Sphinx style."""
    lines = docstring.split("\n")

    has_param = any(re.match(r"^\s*:param\s+", line) for line in lines)
    has_return = any(re.match(r"^\s*:returns?:", line) for line in lines)

    if not has_param and not has_return:
        return None

    param_descs: dict[str, str] = {}
    return_descs: dict[str, str] = {}

    for line in lines:
        # :param name: description
        match = re.match(r"^\s*:param\s+(\w+)\s*:\s*(.*)", line)
        if match:
            param_descs[match.group(1)] = match.group(2).strip()
            continue

        # :returns: description (single return value)
        match = re.match(r"^\s*:returns?\s*:\s*(.*)", line)
        if match:
            desc = match.group(1).strip()
            if desc:
                return_descs["result"] = desc

    return param_descs, return_descs


def _join_desc(parts: list[str]) -> str:
    """Join description parts, stripping trailing empty lines."""
    # Remove trailing empty strings
    while parts and not parts[-1]:
        parts.pop()
    return " ".join(p for p in parts if p)


# --- Parsing ---


def parse_fromfunc_arg(arg: str) -> tuple[Path, str]:
    """Parse a --fromfunc argument of the form 'path.py::func_name'.

    Args:
        arg: String in format "path/to/file.py::function_name"

    Returns:
        Tuple of (file_path, function_name)

    Raises:
        ValueError: If format is invalid or file doesn't exist.
    """
    if "::" not in arg:
        raise ValueError(
            f"Invalid --fromfunc format: '{arg}'. "
            "Expected format: 'path/to/file.py::function_name'"
        )

    parts = arg.split("::", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid --fromfunc format: '{arg}'. "
            "Expected format: 'path/to/file.py::function_name'"
        )

    file_path = Path(parts[0])
    func_name = parts[1]

    if not file_path.exists():
        raise ValueError(f"Source file not found: '{file_path}'")

    if not file_path.suffix == ".py":
        raise ValueError(f"Source file must be a Python file (.py): '{file_path}'")

    return file_path, func_name


def analyze_function(file_path: Path, func_name: str) -> FunctionAnalysis:
    """Analyze a Python function using AST (no importing).

    Args:
        file_path: Path to the Python source file.
        func_name: Name of the function to analyze.

    Returns:
        FunctionAnalysis with extracted information.

    Raises:
        ValueError: If function not found, has missing type hints, or uses unsupported features.
    """
    source = file_path.read_text()
    tree = ast.parse(source, filename=str(file_path))

    # Find the function
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            func_node = node
            break

    if func_node is None:
        raise ValueError(f"Function '{func_name}' not found in '{file_path}'")

    # Extract parameters
    parameters = _extract_parameters(func_node, file_path)

    # Extract output fields from return type annotation + return statements
    output_fields = _extract_output_fields(func_node, file_path)

    # Parse docstring for descriptions
    param_descs, return_descs = _parse_docstring(func_node)
    for param in parameters:
        if param.name in param_descs:
            param.description = param_descs[param.name]
    for out_field in output_fields:
        if out_field.name in return_descs:
            out_field.description = return_descs[out_field.name]
        elif len(output_fields) == 1 and "result" in return_descs:
            # Single output: apply unnamed return description
            out_field.description = return_descs["result"]

    # Detect framework from imports
    detected_framework = _detect_framework(tree)

    # Detect x64 config
    has_x64 = _detect_x64(tree)

    # Compute module name from file path
    module_name = file_path.stem

    return FunctionAnalysis(
        func_name=func_name,
        parameters=parameters,
        output_fields=output_fields,
        detected_framework=detected_framework,
        has_x64=has_x64,
        source_file=file_path,
        module_name=module_name,
        _ast_tree=tree,
        _func_node=func_node,
    )


def _extract_parameters(
    func_node: ast.FunctionDef, file_path: Path
) -> list[ParameterInfo]:
    """Extract parameter information from a function definition."""
    params = []
    args = func_node.args

    # Check for *args and **kwargs
    if args.vararg:
        raise ValueError(
            f"Function '{func_node.name}' in '{file_path}' uses *args, "
            "which is not supported by --fromfunc."
        )
    if args.kwarg:
        raise ValueError(
            f"Function '{func_node.name}' in '{file_path}' uses **kwargs, "
            "which is not supported by --fromfunc."
        )

    # Get all regular arguments (positional + keyword)
    all_args = args.args

    # Calculate defaults offset: defaults are right-aligned to args
    num_defaults = len(args.defaults)
    num_args = len(all_args)
    defaults_offset = num_args - num_defaults

    for i, arg in enumerate(all_args):
        # Skip 'self' parameter
        if arg.arg == "self":
            continue

        # Check for type annotation
        if arg.annotation is None:
            raise ValueError(
                f"Parameter '{arg.arg}' in function '{func_node.name}' "
                f"('{file_path}') is missing a type annotation. "
                "All parameters must have type hints when using --fromfunc."
            )

        annotation_str = ast.unparse(arg.annotation)

        # Get default value if present
        default = None
        default_idx = i - defaults_offset
        if default_idx >= 0:
            default = ast.unparse(args.defaults[default_idx])

        params.append(
            ParameterInfo(
                name=arg.arg,
                annotation=annotation_str,
                default=default,
            )
        )

    # Also handle keyword-only arguments
    for i, arg in enumerate(args.kwonlyargs):
        if arg.annotation is None:
            raise ValueError(
                f"Parameter '{arg.arg}' in function '{func_node.name}' "
                f"('{file_path}') is missing a type annotation. "
                "All parameters must have type hints when using --fromfunc."
            )

        annotation_str = ast.unparse(arg.annotation)

        default = None
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            default = ast.unparse(args.kw_defaults[i])

        params.append(
            ParameterInfo(
                name=arg.arg,
                annotation=annotation_str,
                default=default,
            )
        )

    return params


def _extract_output_fields(
    func_node: ast.FunctionDef, file_path: Path
) -> list[OutputFieldInfo]:
    """Extract output field information from return type annotation + return statements.

    The return type annotation is REQUIRED and determines the field types.
    Return statements are analyzed for field names (dict literal keys, variable names).
    """
    # Require return type annotation
    if func_node.returns is None:
        raise ValueError(
            f"Function '{func_node.name}' in '{file_path}' is missing a return "
            "type annotation. A return type hint is required when using --fromfunc."
        )

    return_annotation = func_node.returns
    annotation_str = ast.unparse(return_annotation)

    # Determine field names from return statements
    field_names = _extract_return_field_names(func_node)

    # Parse the return type to determine output field types
    return _resolve_output_types(return_annotation, annotation_str, field_names)


def _extract_return_field_names(func_node: ast.FunctionDef) -> list[str]:
    """Extract field names from return statements (AST analysis)."""
    return_nodes = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and node.value is not None:
            return_nodes.append(node)

    if not return_nodes:
        return ["result"]

    # Use the first return statement for analysis
    return_value = return_nodes[0].value

    # Dict literal -> keys become field names
    if isinstance(return_value, ast.Dict):
        names = []
        for key in return_value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                names.append(key.value)
            else:
                return ["result"]
        return names if names else ["result"]

    # Named variable -> variable name
    if isinstance(return_value, ast.Name):
        return [return_value.id]

    # Tuple -> element names or result1, result2, ...
    if isinstance(return_value, ast.Tuple):
        names = []
        for i, elt in enumerate(return_value.elts):
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            else:
                names.append(f"result{i + 1}")
        return names

    # Bare expression
    return ["result"]


def _get_annotation_base_name(node: ast.expr) -> str | None:
    """Get the base name of an annotation node (e.g., 'dict', 'Dict', 'tuple')."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _resolve_output_types(
    return_annotation: ast.expr,
    annotation_str: str,
    field_names: list[str],
) -> list[OutputFieldInfo]:
    """Resolve output field types from the return type annotation.

    Handles:
    - Simple types: float, np.ndarray, etc. -> single field with that type
    - dict[str, ValueType]: each field gets ValueType
    - tuple[T1, T2, ...]: paired with field names
    - Bare dict/tuple without params: error
    """
    # Check for parameterized generic types (dict[str, X], tuple[X, Y], etc.)
    if isinstance(return_annotation, ast.Subscript):
        base_name = _get_annotation_base_name(return_annotation.value)

        if base_name in ("dict", "Dict"):
            return _resolve_dict_return(return_annotation, field_names)

        if base_name in ("tuple", "Tuple"):
            return _resolve_tuple_return(return_annotation, field_names)

    # Check for bare dict/tuple without type parameters
    if isinstance(return_annotation, ast.Name):
        if return_annotation.id in ("dict", "Dict"):
            raise ValueError(
                "Return type 'dict' is too vague. Please use "
                "'dict[str, <value_type>]' to specify the value type, "
                "e.g., 'dict[str, float]'. If the values have mixed types, "
                "use 'dict[str, Any]' and edit the generated OutputSchema fields."
            )
        if return_annotation.id in ("tuple", "Tuple"):
            raise ValueError(
                "Return type 'tuple' is too vague. Please use "
                "'tuple[Type1, Type2, ...]' to specify element types."
            )

    # Simple type -> single output field
    return [OutputFieldInfo(name=field_names[0], annotation=annotation_str)]


def _resolve_dict_return(
    annotation: ast.Subscript,
    field_names: list[str],
) -> list[OutputFieldInfo]:
    """Resolve output types for dict[str, ValueType] return."""
    slice_node = annotation.slice

    # dict[str, ValueType]
    if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
        value_type_str = ast.unparse(slice_node.elts[1])
        return [
            OutputFieldInfo(name=name, annotation=value_type_str)
            for name in field_names
        ]

    raise ValueError(
        f"Return type '{ast.unparse(annotation)}' is not supported. "
        "For dict returns, use 'dict[str, <value_type>]', "
        "e.g., 'dict[str, float]'."
    )


def _resolve_tuple_return(
    annotation: ast.Subscript,
    field_names: list[str],
) -> list[OutputFieldInfo]:
    """Resolve output types for tuple[T1, T2, ...] return."""
    slice_node = annotation.slice

    if isinstance(slice_node, ast.Tuple):
        type_strs = [ast.unparse(elt) for elt in slice_node.elts]

        if len(type_strs) != len(field_names):
            # Mismatch: generate result1, result2, ... names
            field_names = [f"result{i + 1}" for i in range(len(type_strs))]

        return [
            OutputFieldInfo(name=name, annotation=type_str)
            for name, type_str in zip(field_names, type_strs, strict=True)
        ]

    # Single-element tuple: tuple[X]
    type_str = ast.unparse(slice_node)
    return [OutputFieldInfo(name=field_names[0], annotation=type_str)]


def _detect_framework(tree: ast.Module) -> str | None:
    """Detect computational framework from imports in the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "jax" or alias.name.startswith("jax."):
                    return "jax"
                if alias.name == "torch" or alias.name.startswith("torch."):
                    return "pytorch"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                if node.module == "jax" or node.module.startswith("jax."):
                    return "jax"
                if node.module == "torch" or node.module.startswith("torch."):
                    return "pytorch"

    # Second pass for numpy (lower priority)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy" or alias.name.startswith("numpy."):
                    return "numpy"
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "numpy" or node.module.startswith("numpy.")
            ):
                return "numpy"

    return None


def _detect_x64(tree: ast.Module) -> bool:
    """Detect jax.config.update('jax_enable_x64', True) in AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "update":
                if (
                    isinstance(func.value, ast.Attribute)
                    and func.value.attr == "config"
                ):
                    if (
                        isinstance(func.value.value, ast.Name)
                        and func.value.value.id == "jax"
                    ):
                        if (
                            len(node.args) >= 2
                            and isinstance(node.args[0], ast.Constant)
                            and node.args[0].value == "jax_enable_x64"
                            and isinstance(node.args[1], ast.Constant)
                            and node.args[1].value is True
                        ):
                            return True
    return False


# --- Type mapping ---
#
# Array types are mapped explicitly to Tesseract Array[..., DType].
# Everything else is passed through as-is to the Pydantic schema,
# which validates at class-definition time. This means all Python
# primitives, built-in generic types (list[float], dict[str, int], etc.),
# typing constructs (Optional, Union, Any, Literal, ...) are supported
# as long as Pydantic supports them.

_NUMPY_ARRAY_TYPES = {
    "np.ndarray",
    "numpy.ndarray",
    "NDArray",
    "npt.NDArray",
    "np.typing.NDArray",
}
_JAX_ARRAY_TYPES = {"jax.Array", "jnp.ndarray", "jax.numpy.ndarray"}
_TORCH_ARRAY_TYPES = {"torch.Tensor", "Tensor"}


def map_type_annotation(
    annotation_str: str,
    framework: str | None = None,
    has_x64: bool = False,
) -> str:
    """Map a Python type annotation string to a Tesseract schema type.

    Array types (numpy, jax, torch) are mapped to Tesseract Array[..., DType].
    All other types are passed through as-is for Pydantic to validate.

    Args:
        annotation_str: The type annotation as a string.
        framework: Detected framework ("jax", "pytorch", "numpy", or None).
        has_x64: Whether jax_enable_x64 is set to True.

    Returns:
        The Tesseract type string for use in schema definitions.
    """
    # Numpy array types
    if annotation_str in _NUMPY_ARRAY_TYPES:
        return "Array[..., Float64]"

    # JAX array types
    if annotation_str in _JAX_ARRAY_TYPES:
        dtype = "Float64" if has_x64 else "Float32"
        return f"Array[..., {dtype}]"

    # PyTorch array types
    if annotation_str in _TORCH_ARRAY_TYPES:
        return "Array[..., Float32]"

    # Everything else: pass through as-is (Pydantic validates at runtime)
    return annotation_str


def _is_array_type(tesseract_type: str) -> bool:
    """Check if a Tesseract type is an array type (should be wrapped with Differentiable)."""
    return tesseract_type.startswith("Array[")


def _should_use_differentiable(framework: str | None) -> bool:
    """Check if the recipe uses Differentiable wrapper for array fields."""
    return framework in ("jax", "pytorch", "numpy")


def _collect_runtime_imports(all_field_strs: str, recipe: str) -> set[str]:
    """Determine which tesseract_core.runtime imports are needed.

    For jax/pytorch recipes, Differentiable and Float32 are already imported
    by the template itself, so we only add what's extra.
    """
    already_imported = set()
    if recipe in ("jax", "pytorch"):
        already_imported = {"Differentiable", "Float32"}

    needed = set()
    if "Array[" in all_field_strs:
        needed.add("Array")
    if "Differentiable[" in all_field_strs:
        needed.add("Differentiable")
    for dtype in ("Float16", "Float32", "Float64", "Int8", "Int16", "Int32", "Int64"):
        if dtype in all_field_strs:
            needed.add(dtype)

    return needed - already_imported


def _collect_typing_imports(all_field_strs: str) -> set[str]:
    """Detect typing module names used in field strings that need importing.

    Scans for capitalized typing constructs (Optional, Union, Any, Literal, etc.)
    that would need 'from typing import ...' in the generated file.
    """
    needed = set()
    for name in _TYPING_NAMES:
        if name in all_field_strs:
            needed.add(name)
    return needed


# --- Structured type support ---


def _extract_referenced_names(annotation_node: ast.expr) -> set[str]:
    """Extract class-type names referenced in an annotation AST node.

    Walks the annotation tree, collects ast.Name nodes, and excludes
    builtins, typing names, bare array type names, and names that are
    the prefix of a dotted attribute access (e.g., 'jax' in 'jax.Array').
    """
    excluded = _BUILTIN_TYPE_NAMES | _TYPING_NAMES | _ARRAY_BARE_NAMES

    # Collect Name nodes that are children of Attribute nodes (dotted prefixes)
    attr_value_ids: set[int] = set()
    for node in ast.walk(annotation_node):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            attr_value_ids.add(id(node.value))

    names = set()
    for node in ast.walk(annotation_node):
        if (
            isinstance(node, ast.Name)
            and node.id not in excluded
            and id(node) not in attr_value_ids
        ):
            names.add(node.id)
    return names


def _build_source_name_map(tree: ast.Module) -> dict[str, NameOrigin]:
    """Build a map of names defined or imported at the top level of a module."""
    name_map: dict[str, NameOrigin] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            name_map[node.name] = NameOrigin(
                name=node.name, origin="defined", class_node=node
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_name = alias.asname if alias.asname else alias.name
                name_map[imported_name] = NameOrigin(
                    name=imported_name, origin="imported"
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_name = alias.asname if alias.asname else alias.name
                name_map[imported_name] = NameOrigin(
                    name=imported_name, origin="imported"
                )

    return name_map


def _get_decorator_name(node: ast.expr) -> str | None:
    """Extract the name from a decorator node.

    Handles: @dataclass, @dataclasses.dataclass, @dataclass(frozen=True)
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _get_decorator_name(node.func)
    return None


def _get_base_class_name(node: ast.expr) -> str | None:
    """Extract the name from a base class node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _classify_class(
    class_node: ast.ClassDef,
    source_name_map: dict[str, NameOrigin],
) -> str:
    """Classify a class definition as namedtuple/dataclass/typeddict/basemodel/plain.

    Priority: decorators → base classes → same-file inheritance → plain.
    """
    # Check decorators
    for decorator in class_node.decorator_list:
        dec_name = _get_decorator_name(decorator)
        if dec_name == "dataclass":
            return "dataclass"

    # Check base classes
    for base in class_node.bases:
        base_name = _get_base_class_name(base)
        if base_name == "NamedTuple":
            return "namedtuple"
        if base_name == "TypedDict":
            return "typeddict"
        if base_name == "BaseModel":
            return "basemodel"

        # Check if base class is defined in the same file and trace its classification
        if base_name and base_name in source_name_map:
            origin = source_name_map[base_name]
            if origin.origin == "defined" and origin.class_node is not None:
                parent_kind = _classify_class(origin.class_node, source_name_map)
                if parent_kind != "plain":
                    return parent_kind

    return "plain"


def _extract_class_fields(
    class_node: ast.ClassDef, kind: str
) -> list[tuple[str, str, str | None, str | None]]:
    """Extract fields from a structured class definition.

    Returns list of (field_name, annotation_str, default_str_or_None, description_or_None).
    """
    fields = []
    for stmt in class_node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            annotation = ast.unparse(stmt.annotation)
            default = None
            if stmt.value is not None:
                default = ast.unparse(stmt.value)
            fields.append((name, annotation, default, None))
    return fields


def _collect_type_imports(
    analysis: FunctionAnalysis,
) -> tuple[list[str], dict[str, _FlattenInfo | None]]:
    """Determine which class names to import and which to flatten.

    Returns:
        (names_to_import, flatten_info) where:
        - names_to_import: class names to add to the source module import line
        - flatten_info: dict with keys "flatten_input" and "flatten_output",
          each either None or a _FlattenInfo
    """
    tree = analysis._ast_tree
    func_node = analysis._func_node
    if tree is None or func_node is None:
        return [], {"flatten_input": None, "flatten_output": None}

    # Gather all annotation nodes from params + return
    annotation_nodes: list[ast.expr] = []
    for arg in func_node.args.args:
        if arg.annotation is not None and arg.arg != "self":
            annotation_nodes.append(arg.annotation)
    for arg in func_node.args.kwonlyargs:
        if arg.annotation is not None:
            annotation_nodes.append(arg.annotation)
    if func_node.returns is not None:
        annotation_nodes.append(func_node.returns)

    # Extract all referenced names
    all_referenced: set[str] = set()
    for ann_node in annotation_nodes:
        all_referenced |= _extract_referenced_names(ann_node)

    if not all_referenced:
        return [], {"flatten_input": None, "flatten_output": None}

    # Build source name map
    source_name_map = _build_source_name_map(tree)

    # Validate and classify each referenced name
    names_to_import: list[str] = []
    flatten_input: _FlattenInfo | None = None
    flatten_output: _FlattenInfo | None = None

    # Determine if input should be flattened:
    # exactly 1 parameter AND that parameter's type is a bare structured class name
    non_self_args = [a for a in func_node.args.args if a.arg != "self"] + list(
        func_node.args.kwonlyargs
    )

    can_flatten_input = (
        len(non_self_args) == 1
        and non_self_args[0].annotation is not None
        and isinstance(non_self_args[0].annotation, ast.Name)
    )

    input_flatten_name: str | None = None
    if can_flatten_input:
        input_flatten_name = non_self_args[0].annotation.id  # type: ignore[union-attr]

    # Determine if output should be flattened:
    # return type is a bare Name (not subscript like list[Point])
    can_flatten_output = func_node.returns is not None and isinstance(
        func_node.returns, ast.Name
    )
    output_flatten_name: str | None = None
    if can_flatten_output:
        output_flatten_name = func_node.returns.id  # type: ignore[union-attr]

    # Determine which names appear in input annotations vs return annotation
    input_referenced: set[str] = set()
    for arg in non_self_args:
        if arg.annotation is not None:
            input_referenced |= _extract_referenced_names(arg.annotation)
    output_referenced: set[str] = set()
    if func_node.returns is not None:
        output_referenced = _extract_referenced_names(func_node.returns)

    for name in sorted(all_referenced):
        if name not in source_name_map:
            # Name not found in source - it might be from a third-party lib.
            # Add to imports so the generated file can reference it.
            names_to_import.append(name)
            continue

        origin = source_name_map[name]

        if origin.origin == "imported":
            # Already imported in source file - add to our import line
            names_to_import.append(name)
            continue

        # It's defined in the source file
        assert origin.class_node is not None
        kind = _classify_class(origin.class_node, source_name_map)

        if kind == "plain":
            raise ValueError(
                f"Class '{name}' is a plain class without a recognized structure "
                f"(NamedTuple, dataclass, TypedDict, or BaseModel). "
                f"Plain classes are not supported by --fromfunc. "
                f"Please convert '{name}' to a NamedTuple, dataclass, TypedDict, "
                f"or Pydantic BaseModel."
            )

        class_fields = _extract_class_fields(origin.class_node, kind)
        used_in_input = name in input_referenced
        used_in_output = name in output_referenced

        # Determine flattening for this name
        should_flatten_as_input = name == input_flatten_name and used_in_input
        should_flatten_as_output = name == output_flatten_name and used_in_output

        if should_flatten_as_input:
            flatten_input = _FlattenInfo(class_name=name, fields=class_fields)
        if should_flatten_as_output:
            flatten_output = _FlattenInfo(class_name=name, fields=class_fields)

        # If this name is used in input but NOT flattened, it needs to be imported
        needs_import = False
        if used_in_input and not should_flatten_as_input:
            needs_import = True
        # If used in output but NOT flattened, it needs to be imported
        if used_in_output and not should_flatten_as_output:
            needs_import = True
        if needs_import:
            names_to_import.append(name)

    flatten_info = {
        "flatten_input": flatten_input,
        "flatten_output": flatten_output,
    }

    return names_to_import, flatten_info


# --- Recipe detection ---


def detect_recipe_from_framework(framework: str | None) -> str:
    """Map a detected framework to a recipe name."""
    if framework == "jax":
        return "jax"
    if framework == "pytorch":
        return "pytorch"
    return "base"


# --- Template generation ---


def generate_template_vars(analysis: FunctionAnalysis, recipe: str) -> dict:
    """Generate Jinja2 template variables from a FunctionAnalysis.

    Args:
        analysis: The result of analyzing the source function.
        recipe: The recipe to generate for ("base", "jax", "pytorch").

    Returns:
        Dictionary of template variables to merge into the template context.
    """
    framework = analysis.detected_framework
    use_differentiable = _should_use_differentiable(framework)

    # Collect structured type info
    names_to_import, flatten_info = _collect_type_imports(analysis)
    flatten_input: _FlattenInfo | None = flatten_info["flatten_input"]
    flatten_output: _FlattenInfo | None = flatten_info["flatten_output"]

    # Error if structured types used with jax/pytorch recipes
    has_structured = bool(names_to_import) or flatten_input or flatten_output
    if has_structured and recipe in ("jax", "pytorch"):
        raise ValueError(
            f"Structured types (NamedTuple, dataclass, etc.) are not supported "
            f"with the '{recipe}' recipe. The '{recipe}' recipe uses "
            f"model_dump() which flattens structured types. "
            f"Please use the 'base' recipe instead."
        )

    # Generate input schema fields
    if flatten_input is not None:
        input_fields = _build_field_strings(
            flatten_input.fields, framework, analysis.has_x64, use_differentiable
        )
    else:
        input_fields = []
        for param in analysis.parameters:
            tesseract_type = map_type_annotation(
                param.annotation, framework, analysis.has_x64
            )
            if use_differentiable and _is_array_type(tesseract_type):
                field_type = f"Differentiable[{tesseract_type}]"
            else:
                field_type = tesseract_type

            input_fields.append(
                _build_field_str(
                    param.name, field_type, param.default, param.description
                )
            )

    # Generate output schema fields
    if flatten_output is not None:
        output_fields = _build_field_strings(
            flatten_output.fields, framework, analysis.has_x64, use_differentiable
        )
    else:
        output_fields = []
        for out_field in analysis.output_fields:
            tesseract_type = map_type_annotation(
                out_field.annotation, framework, analysis.has_x64
            )
            if use_differentiable and _is_array_type(tesseract_type):
                field_type = f"Differentiable[{tesseract_type}]"
            else:
                field_type = tesseract_type

            output_fields.append(
                _build_field_str(
                    out_field.name, field_type, None, out_field.description
                )
            )

    input_schema_fields = "\n".join(input_fields) if input_fields else "    pass"
    output_schema_fields = "\n".join(output_fields) if output_fields else "    pass"

    # Collect all field type strings to determine needed imports
    all_field_strs = input_schema_fields + "\n" + output_schema_fields
    runtime_imports = _collect_runtime_imports(all_field_strs, recipe)
    typing_imports = _collect_typing_imports(all_field_strs)
    needs_field_import = "Field(" in all_field_strs

    # Generate extra imports
    import_lines = []
    if needs_field_import:
        import_lines.append("from pydantic import Field")
    if typing_imports:
        import_lines.append(f"from typing import {', '.join(sorted(typing_imports))}")
    if runtime_imports:
        import_lines.append(
            f"from tesseract_core.runtime import {', '.join(sorted(runtime_imports))}"
        )

    # Build the source module import line
    source_imports = [analysis.func_name, *sorted(names_to_import)]
    import_lines.append(
        f"from {analysis.module_name} import {', '.join(source_imports)}"
    )
    extra_imports = "\n".join(import_lines)

    # Generate param unpacking
    param_names = [p.name for p in analysis.parameters]
    output_field_names = [f.name for f in analysis.output_fields]

    # Generate wrapper bodies based on recipe
    template_vars = {
        "extra_imports": extra_imports,
        "input_schema_fields": input_schema_fields,
        "output_schema_fields": output_schema_fields,
    }

    if recipe == "base":
        template_vars["apply_body"] = _generate_base_apply_body(
            analysis.func_name,
            param_names,
            output_field_names,
            flatten_input=flatten_input,
            flatten_output=flatten_output,
        )
    elif recipe == "jax":
        template_vars["apply_jit_body"] = _generate_jax_apply_jit_body(
            analysis.func_name, param_names, output_field_names
        )
    elif recipe == "pytorch":
        template_vars["evaluate_body"] = _generate_pytorch_evaluate_body(
            analysis.func_name, param_names, output_field_names
        )

    return template_vars


def _escape_description(desc: str) -> str:
    """Escape a description string for use in a Python string literal."""
    return desc.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _build_field_str(
    name: str,
    field_type: str,
    default: str | None,
    description: str | None,
) -> str:
    """Build a single schema field string, using Field() when needed."""
    if description:
        escaped = _escape_description(description)
        if default is not None:
            return f'    {name}: {field_type} = Field(description="{escaped}", default={default})'
        return f'    {name}: {field_type} = Field(description="{escaped}")'
    if default is not None:
        return f"    {name}: {field_type} = {default}"
    return f"    {name}: {field_type}"


def _build_field_strings(
    fields: list[tuple[str, str, str | None, str | None]],
    framework: str | None,
    has_x64: bool,
    use_differentiable: bool,
) -> list[str]:
    """Build schema field strings from (name, annotation, default, description) tuples."""
    result = []
    for name, annotation, default, description in fields:
        tesseract_type = map_type_annotation(annotation, framework, has_x64)
        if use_differentiable and _is_array_type(tesseract_type):
            field_type = f"Differentiable[{tesseract_type}]"
        else:
            field_type = tesseract_type

        result.append(_build_field_str(name, field_type, default, description))
    return result


def _generate_base_apply_body(
    func_name: str,
    param_names: list[str],
    output_field_names: list[str],
    flatten_input: _FlattenInfo | None = None,
    flatten_output: _FlattenInfo | None = None,
) -> str:
    """Generate apply() body for the base recipe."""
    lines = []

    if flatten_input is not None:
        # Reconstruct the structured type from flattened fields
        field_names = [f[0] for f in flatten_input.fields]
        kwargs = ", ".join(f"{f}=inputs.{f}" for f in field_names)
        # The original param name is param_names[0] (single param)
        original_param = param_names[0]
        lines.append(f"    {original_param} = {flatten_input.class_name}({kwargs})")
        lines.append("")
        lines.append(f"    result = {func_name}({original_param})")
    else:
        for name in param_names:
            lines.append(f"    {name} = inputs.{name}")
        lines.append("")
        lines.append(f"    result = {func_name}({', '.join(param_names)})")

    lines.append("")

    if flatten_output is not None:
        # Unpack fields from result
        field_names = [f[0] for f in flatten_output.fields]
        output_kwargs = ", ".join(f"{f}=result.{f}" for f in field_names)
        lines.append(f"    return OutputSchema({output_kwargs})")
    elif len(output_field_names) == 1:
        lines.append(f"    return OutputSchema({output_field_names[0]}=result)")
    else:
        output_kwargs = ", ".join(
            f'{name}=result["{name}"]' for name in output_field_names
        )
        lines.append(f"    return OutputSchema({output_kwargs})")

    return "\n".join(lines)


def _generate_jax_apply_jit_body(
    func_name: str,
    param_names: list[str],
    output_field_names: list[str],
) -> str:
    """Generate apply_jit() body for the jax recipe."""
    lines = []
    for name in param_names:
        lines.append(f'    {name} = inputs["{name}"]')
    lines.append("")
    if len(output_field_names) == 1:
        lines.append(f"    result = {func_name}({', '.join(param_names)})")
        lines.append("")
        lines.append(f'    return {{"{output_field_names[0]}": result}}')
    else:
        lines.append(f"    result = {func_name}({', '.join(param_names)})")
        lines.append("")
        output_items = ", ".join(
            f'"{name}": result["{name}"]' for name in output_field_names
        )
        lines.append(f"    return {{{output_items}}}")

    return "\n".join(lines)


def _generate_pytorch_evaluate_body(
    func_name: str,
    param_names: list[str],
    output_field_names: list[str],
) -> str:
    """Generate evaluate() body for the pytorch recipe."""
    lines = []
    for name in param_names:
        lines.append(f'    {name} = inputs["{name}"]')
    lines.append("")
    if len(output_field_names) == 1:
        lines.append(f"    result = {func_name}({', '.join(param_names)})")
        lines.append("")
        lines.append(f'    return {{"{output_field_names[0]}": result}}')
    else:
        lines.append(f"    result = {func_name}({', '.join(param_names)})")
        lines.append("")
        output_items = ", ".join(
            f'"{name}": result["{name}"]' for name in output_field_names
        )
        lines.append(f"    return {{{output_items}}}")

    return "\n".join(lines)
