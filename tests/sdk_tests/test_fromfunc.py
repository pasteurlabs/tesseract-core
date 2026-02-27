# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import textwrap

import pytest

from tesseract_core.sdk import engine, fromfunc

# --- Fixtures ---


@pytest.fixture
def simple_func_file(tmp_path):
    """Create a simple Python file with a typed function."""
    code = textwrap.dedent("""\
        def add(x: float, y: float) -> float:
            return x + y
    """)
    p = tmp_path / "simple.py"
    p.write_text(code)
    return p


@pytest.fixture
def numpy_func_file(tmp_path):
    """Create a Python file with numpy types."""
    code = textwrap.dedent("""\
        import numpy as np

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return a @ b
    """)
    p = tmp_path / "numpy_func.py"
    p.write_text(code)
    return p


@pytest.fixture
def jax_func_file(tmp_path):
    """Create a Python file with JAX types."""
    code = textwrap.dedent("""\
        import jax
        import jax.numpy as jnp

        def vector_add(a: jax.Array, b: jax.Array) -> jax.Array:
            return a + b
    """)
    p = tmp_path / "jax_func.py"
    p.write_text(code)
    return p


@pytest.fixture
def jax_x64_func_file(tmp_path):
    """Create a Python file with JAX x64 enabled."""
    code = textwrap.dedent("""\
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        def vector_add(a: jax.Array, b: jax.Array) -> jax.Array:
            return a + b
    """)
    p = tmp_path / "jax_x64_func.py"
    p.write_text(code)
    return p


@pytest.fixture
def torch_func_file(tmp_path):
    """Create a Python file with PyTorch types."""
    code = textwrap.dedent("""\
        import torch

        def forward(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x @ weight
    """)
    p = tmp_path / "torch_func.py"
    p.write_text(code)
    return p


@pytest.fixture
def dict_return_file(tmp_path):
    """Create a Python file with dict return."""
    code = textwrap.dedent("""\
        def compute(x: float, y: float) -> dict[str, float]:
            return {"sum": x + y, "product": x * y}
    """)
    p = tmp_path / "dict_return.py"
    p.write_text(code)
    return p


@pytest.fixture
def default_values_file(tmp_path):
    """Create a Python file with default parameter values."""
    code = textwrap.dedent("""\
        def scale(x: float, factor: float = 2.0) -> float:
            return x * factor
    """)
    p = tmp_path / "defaults.py"
    p.write_text(code)
    return p


@pytest.fixture
def no_type_hint_file(tmp_path):
    """Create a Python file with missing type hints."""
    code = textwrap.dedent("""\
        def broken(x, y: float) -> float:
            return x + y
    """)
    p = tmp_path / "no_hints.py"
    p.write_text(code)
    return p


@pytest.fixture
def no_return_hint_file(tmp_path):
    """Create a Python file with missing return type hint."""
    code = textwrap.dedent("""\
        def broken(x: float, y: float):
            return x + y
    """)
    p = tmp_path / "no_return_hint.py"
    p.write_text(code)
    return p


@pytest.fixture
def varargs_file(tmp_path):
    """Create a Python file with *args."""
    code = textwrap.dedent("""\
        def broken(*args: float) -> float:
            return sum(args)
    """)
    p = tmp_path / "varargs.py"
    p.write_text(code)
    return p


@pytest.fixture
def kwargs_file(tmp_path):
    """Create a Python file with **kwargs."""
    code = textwrap.dedent("""\
        def broken(**kwargs: float) -> float:
            return sum(kwargs.values())
    """)
    p = tmp_path / "kwargs.py"
    p.write_text(code)
    return p


# --- Structured type fixtures ---


@pytest.fixture
def namedtuple_single_param_file(tmp_path):
    """NamedTuple with single param (should flatten)."""
    code = textwrap.dedent("""\
        from typing import NamedTuple

        class Point(NamedTuple):
            x: float
            y: float

        def translate(p: Point) -> Point:
            return Point(x=p.x + 1, y=p.y + 1)
    """)
    p = tmp_path / "nt_single.py"
    p.write_text(code)
    return p


@pytest.fixture
def namedtuple_multi_param_file(tmp_path):
    """NamedTuple with multiple params (should nest input, flatten output)."""
    code = textwrap.dedent("""\
        from typing import NamedTuple

        class Point(NamedTuple):
            x: float
            y: float

        def translate(p: Point, dx: float) -> Point:
            return Point(x=p.x + dx, y=p.y)
    """)
    p = tmp_path / "nt_multi.py"
    p.write_text(code)
    return p


@pytest.fixture
def dataclass_file(tmp_path):
    """Dataclass with defaults."""
    code = textwrap.dedent("""\
        from dataclasses import dataclass

        @dataclass
        class Config:
            width: int
            height: int
            scale: float = 1.0

        def resize(cfg: Config) -> Config:
            return Config(
                width=int(cfg.width * cfg.scale),
                height=int(cfg.height * cfg.scale),
                scale=cfg.scale,
            )
    """)
    p = tmp_path / "dc_func.py"
    p.write_text(code)
    return p


@pytest.fixture
def plain_class_file(tmp_path):
    """Plain class (should error)."""
    code = textwrap.dedent("""\
        class Foo:
            def __init__(self, x: float):
                self.x = x

        def process(f: Foo) -> float:
            return f.x
    """)
    p = tmp_path / "plain_cls.py"
    p.write_text(code)
    return p


@pytest.fixture
def basemodel_file(tmp_path):
    """Pydantic BaseModel."""
    code = textwrap.dedent("""\
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str
            price: float

        def discount(item: Item) -> Item:
            return Item(name=item.name, price=item.price * 0.9)
    """)
    p = tmp_path / "bm_func.py"
    p.write_text(code)
    return p


@pytest.fixture
def typeddict_file(tmp_path):
    """TypedDict."""
    code = textwrap.dedent("""\
        from typing import TypedDict

        class Coords(TypedDict):
            lat: float
            lon: float

        def shift(c: Coords) -> Coords:
            return {"lat": c["lat"] + 1, "lon": c["lon"] + 1}
    """)
    p = tmp_path / "td_func.py"
    p.write_text(code)
    return p


@pytest.fixture
def list_of_structured_return_file(tmp_path):
    """Return type is list[Point] - should NOT flatten output."""
    code = textwrap.dedent("""\
        from typing import NamedTuple

        class Point(NamedTuple):
            x: float
            y: float

        def make_points(n: int) -> list[Point]:
            return [Point(x=float(i), y=float(i)) for i in range(n)]
    """)
    p = tmp_path / "list_struct.py"
    p.write_text(code)
    return p


@pytest.fixture
def subclass_namedtuple_file(tmp_path):
    """Subclass of a NamedTuple-derived class."""
    code = textwrap.dedent("""\
        from typing import NamedTuple

        class Point(NamedTuple):
            x: float
            y: float

        class Point3D(Point):
            z: float

        def elevate(p: Point3D) -> Point3D:
            return Point3D(x=p.x, y=p.y, z=p.z + 1)
    """)
    p = tmp_path / "subclass_nt.py"
    p.write_text(code)
    return p


@pytest.fixture
def dataclass_variants_file(tmp_path):
    """Dataclass with different decorator variants."""
    code = textwrap.dedent("""\
        import dataclasses

        @dataclasses.dataclass
        class A:
            x: float

        from dataclasses import dataclass

        @dataclass(frozen=True)
        class B:
            y: float

        @dataclass
        class C:
            z: float

        def combine(a: A, b: B, c: C) -> float:
            return a.x + b.y + c.z
    """)
    p = tmp_path / "dc_variants.py"
    p.write_text(code)
    return p


# --- Tests for parse_fromfunc_arg ---


class TestParseFromfuncArg:
    def test_valid_format(self, simple_func_file):
        path, name = fromfunc.parse_fromfunc_arg(f"{simple_func_file}::add")
        assert path == simple_func_file
        assert name == "add"

    def test_missing_separator(self):
        with pytest.raises(ValueError, match="Expected format"):
            fromfunc.parse_fromfunc_arg("file.py:func")

    def test_empty_parts(self):
        with pytest.raises(ValueError, match="Expected format"):
            fromfunc.parse_fromfunc_arg("::func")

    def test_empty_func_name(self):
        with pytest.raises(ValueError, match="Expected format"):
            fromfunc.parse_fromfunc_arg("file.py::")

    def test_file_not_found(self):
        with pytest.raises(ValueError, match="Source file not found"):
            fromfunc.parse_fromfunc_arg("/nonexistent/file.py::func")

    def test_non_python_file(self, tmp_path):
        p = tmp_path / "file.txt"
        p.write_text("hello")
        with pytest.raises(ValueError, match="must be a Python file"):
            fromfunc.parse_fromfunc_arg(f"{p}::func")


# --- Tests for analyze_function ---


class TestAnalyzeFunction:
    def test_simple_function(self, simple_func_file):
        analysis = fromfunc.analyze_function(simple_func_file, "add")
        assert analysis.func_name == "add"
        assert len(analysis.parameters) == 2
        assert analysis.parameters[0].name == "x"
        assert analysis.parameters[0].annotation == "float"
        assert analysis.parameters[1].name == "y"
        assert analysis.parameters[1].annotation == "float"
        # Output uses return type hint
        assert len(analysis.output_fields) == 1
        assert analysis.output_fields[0].annotation == "float"

    def test_numpy_detection(self, numpy_func_file):
        analysis = fromfunc.analyze_function(numpy_func_file, "matmul")
        assert analysis.detected_framework == "numpy"
        assert len(analysis.parameters) == 2
        assert analysis.parameters[0].annotation == "np.ndarray"
        # Return type is np.ndarray
        assert analysis.output_fields[0].annotation == "np.ndarray"

    def test_jax_detection(self, jax_func_file):
        analysis = fromfunc.analyze_function(jax_func_file, "vector_add")
        assert analysis.detected_framework == "jax"

    def test_jax_x64_detection(self, jax_x64_func_file):
        analysis = fromfunc.analyze_function(jax_x64_func_file, "vector_add")
        assert analysis.detected_framework == "jax"
        assert analysis.has_x64 is True

    def test_torch_detection(self, torch_func_file):
        analysis = fromfunc.analyze_function(torch_func_file, "forward")
        assert analysis.detected_framework == "pytorch"

    def test_dict_return(self, dict_return_file):
        analysis = fromfunc.analyze_function(dict_return_file, "compute")
        assert len(analysis.output_fields) == 2
        assert analysis.output_fields[0].name == "sum"
        assert analysis.output_fields[0].annotation == "float"
        assert analysis.output_fields[1].name == "product"
        assert analysis.output_fields[1].annotation == "float"

    def test_named_variable_return(self, tmp_path):
        code = textwrap.dedent("""\
            def compute(x: float) -> float:
                result = x * 2
                return result
        """)
        p = tmp_path / "named_return.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "compute")
        assert len(analysis.output_fields) == 1
        assert analysis.output_fields[0].name == "result"
        assert analysis.output_fields[0].annotation == "float"

    def test_bare_expression_return(self, tmp_path):
        code = textwrap.dedent("""\
            def compute(x: float, y: float) -> float:
                return x + y
        """)
        p = tmp_path / "bare_return.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "compute")
        assert len(analysis.output_fields) == 1
        assert analysis.output_fields[0].name == "result"
        assert analysis.output_fields[0].annotation == "float"

    def test_tuple_return(self, tmp_path):
        code = textwrap.dedent("""\
            def compute(x: float) -> tuple[float, int]:
                a = x + 1
                b = int(x * 2)
                return a, b
        """)
        p = tmp_path / "tuple_return.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "compute")
        assert len(analysis.output_fields) == 2
        assert analysis.output_fields[0].name == "a"
        assert analysis.output_fields[0].annotation == "float"
        assert analysis.output_fields[1].name == "b"
        assert analysis.output_fields[1].annotation == "int"

    def test_default_values(self, default_values_file):
        analysis = fromfunc.analyze_function(default_values_file, "scale")
        assert len(analysis.parameters) == 2
        assert analysis.parameters[0].default is None
        assert analysis.parameters[1].default == "2.0"

    def test_missing_type_hint_error(self, no_type_hint_file):
        with pytest.raises(ValueError, match="missing a type annotation"):
            fromfunc.analyze_function(no_type_hint_file, "broken")

    def test_missing_return_type_hint_error(self, no_return_hint_file):
        with pytest.raises(ValueError, match="missing a return type annotation"):
            fromfunc.analyze_function(no_return_hint_file, "broken")

    def test_bare_dict_return_type_error(self, tmp_path):
        code = textwrap.dedent("""\
            def compute(x: float) -> dict:
                return {"a": x}
        """)
        p = tmp_path / "bare_dict.py"
        p.write_text(code)
        with pytest.raises(ValueError, match="dict\\[str, Any\\]"):
            fromfunc.analyze_function(p, "compute")

    def test_bare_tuple_return_type_error(self, tmp_path):
        code = textwrap.dedent("""\
            def compute(x: float) -> tuple:
                return (x, x + 1)
        """)
        p = tmp_path / "bare_tuple.py"
        p.write_text(code)
        with pytest.raises(ValueError, match="too vague"):
            fromfunc.analyze_function(p, "compute")

    def test_varargs_error(self, varargs_file):
        with pytest.raises(ValueError, match="\\*args"):
            fromfunc.analyze_function(varargs_file, "broken")

    def test_kwargs_error(self, kwargs_file):
        with pytest.raises(ValueError, match="\\*\\*kwargs"):
            fromfunc.analyze_function(kwargs_file, "broken")

    def test_function_not_found(self, simple_func_file):
        with pytest.raises(ValueError, match="not found"):
            fromfunc.analyze_function(simple_func_file, "nonexistent")

    def test_module_name(self, simple_func_file):
        analysis = fromfunc.analyze_function(simple_func_file, "add")
        assert analysis.module_name == "simple"

    def test_kwonly_args(self, tmp_path):
        code = textwrap.dedent("""\
            def func(x: float, *, scale: float = 1.0) -> float:
                return x * scale
        """)
        p = tmp_path / "kwonly.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "func")
        assert len(analysis.parameters) == 2
        assert analysis.parameters[1].name == "scale"
        assert analysis.parameters[1].default == "1.0"

    def test_stores_ast_tree_and_func_node(self, simple_func_file):
        analysis = fromfunc.analyze_function(simple_func_file, "add")
        assert analysis._ast_tree is not None
        assert analysis._func_node is not None
        assert analysis._func_node.name == "add"


# --- Tests for map_type_annotation ---


class TestMapTypeAnnotation:
    def test_python_primitives_passthrough(self):
        """Python primitives pass through as-is."""
        assert fromfunc.map_type_annotation("float") == "float"
        assert fromfunc.map_type_annotation("int") == "int"
        assert fromfunc.map_type_annotation("bool") == "bool"
        assert fromfunc.map_type_annotation("str") == "str"
        assert fromfunc.map_type_annotation("bytes") == "bytes"

    def test_generic_types_passthrough(self):
        """Generic built-in types pass through for Pydantic to handle."""
        assert fromfunc.map_type_annotation("list[float]") == "list[float]"
        assert fromfunc.map_type_annotation("dict[str, int]") == "dict[str, int]"
        assert fromfunc.map_type_annotation("tuple[int, ...]") == "tuple[int, ...]"
        assert fromfunc.map_type_annotation("set[str]") == "set[str]"
        assert fromfunc.map_type_annotation("float | None") == "float | None"

    def test_typing_constructs_passthrough(self):
        """Typing module types pass through."""
        assert fromfunc.map_type_annotation("Optional[float]") == "Optional[float]"
        assert fromfunc.map_type_annotation("Union[int, float]") == "Union[int, float]"
        assert fromfunc.map_type_annotation("Any") == "Any"
        assert fromfunc.map_type_annotation("Literal['a', 'b']") == "Literal['a', 'b']"

    def test_numpy_types(self):
        assert fromfunc.map_type_annotation("np.ndarray") == "Array[..., Float64]"
        assert fromfunc.map_type_annotation("numpy.ndarray") == "Array[..., Float64]"
        assert fromfunc.map_type_annotation("NDArray") == "Array[..., Float64]"

    def test_jax_types_no_x64(self):
        assert fromfunc.map_type_annotation("jax.Array") == "Array[..., Float32]"
        assert fromfunc.map_type_annotation("jnp.ndarray") == "Array[..., Float32]"

    def test_jax_types_with_x64(self):
        assert (
            fromfunc.map_type_annotation("jax.Array", has_x64=True)
            == "Array[..., Float64]"
        )

    def test_torch_types(self):
        assert fromfunc.map_type_annotation("torch.Tensor") == "Array[..., Float32]"
        assert fromfunc.map_type_annotation("Tensor") == "Array[..., Float32]"

    def test_unknown_type_passes_through(self):
        """Unknown types pass through instead of erroring."""
        assert fromfunc.map_type_annotation("MyCustomClass") == "MyCustomClass"
        assert fromfunc.map_type_annotation("pathlib.Path") == "pathlib.Path"


# --- Tests for detect_recipe_from_framework ---


class TestDetectRecipeFromFramework:
    def test_jax(self):
        assert fromfunc.detect_recipe_from_framework("jax") == "jax"

    def test_pytorch(self):
        assert fromfunc.detect_recipe_from_framework("pytorch") == "pytorch"

    def test_numpy(self):
        assert fromfunc.detect_recipe_from_framework("numpy") == "base"

    def test_none(self):
        assert fromfunc.detect_recipe_from_framework(None) == "base"


# --- Tests for _extract_referenced_names ---


class TestExtractReferencedNames:
    def _parse_annotation(self, code: str) -> ast.expr:
        """Helper: parse a type annotation string into an AST node."""
        tree = ast.parse(f"x: {code}", mode="exec")
        return tree.body[0].annotation

    def test_primitives_return_empty(self):
        node = self._parse_annotation("float")
        assert fromfunc._extract_referenced_names(node) == set()

    def test_bare_class_name(self):
        node = self._parse_annotation("Point")
        assert fromfunc._extract_referenced_names(node) == {"Point"}

    def test_list_of_class(self):
        node = self._parse_annotation("list[Point]")
        assert fromfunc._extract_referenced_names(node) == {"Point"}

    def test_optional_class(self):
        node = self._parse_annotation("Optional[Point]")
        assert fromfunc._extract_referenced_names(node) == {"Point"}

    def test_ndarray_excluded(self):
        node = self._parse_annotation("NDArray")
        assert fromfunc._extract_referenced_names(node) == set()

    def test_tensor_excluded(self):
        node = self._parse_annotation("Tensor")
        assert fromfunc._extract_referenced_names(node) == set()

    def test_typing_names_excluded(self):
        node = self._parse_annotation("Union[int, str]")
        assert fromfunc._extract_referenced_names(node) == set()

    def test_multiple_classes(self):
        node = self._parse_annotation("dict[Point, Vector]")
        assert fromfunc._extract_referenced_names(node) == {"Point", "Vector"}

    def test_builtin_names_excluded(self):
        node = self._parse_annotation("dict[str, int]")
        assert fromfunc._extract_referenced_names(node) == set()


# --- Tests for _classify_class ---


class TestClassifyClass:
    def _get_class_and_map(self, code: str, class_name: str | None = None):
        """Parse code and return (class_node, source_name_map)."""
        tree = ast.parse(textwrap.dedent(code))
        name_map = fromfunc._build_source_name_map(tree)
        if class_name is None:
            # Get first class
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    return node, name_map
        else:
            return name_map[class_name].class_node, name_map

    def test_namedtuple(self):
        node, nmap = self._get_class_and_map("""\
            from typing import NamedTuple
            class Point(NamedTuple):
                x: float
                y: float
        """)
        assert fromfunc._classify_class(node, nmap) == "namedtuple"

    def test_typeddict(self):
        node, nmap = self._get_class_and_map("""\
            from typing import TypedDict
            class Coords(TypedDict):
                lat: float
                lon: float
        """)
        assert fromfunc._classify_class(node, nmap) == "typeddict"

    def test_basemodel(self):
        node, nmap = self._get_class_and_map("""\
            from pydantic import BaseModel
            class Item(BaseModel):
                name: str
        """)
        assert fromfunc._classify_class(node, nmap) == "basemodel"

    def test_dataclass_decorator(self):
        node, nmap = self._get_class_and_map("""\
            from dataclasses import dataclass
            @dataclass
            class Cfg:
                x: float
        """)
        assert fromfunc._classify_class(node, nmap) == "dataclass"

    def test_dataclasses_dot_dataclass(self):
        node, nmap = self._get_class_and_map("""\
            import dataclasses
            @dataclasses.dataclass
            class Cfg:
                x: float
        """)
        assert fromfunc._classify_class(node, nmap) == "dataclass"

    def test_dataclass_with_args(self):
        node, nmap = self._get_class_and_map("""\
            from dataclasses import dataclass
            @dataclass(frozen=True)
            class Cfg:
                x: float
        """)
        assert fromfunc._classify_class(node, nmap) == "dataclass"

    def test_plain_class(self):
        node, nmap = self._get_class_and_map("""\
            class Foo:
                pass
        """)
        assert fromfunc._classify_class(node, nmap) == "plain"

    def test_subclass_inherits_kind(self):
        node, nmap = self._get_class_and_map(
            """\
            from typing import NamedTuple
            class Point(NamedTuple):
                x: float
                y: float
            class Point3D(Point):
                z: float
        """,
            "Point3D",
        )
        assert fromfunc._classify_class(node, nmap) == "namedtuple"


# --- Tests for _extract_class_fields ---


class TestExtractClassFields:
    def _get_class_node(self, code: str) -> ast.ClassDef:
        tree = ast.parse(textwrap.dedent(code))
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                return node

    def test_namedtuple_fields(self):
        node = self._get_class_node("""\
            from typing import NamedTuple
            class Point(NamedTuple):
                x: float
                y: float
        """)
        fields = fromfunc._extract_class_fields(node, "namedtuple")
        assert fields == [("x", "float", None, None), ("y", "float", None, None)]

    def test_dataclass_with_defaults(self):
        node = self._get_class_node("""\
            from dataclasses import dataclass
            @dataclass
            class Cfg:
                width: int
                height: int
                scale: float = 1.0
        """)
        fields = fromfunc._extract_class_fields(node, "dataclass")
        assert fields == [
            ("width", "int", None, None),
            ("height", "int", None, None),
            ("scale", "float", "1.0", None),
        ]

    def test_basemodel_fields(self):
        node = self._get_class_node("""\
            from pydantic import BaseModel
            class Item(BaseModel):
                name: str
                price: float = 0.0
        """)
        fields = fromfunc._extract_class_fields(node, "basemodel")
        assert fields == [("name", "str", None, None), ("price", "float", "0.0", None)]


# --- Tests for flatten behavior ---


class TestFlattenBehavior:
    def test_single_namedtuple_param_flattens_input(self, namedtuple_single_param_file):
        analysis = fromfunc.analyze_function(namedtuple_single_param_file, "translate")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        # Input should have x and y flattened from Point
        assert "x: float" in tvars["input_schema_fields"]
        assert "y: float" in tvars["input_schema_fields"]
        assert "Point" not in tvars["input_schema_fields"]

    def test_single_namedtuple_param_flattens_output(
        self, namedtuple_single_param_file
    ):
        analysis = fromfunc.analyze_function(namedtuple_single_param_file, "translate")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        # Output should have x and y flattened from Point
        assert "x: float" in tvars["output_schema_fields"]
        assert "y: float" in tvars["output_schema_fields"]

    def test_multi_param_does_not_flatten_input(self, namedtuple_multi_param_file):
        analysis = fromfunc.analyze_function(namedtuple_multi_param_file, "translate")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        # Input should have p: Point (nested) and dx: float
        assert "p: Point" in tvars["input_schema_fields"]
        assert "dx: float" in tvars["input_schema_fields"]

    def test_multi_param_flattens_output(self, namedtuple_multi_param_file):
        analysis = fromfunc.analyze_function(namedtuple_multi_param_file, "translate")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        # Output should still be flattened (single return type)
        assert "x: float" in tvars["output_schema_fields"]
        assert "y: float" in tvars["output_schema_fields"]

    def test_list_return_not_flattened(self, list_of_structured_return_file):
        analysis = fromfunc.analyze_function(
            list_of_structured_return_file, "make_points"
        )
        tvars = fromfunc.generate_template_vars(analysis, "base")
        # Output should NOT be flattened (list[Point] is generic wrapper)
        assert "list[Point]" in tvars["output_schema_fields"]

    def test_multi_param_imports_nested_type(self, namedtuple_multi_param_file):
        analysis = fromfunc.analyze_function(namedtuple_multi_param_file, "translate")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        # Point should be imported alongside the function
        assert "Point" in tvars["extra_imports"]
        assert "from nt_multi import" in tvars["extra_imports"]


# --- Tests for generate_template_vars ---


class TestGenerateTemplateVars:
    def test_base_recipe(self, simple_func_file):
        analysis = fromfunc.analyze_function(simple_func_file, "add")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        assert "extra_imports" in tvars
        assert "from simple import add" in tvars["extra_imports"]
        assert "input_schema_fields" in tvars
        assert "x: float" in tvars["input_schema_fields"]
        assert "y: float" in tvars["input_schema_fields"]
        assert "output_schema_fields" in tvars
        assert "result: float" in tvars["output_schema_fields"]
        assert "apply_body" in tvars
        assert "add(x, y)" in tvars["apply_body"]

    def test_jax_recipe(self, jax_func_file):
        analysis = fromfunc.analyze_function(jax_func_file, "vector_add")
        tvars = fromfunc.generate_template_vars(analysis, "jax")
        assert "apply_jit_body" in tvars
        assert "vector_add(a, b)" in tvars["apply_jit_body"]
        assert "Differentiable[" in tvars["input_schema_fields"]
        # Output type from return hint (jax.Array -> Array[..., Float32])
        assert "Differentiable[Array[..., Float32]]" in tvars["output_schema_fields"]

    def test_pytorch_recipe(self, torch_func_file):
        analysis = fromfunc.analyze_function(torch_func_file, "forward")
        tvars = fromfunc.generate_template_vars(analysis, "pytorch")
        assert "evaluate_body" in tvars
        assert "forward(x, weight)" in tvars["evaluate_body"]
        assert "Differentiable[" in tvars["input_schema_fields"]
        # Output type from return hint (torch.Tensor -> Array[..., Float32])
        assert "Differentiable[Array[..., Float32]]" in tvars["output_schema_fields"]

    def test_default_values_in_schema(self, default_values_file):
        analysis = fromfunc.analyze_function(default_values_file, "scale")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        assert "factor: float = 2.0" in tvars["input_schema_fields"]

    def test_dict_return_multi_output(self, dict_return_file):
        analysis = fromfunc.analyze_function(dict_return_file, "compute")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        assert "sum: float" in tvars["output_schema_fields"]
        assert "product: float" in tvars["output_schema_fields"]

    def test_numpy_arrays_are_differentiable(self, numpy_func_file):
        """Numpy arrays should be wrapped with Differentiable."""
        analysis = fromfunc.analyze_function(numpy_func_file, "matmul")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        assert "Differentiable[Array[..., Float64]]" in tvars["input_schema_fields"]
        assert "Differentiable[Array[..., Float64]]" in tvars["output_schema_fields"]
        # Differentiable should be imported
        assert "Differentiable" in tvars["extra_imports"]

    def test_typing_imports_detected(self, tmp_path):
        """Typing constructs like Optional get auto-imported."""
        code = textwrap.dedent("""\
            from typing import Optional

            def func(x: float, name: Optional[str] = None) -> float:
                return x
        """)
        p = tmp_path / "typing_func.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "func")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        assert "from typing import Optional" in tvars["extra_imports"]

    def test_builtin_generics_no_typing_import(self, tmp_path):
        """Built-in generics like list[float] don't need typing imports."""
        code = textwrap.dedent("""\
            def func(items: list[float]) -> list[float]:
                return items
        """)
        p = tmp_path / "builtin_generics.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "func")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        assert "from typing import" not in tvars["extra_imports"]
        assert "list[float]" in tvars["input_schema_fields"]
        assert "list[float]" in tvars["output_schema_fields"]


# --- Tests for structured type wrapper body ---


class TestStructuredWrapperBody:
    def test_flattened_input_reconstructs_class(self, namedtuple_single_param_file):
        analysis = fromfunc.analyze_function(namedtuple_single_param_file, "translate")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        body = tvars["apply_body"]
        # Should reconstruct Point from flattened fields
        assert "p = Point(x=inputs.x, y=inputs.y)" in body
        assert "translate(p)" in body

    def test_flattened_output_unpacks_result(self, namedtuple_single_param_file):
        analysis = fromfunc.analyze_function(namedtuple_single_param_file, "translate")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        body = tvars["apply_body"]
        # Should unpack Point fields from result
        assert "OutputSchema(x=result.x, y=result.y)" in body

    def test_nested_input_direct_access(self, namedtuple_multi_param_file):
        analysis = fromfunc.analyze_function(namedtuple_multi_param_file, "translate")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        body = tvars["apply_body"]
        # Should access nested param directly
        assert "p = inputs.p" in body
        assert "dx = inputs.dx" in body
        assert "translate(p, dx)" in body

    def test_nested_input_flattened_output(self, namedtuple_multi_param_file):
        analysis = fromfunc.analyze_function(namedtuple_multi_param_file, "translate")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        body = tvars["apply_body"]
        # Output still flattened
        assert "OutputSchema(x=result.x, y=result.y)" in body


# --- Tests for full generation (valid Python output) ---


class TestFullGeneration:
    @pytest.mark.parametrize("recipe", ["base", "jax", "pytorch"])
    def test_generated_output_is_valid_python(self, simple_func_file, recipe):
        """Verify the generated tesseract_api.py is valid Python."""
        analysis = fromfunc.analyze_function(simple_func_file, "add")
        tvars = fromfunc.generate_template_vars(analysis, recipe)

        target_dir = simple_func_file.parent / "output"
        target_dir.mkdir()

        api_path = engine.init_api(
            target_dir,
            "test_tesseract",
            recipe=recipe,
            fromfunc_vars=tvars,
            source_file=simple_func_file,
        )

        # Verify the source file was copied
        assert (target_dir / simple_func_file.name).exists()

        # Verify the generated file is valid Python
        content = api_path.read_text()
        ast.parse(content)

    def test_base_generation_complete(self, simple_func_file):
        """Full base recipe generation produces expected structure."""
        analysis = fromfunc.analyze_function(simple_func_file, "add")
        tvars = fromfunc.generate_template_vars(analysis, "base")

        target_dir = simple_func_file.parent / "output"
        target_dir.mkdir()

        api_path = engine.init_api(
            target_dir,
            "test_tesseract",
            recipe="base",
            fromfunc_vars=tvars,
            source_file=simple_func_file,
        )

        content = api_path.read_text()
        assert "from simple import add" in content
        assert "class InputSchema" in content
        assert "class OutputSchema" in content
        assert "x: float" in content
        assert "y: float" in content
        assert "def apply(inputs: InputSchema)" in content

    def test_jax_generation_with_x64(self, jax_x64_func_file):
        """JAX recipe with x64 uses Float64."""
        analysis = fromfunc.analyze_function(jax_x64_func_file, "vector_add")
        tvars = fromfunc.generate_template_vars(analysis, "jax")

        assert "Float64" in tvars["input_schema_fields"]
        assert "Float64" in tvars["output_schema_fields"]

    def test_without_fromfunc_preserves_defaults(self, tmp_path):
        """Verify init without --fromfunc still produces valid default templates."""
        for recipe in ["base", "jax", "pytorch"]:
            target_dir = tmp_path / f"default_{recipe}"
            target_dir.mkdir()
            api_path = engine.init_api(target_dir, "test_default", recipe=recipe)
            content = api_path.read_text()
            ast.parse(content)
            assert "class InputSchema" in content
            assert "class OutputSchema" in content

    def test_numpy_differentiable_generation(self, numpy_func_file):
        """Numpy arrays are differentiable in generated output."""
        analysis = fromfunc.analyze_function(numpy_func_file, "matmul")
        tvars = fromfunc.generate_template_vars(analysis, "base")

        target_dir = numpy_func_file.parent / "output"
        target_dir.mkdir()

        api_path = engine.init_api(
            target_dir,
            "test_np",
            recipe="base",
            fromfunc_vars=tvars,
            source_file=numpy_func_file,
        )

        content = api_path.read_text()
        ast.parse(content)
        assert "Differentiable[Array[..., Float64]]" in content
        assert "from tesseract_core.runtime import" in content

    def test_dict_return_output_types_from_hint(self, dict_return_file):
        """Dict return type hint provides value type for all output fields."""
        analysis = fromfunc.analyze_function(dict_return_file, "compute")
        tvars = fromfunc.generate_template_vars(analysis, "base")

        target_dir = dict_return_file.parent / "output"
        target_dir.mkdir()

        api_path = engine.init_api(
            target_dir,
            "test_dict",
            recipe="base",
            fromfunc_vars=tvars,
            source_file=dict_return_file,
        )

        content = api_path.read_text()
        ast.parse(content)
        assert "sum: float" in content
        assert "product: float" in content

    def test_tuple_return_output_types(self, tmp_path):
        """Tuple return type hint provides per-element types."""
        code = textwrap.dedent("""\
            import numpy as np

            def compute(x: float) -> tuple[float, np.ndarray]:
                return x, np.array([x])
        """)
        p = tmp_path / "tuple_func.py"
        p.write_text(code)

        analysis = fromfunc.analyze_function(p, "compute")
        tvars = fromfunc.generate_template_vars(analysis, "base")

        target_dir = tmp_path / "output"
        target_dir.mkdir()

        api_path = engine.init_api(
            target_dir,
            "test_tuple",
            recipe="base",
            fromfunc_vars=tvars,
            source_file=p,
        )

        content = api_path.read_text()
        ast.parse(content)
        # First element: named "x" from `return x, np.array([x])`
        # float (not array, no Differentiable)
        assert "x: float" in content
        # Second element: unnamed -> result2
        # np.ndarray -> Differentiable[Array[..., Float64]]
        assert "result2: Differentiable[Array[..., Float64]]" in content


# --- Integration tests ---


class TestIntegration:
    def test_end_to_end_base(self, tmp_path):
        """Full end-to-end: create source file, analyze, generate, validate."""
        source = textwrap.dedent("""\
            def multiply(x: float, y: float) -> float:
                return x * y
        """)
        src_file = tmp_path / "multiply.py"
        src_file.write_text(source)

        file_path, func_name = fromfunc.parse_fromfunc_arg(f"{src_file}::multiply")
        analysis = fromfunc.analyze_function(file_path, func_name)
        recipe = fromfunc.detect_recipe_from_framework(analysis.detected_framework)
        assert recipe == "base"

        tvars = fromfunc.generate_template_vars(analysis, recipe)
        target_dir = tmp_path / "output"
        target_dir.mkdir()
        api_path = engine.init_api(
            target_dir,
            "multiply_tesseract",
            recipe=recipe,
            fromfunc_vars=tvars,
            source_file=analysis.source_file,
        )

        assert api_path.exists()
        assert (target_dir / "multiply.py").exists()
        assert (target_dir / "tesseract_config.yaml").exists()
        assert (target_dir / "tesseract_requirements.txt").exists()

        content = api_path.read_text()
        ast.parse(content)
        assert "from multiply import multiply" in content
        assert "x: float" in content
        assert "multiply(x, y)" in content
        # Output type from return hint
        assert "result: float" in content

    def test_end_to_end_numpy_differentiable(self, tmp_path):
        """Numpy functions produce differentiable arrays."""
        source = textwrap.dedent("""\
            import numpy as np

            def dot_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                return np.dot(a, b)
        """)
        src_file = tmp_path / "dot.py"
        src_file.write_text(source)

        _, func_name = fromfunc.parse_fromfunc_arg(f"{src_file}::dot_product")
        analysis = fromfunc.analyze_function(src_file, func_name)

        recipe = fromfunc.detect_recipe_from_framework(analysis.detected_framework)
        assert recipe == "base"

        tvars = fromfunc.generate_template_vars(analysis, recipe)
        target_dir = tmp_path / "output"
        target_dir.mkdir()
        api_path = engine.init_api(
            target_dir,
            "dot_tesseract",
            recipe=recipe,
            fromfunc_vars=tvars,
            source_file=src_file,
        )

        content = api_path.read_text()
        ast.parse(content)
        assert "Differentiable[Array[..., Float64]]" in content

    def test_end_to_end_jax(self, tmp_path):
        """JAX function generates jax recipe with Differentiable."""
        source = textwrap.dedent("""\
            import jax
            import jax.numpy as jnp

            def my_jax_func(a: jax.Array, b: jax.Array) -> jax.Array:
                return a + b
        """)
        src_file = tmp_path / "jax_func.py"
        src_file.write_text(source)

        _, func_name = fromfunc.parse_fromfunc_arg(f"{src_file}::my_jax_func")
        analysis = fromfunc.analyze_function(src_file, func_name)

        recipe = fromfunc.detect_recipe_from_framework(analysis.detected_framework)
        assert recipe == "jax"

        tvars = fromfunc.generate_template_vars(analysis, recipe)
        target_dir = tmp_path / "output"
        target_dir.mkdir()
        api_path = engine.init_api(
            target_dir,
            "jax_tesseract",
            recipe=recipe,
            fromfunc_vars=tvars,
            source_file=src_file,
        )

        content = api_path.read_text()
        ast.parse(content)
        assert "Differentiable[Array[..., Float32]]" in content
        assert "from jax_func import my_jax_func" in content

    def test_end_to_end_generic_types(self, tmp_path):
        """Functions with generic types (list, Optional) work end-to-end."""
        source = textwrap.dedent("""\
            from typing import Optional

            def process(items: list[float], threshold: Optional[float] = None) -> list[float]:
                if threshold is not None:
                    items = [x for x in items if x > threshold]
                return items
        """)
        src_file = tmp_path / "generic_func.py"
        src_file.write_text(source)

        _, func_name = fromfunc.parse_fromfunc_arg(f"{src_file}::process")
        analysis = fromfunc.analyze_function(src_file, func_name)
        tvars = fromfunc.generate_template_vars(analysis, "base")

        target_dir = tmp_path / "output"
        target_dir.mkdir()
        api_path = engine.init_api(
            target_dir,
            "generic_tesseract",
            recipe="base",
            fromfunc_vars=tvars,
            source_file=src_file,
        )

        content = api_path.read_text()
        ast.parse(content)
        assert "items: list[float]" in content
        assert "threshold: Optional[float] = None" in content
        # return is `return items` -> field name is "items"
        assert "from typing import Optional" in content

    def test_end_to_end_dict_return_with_array_values(self, tmp_path):
        """Dict return with array value type gets Differentiable wrapping."""
        source = textwrap.dedent("""\
            import numpy as np

            def compute(x: np.ndarray) -> dict[str, np.ndarray]:
                return {"doubled": x * 2, "squared": x ** 2}
        """)
        src_file = tmp_path / "dict_array.py"
        src_file.write_text(source)

        analysis = fromfunc.analyze_function(src_file, "compute")
        tvars = fromfunc.generate_template_vars(analysis, "base")

        target_dir = tmp_path / "output"
        target_dir.mkdir()
        api_path = engine.init_api(
            target_dir,
            "dict_array_tesseract",
            recipe="base",
            fromfunc_vars=tvars,
            source_file=src_file,
        )

        content = api_path.read_text()
        ast.parse(content)
        assert "doubled: Differentiable[Array[..., Float64]]" in content
        assert "squared: Differentiable[Array[..., Float64]]" in content


# --- Structured type integration tests ---


class TestStructuredTypeIntegration:
    def test_single_namedtuple_full_generation(self, namedtuple_single_param_file):
        """Single NamedTuple param + return → fully flattened, valid Python."""
        analysis = fromfunc.analyze_function(namedtuple_single_param_file, "translate")
        tvars = fromfunc.generate_template_vars(analysis, "base")

        target_dir = namedtuple_single_param_file.parent / "output"
        target_dir.mkdir()
        api_path = engine.init_api(
            target_dir,
            "test_nt",
            recipe="base",
            fromfunc_vars=tvars,
            source_file=namedtuple_single_param_file,
        )

        content = api_path.read_text()
        ast.parse(content)
        # Flattened: x and y directly in schemas
        assert "x: float" in content
        assert "y: float" in content
        # Point should NOT be imported (it's flattened)
        assert "import Point" not in content
        # Reconstruction in apply body
        assert "Point(x=inputs.x, y=inputs.y)" in content
        # Unpacking in return
        assert "OutputSchema(x=result.x, y=result.y)" in content

    def test_dataclass_with_defaults_flattened(self, dataclass_file):
        """Dataclass param with defaults → flattened correctly."""
        analysis = fromfunc.analyze_function(dataclass_file, "resize")
        tvars = fromfunc.generate_template_vars(analysis, "base")

        # Input fields should include defaults
        assert "width: int" in tvars["input_schema_fields"]
        assert "height: int" in tvars["input_schema_fields"]
        assert "scale: float = 1.0" in tvars["input_schema_fields"]

        target_dir = dataclass_file.parent / "output"
        target_dir.mkdir()
        api_path = engine.init_api(
            target_dir,
            "test_dc",
            recipe="base",
            fromfunc_vars=tvars,
            source_file=dataclass_file,
        )
        content = api_path.read_text()
        ast.parse(content)

    def test_multi_param_nested_generation(self, namedtuple_multi_param_file):
        """Multi-param with structured type → nested, valid Python."""
        analysis = fromfunc.analyze_function(namedtuple_multi_param_file, "translate")
        tvars = fromfunc.generate_template_vars(analysis, "base")

        target_dir = namedtuple_multi_param_file.parent / "output"
        target_dir.mkdir()
        api_path = engine.init_api(
            target_dir,
            "test_nested",
            recipe="base",
            fromfunc_vars=tvars,
            source_file=namedtuple_multi_param_file,
        )

        content = api_path.read_text()
        ast.parse(content)
        # Point imported for nesting (func_name first, then sorted type names)
        assert "from nt_multi import translate, Point" in content
        assert "p: Point" in content
        assert "dx: float" in content

    def test_plain_class_error(self, plain_class_file):
        """Plain class → clear error."""
        analysis = fromfunc.analyze_function(plain_class_file, "process")
        with pytest.raises(ValueError, match="plain class"):
            fromfunc.generate_template_vars(analysis, "base")

    def test_structured_type_jax_recipe_error(self, tmp_path):
        """Structured type + jax recipe → clear error."""
        code = textwrap.dedent("""\
            import jax
            from typing import NamedTuple

            class Params(NamedTuple):
                w: float
                b: float

            def predict(p: Params) -> float:
                return p.w + p.b
        """)
        src_file = tmp_path / "jax_struct.py"
        src_file.write_text(code)

        analysis = fromfunc.analyze_function(src_file, "predict")
        with pytest.raises(ValueError, match=r"not supported.*jax"):
            fromfunc.generate_template_vars(analysis, "jax")

    def test_basemodel_single_param(self, basemodel_file):
        """BaseModel param → flattened correctly."""
        analysis = fromfunc.analyze_function(basemodel_file, "discount")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        assert "name: str" in tvars["input_schema_fields"]
        assert "price: float" in tvars["input_schema_fields"]

    def test_typeddict_single_param(self, typeddict_file):
        """TypedDict param → flattened correctly."""
        analysis = fromfunc.analyze_function(typeddict_file, "shift")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        assert "lat: float" in tvars["input_schema_fields"]
        assert "lon: float" in tvars["input_schema_fields"]

    def test_subclass_namedtuple(self, subclass_namedtuple_file):
        """Subclass of NamedTuple is detected and flattened."""
        analysis = fromfunc.analyze_function(subclass_namedtuple_file, "elevate")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        # Point3D only has z as an AnnAssign, but that's what's in its body
        assert "z: float" in tvars["input_schema_fields"]

    def test_list_of_structured_imports(self, list_of_structured_return_file):
        """list[Point] return → Point imported, not flattened."""
        analysis = fromfunc.analyze_function(
            list_of_structured_return_file, "make_points"
        )
        tvars = fromfunc.generate_template_vars(analysis, "base")
        # Point should be imported
        assert "Point" in tvars["extra_imports"]
        # Return type should be list[Point] (pass-through)
        assert "list[Point]" in tvars["output_schema_fields"]


# --- Tests for docstring parsing ---


def _make_func_node(source: str) -> ast.FunctionDef:
    """Parse source code and return the first FunctionDef node."""
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError("No function found")


class TestParseDocstring:
    def test_google_basic_args(self):
        node = _make_func_node('''\
            def f(x: float, y: float) -> float:
                """Do something.

                Args:
                    x: The first number.
                    y: The second number.
                """
                return x + y
        ''')
        params, returns = fromfunc._parse_docstring(node)
        assert params == {"x": "The first number.", "y": "The second number."}
        assert returns == {}

    def test_google_multiline_continuation(self):
        node = _make_func_node('''\
            def f(x: float) -> float:
                """Do something.

                Args:
                    x: The first number with
                        a long description.
                """
                return x
        ''')
        params, _ = fromfunc._parse_docstring(node)
        assert params == {"x": "The first number with a long description."}

    def test_google_args_and_returns(self):
        node = _make_func_node('''\
            def f(x: float) -> float:
                """Do something.

                Args:
                    x: The input value.

                Returns:
                    result: The computed output.
                """
                return x
        ''')
        params, returns = fromfunc._parse_docstring(node)
        assert params == {"x": "The input value."}
        assert returns == {"result": "The computed output."}

    def test_google_arguments_alias(self):
        node = _make_func_node('''\
            def f(x: float) -> float:
                """Do something.

                Arguments:
                    x: Input.
                """
                return x
        ''')
        params, _ = fromfunc._parse_docstring(node)
        assert params == {"x": "Input."}

    def test_numpy_basic(self):
        node = _make_func_node('''\
            def f(x: float, y: float) -> float:
                """Do something.

                Parameters
                ----------
                x : float
                    The first number.
                y : float
                    The second number.
                """
                return x + y
        ''')
        params, returns = fromfunc._parse_docstring(node)
        assert params == {"x": "The first number.", "y": "The second number."}
        assert returns == {}

    def test_numpy_with_returns(self):
        node = _make_func_node('''\
            def f(x: float) -> float:
                """Do something.

                Parameters
                ----------
                x : float
                    The input.

                Returns
                -------
                result : float
                    The output.
                """
                return x
        ''')
        params, returns = fromfunc._parse_docstring(node)
        assert params == {"x": "The input."}
        assert returns == {"result": "The output."}

    def test_sphinx_basic(self):
        node = _make_func_node('''\
            def f(x: float, y: float) -> float:
                """Do something.

                :param x: The first number.
                :param y: The second number.
                """
                return x + y
        ''')
        params, returns = fromfunc._parse_docstring(node)
        assert params == {"x": "The first number.", "y": "The second number."}
        assert returns == {}

    def test_sphinx_returns(self):
        node = _make_func_node('''\
            def f(x: float) -> float:
                """Do something.

                :param x: Input.
                :returns: The output value.
                """
                return x
        ''')
        params, returns = fromfunc._parse_docstring(node)
        assert params == {"x": "Input."}
        assert returns == {"result": "The output value."}

    def test_sphinx_return_alias(self):
        node = _make_func_node('''\
            def f(x: float) -> float:
                """Do something.

                :param x: Input.
                :return: The output value.
                """
                return x
        ''')
        _, returns = fromfunc._parse_docstring(node)
        assert returns == {"result": "The output value."}

    def test_no_docstring(self):
        node = _make_func_node("""\
            def f(x: float) -> float:
                return x
        """)
        params, returns = fromfunc._parse_docstring(node)
        assert params == {}
        assert returns == {}

    def test_unknown_format(self):
        node = _make_func_node('''\
            def f(x: float) -> float:
                """Just a plain description without any structured format."""
                return x
        ''')
        params, returns = fromfunc._parse_docstring(node)
        assert params == {}
        assert returns == {}

    def test_args_only_no_returns(self):
        node = _make_func_node('''\
            def f(x: float) -> float:
                """Do something.

                Args:
                    x: The input.
                """
                return x
        ''')
        params, returns = fromfunc._parse_docstring(node)
        assert params == {"x": "The input."}
        assert returns == {}


class TestDescriptionPropagation:
    def test_analyze_populates_descriptions(self, tmp_path):
        code = textwrap.dedent('''\
            def add(x: float, y: float) -> float:
                """Add two numbers.

                Args:
                    x: The first number.
                    y: The second number.

                Returns:
                    result: The sum.
                """
                return x + y
        ''')
        p = tmp_path / "doc_func.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "add")
        assert analysis.parameters[0].description == "The first number."
        assert analysis.parameters[1].description == "The second number."
        assert analysis.output_fields[0].description == "The sum."

    def test_description_in_field_strings(self, tmp_path):
        code = textwrap.dedent('''\
            def add(x: float, y: float) -> float:
                """Add.

                Args:
                    x: The first number.
                    y: The second number.
                """
                return x + y
        ''')
        p = tmp_path / "doc_func.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "add")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        assert 'Field(description="The first number.")' in tvars["input_schema_fields"]
        assert 'Field(description="The second number.")' in tvars["input_schema_fields"]

    def test_description_with_default(self, tmp_path):
        code = textwrap.dedent('''\
            def scale(x: float, factor: float = 2.0) -> float:
                """Scale.

                Args:
                    x: The value.
                    factor: The scale factor.
                """
                return x * factor
        ''')
        p = tmp_path / "doc_func.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "scale")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        assert (
            'Field(description="The scale factor.", default=2.0)'
            in tvars["input_schema_fields"]
        )

    def test_no_docstring_no_field(self, tmp_path):
        code = textwrap.dedent("""\
            def add(x: float, y: float) -> float:
                return x + y
        """)
        p = tmp_path / "doc_func.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "add")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        assert "Field(" not in tvars["input_schema_fields"]
        assert "Field(" not in tvars["output_schema_fields"]

    def test_field_import_only_when_needed(self, tmp_path):
        code_with_doc = textwrap.dedent('''\
            def add(x: float, y: float) -> float:
                """Add.

                Args:
                    x: First.
                    y: Second.
                """
                return x + y
        ''')
        p1 = tmp_path / "with_doc.py"
        p1.write_text(code_with_doc)
        analysis1 = fromfunc.analyze_function(p1, "add")
        tvars1 = fromfunc.generate_template_vars(analysis1, "base")
        assert "from pydantic import Field" in tvars1["extra_imports"]

        code_no_doc = textwrap.dedent("""\
            def add(x: float, y: float) -> float:
                return x + y
        """)
        p2 = tmp_path / "no_doc.py"
        p2.write_text(code_no_doc)
        analysis2 = fromfunc.analyze_function(p2, "add")
        tvars2 = fromfunc.generate_template_vars(analysis2, "base")
        assert "from pydantic import Field" not in tvars2["extra_imports"]

    def test_full_generation_valid_python(self, tmp_path):
        code = textwrap.dedent('''\
            def add(x: float, y: float) -> float:
                """Add two numbers.

                Args:
                    x: The first number.
                    y: The second number.

                Returns:
                    result: The sum.
                """
                return x + y
        ''')
        p = tmp_path / "doc_func.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "add")
        tvars = fromfunc.generate_template_vars(analysis, "base")

        target_dir = tmp_path / "output"
        target_dir.mkdir()
        api_path = engine.init_api(
            target_dir,
            "test_doc",
            recipe="base",
            fromfunc_vars=tvars,
            source_file=p,
        )
        content = api_path.read_text()
        ast.parse(content)
        assert 'Field(description="The first number.")' in content
        assert 'Field(description="The sum.")' in content
        assert "from pydantic import Field" in content


class TestDescriptionWithStructuredTypes:
    def test_flattened_input_no_param_descriptions(self, tmp_path):
        """Flattened input fields come from the class, not the function docstring."""
        code = textwrap.dedent('''\
            from typing import NamedTuple

            class Point(NamedTuple):
                x: float
                y: float

            def translate(p: Point) -> float:
                """Translate.

                Args:
                    p: The point to translate.
                """
                return p.x + p.y
        ''')
        p = tmp_path / "struct_doc.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "translate")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        # Flattened fields don't get param descriptions
        assert "Field(" not in tvars["input_schema_fields"]

    def test_nested_input_with_description(self, tmp_path):
        """Nested (non-flattened) param with description gets Field()."""
        code = textwrap.dedent('''\
            from typing import NamedTuple

            class Point(NamedTuple):
                x: float
                y: float

            def translate(p: Point, dx: float) -> float:
                """Translate.

                Args:
                    p: The point.
                    dx: The x offset.
                """
                return p.x + dx
        ''')
        p = tmp_path / "nested_doc.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "translate")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        assert 'Field(description="The point.")' in tvars["input_schema_fields"]
        assert 'Field(description="The x offset.")' in tvars["input_schema_fields"]

    def test_output_flatten_with_returns_docstring(self, tmp_path):
        """Flattened output fields come from the class, Returns docstring ignored."""
        code = textwrap.dedent('''\
            from typing import NamedTuple

            class Point(NamedTuple):
                x: float
                y: float

            def make_point(val: float) -> Point:
                """Make a point.

                Args:
                    val: The value.

                Returns:
                    result: A point.
                """
                return Point(x=val, y=val)
        ''')
        p = tmp_path / "out_struct.py"
        p.write_text(code)
        analysis = fromfunc.analyze_function(p, "make_point")
        tvars = fromfunc.generate_template_vars(analysis, "base")
        # Flattened output fields don't get return descriptions
        assert "Field(" not in tvars["output_schema_fields"]
