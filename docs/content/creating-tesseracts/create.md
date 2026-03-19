# Creating Tesseracts

This page walks through creating your own Tesseracts, starting from a basic example and building up to advanced patterns. We recommend reading the [Get Started page](tr-quickstart) first.

## Initialize a new Tesseract

Run the following to initialize everything needed to define a new Tesseract in the current directory:

```bash
$ tesseract init --name my_tesseract
```

This creates three files:

- `tesseract_api.py` — Python module defining the Tesseract's computations.
- `tesseract_config.yaml` — metadata (name, version), build options (base image, custom build steps, external data), and more.
- `tesseract_requirements.txt` — Python dependencies in [pip requirements file format](https://pip.pypa.io/en/stable/reference/requirements-file-format/).

Use `--target-dir [DIRECTORY]` to create these files elsewhere. Other useful options:

- `--recipe` — use a ready-made template for common scenarios (e.g., generating gradient endpoints from JAX functions).
- `--help` — list all available options and recipes.

## Define a simple Tesseract

The generated `tesseract_api.py` contains boilerplate code to guide you. Let's walk through it section by section, implementing a simple `helloworld` Tesseract that accepts a `name` and returns `"Hello {name}!"`.

The first section defines the input and output schemas:

```python
class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    pass
```

Input and output schemas are defined as [Pydantic](https://docs.pydantic.dev/latest/) models[^1]. For `helloworld`, we need a string in and a string out:

```python
class InputSchema(BaseModel):
    name: str = Field(
        description="Name of the person you want to greet."
    )

class OutputSchema(BaseModel):
    greeting: str = Field(description="A greeting!")
```

Field descriptions are optional but recommended — they appear in the auto-generated docs and schemas. You can also use default values, validators, and other Pydantic features (see [Pydantic docs](https://docs.pydantic.dev/latest/)).

Below the schemas, you'll find the required endpoints:

```python
def apply(inputs: InputSchema) -> OutputSchema:
    ...
```

Currently, only `apply` is required. This is where you define the Tesseract's core computation. For `helloworld`:

```python
def apply(inputs: InputSchema) -> OutputSchema:
    """Greet a person whose name is given as input."""
    return OutputSchema(greeting=f"Hello {inputs.name}")
```

```{note}
Docstrings on `apply` (and other endpoints) are included in the auto-generated docs.
```

The last section contains templates for optional endpoints:

```python
# def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
#     return {}

...
```

Leave these untouched for now — `helloworld` is not differentiable.

```{tip}
For a Tesseract with all optional endpoints implemented, see the [Univariate example](../examples/building-blocks/univariate.md).
```

Finally, set the name and version in `tesseract_config.yaml`:

```yaml
name: "helloworld"
version: "1.0.0"
description: "A sample Python app"
```

You're now ready to build your first Tesseract.

```{tip}
Before building, you can test locally without containers using `tesseract-runtime`. See [Debugging and Development](../misc/debugging.md) for details.
```

## Build a Tesseract

To build, run `tesseract build` from the directory containing `tesseract_api.py`:

```
$ tesseract build .
```

The Tesseract name comes from `tesseract_config.yaml`. By default, the image is tagged with both the specified version (`1.0.0`) and `latest`, so you can reference it as `helloworld:1.0.0` or `helloworld:latest`.

### View built Tesseracts

To list all locally available Tesseracts:

```bash
$ tesseract list
```

The output is a table of Tesseract images with their ID, name, version, and description:

```bash
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID                  ┃ Tags                  ┃ Name       ┃ Version ┃ Description                               ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ sha256:d4bdc2c29eb1 │ ['helloworld:latest'] │ helloworld │ 1.0.0   │ A sample Python app                       │
└─────────────────────┴───────────────────────┴────────────┴─────────┴───────────────────────────────────────────┘
```

## Arrays in the schema

N-dimensional arrays are central to scientific computing. Use the `tesseract_core.runtime.Array` type annotation to define them:

```python
from tesseract_core.runtime import Array, Float32

class InputSchema(BaseModel):
    x: Array[(3,), Float32] = Field(
        description="A 3D vector",
    )
    r: Array[(3, 3), Float32] = Field(
        description="A 3x3 matrix",
    )
    s: Float64 = Field(description="A scalar")
    v: Array[(None,), Float32] = Field(
        description="A vector of unknown length",
    )
    p: Array[..., Float32] = Field(
        description="An array of any shape",
    )
```

The first parameter is the `shape`, the second is the `dtype` — both follow NumPy conventions.
Inside a Tesseract, `Array` fields are cast to `numpy.ndarray` objects with the given dtype and shape, so standard NumPy operations work directly. For example, `r @ x + s` would multiply the matrix `r` by the vector `x` and add the broadcasted scalar `s`.

For scalars, use `tesseract_core.runtime.Float32`, `Float64`, `Int32`, etc. (see the [runtime API reference](../api/tesseract-runtime-api.md) for the full list). Plain `float` works but does not support the [differentiability features](tr-create-diff) described below.

## Nested schemas

Since `InputSchema` and `OutputSchema` are Pydantic `BaseModel`s, they support nesting other models within them:

```python
class Mesh(BaseModel):
    """A simple mesh schema."""
    points: Array[(None, 3), Float32]
    num_points_per_cell: Array[(None,), Float32]
    cell_connectivity: Array[(None,), Int32]

class InputSchema(BaseModel):
    wing_shape: Mesh
    propeller_shape: Mesh
```

## Dicts and lists in schemas

Schemas support `dict` and `list` containers for variable-length collections or dynamically keyed data.

### Dicts

Use `dict[str, ...]` to define a dictionary with string keys:

```python
class InputSchema(BaseModel):
    params: dict[str, Differentiable[Array[(None,), Float32]]] = Field(
        description="A dictionary of parameter arrays, e.g. {'x': array, 'y': array}.",
    )
```

Inside `apply`, dict fields are accessed by key:

```python
def apply(inputs: InputSchema) -> OutputSchema:
    x = inputs.params["x"]
    y = inputs.params["y"]
    ...
```

### Lists

Use `list[...]` to define a list of values:

```python
class InputSchema(BaseModel):
    coefficients: list[Differentiable[Array[(None,), Float32]]] = Field(
        description="A list of coefficient arrays.",
    )
```

Inside `apply`, list fields are accessed by index:

```python
def apply(inputs: InputSchema) -> OutputSchema:
    c0 = inputs.coefficients[0]
    c1 = inputs.coefficients[1]
    ...
```

### Combining dicts, lists, and nested models

These containers can be freely nested and combined with `BaseModel` subclasses:

```python
class NestedParams(BaseModel):
    z: Differentiable[Array[(5,), Float32]]
    gamma: dict[str, Differentiable[Array[(None,), Float32]]]


class InputSchema(BaseModel):
    alpha: dict[str, Differentiable[Array[(None,), Float32]]]
    beta: NestedParams
    delta: list[Differentiable[Array[(None,), Float32]]]


class OutputSchema(BaseModel):
    result: Differentiable[Array[(3,), Float32]]
    result_dict: dict[str, Differentiable[Array[(None,), Float32]]]
    result_list: list[Differentiable[Array[(None,), Float32]]]
```

```{tip}
Dicts and lists also work with non-differentiable types (e.g. `dict[str, Array[(None,), Float32]]`), and with non-array types (e.g. `dict[str, str]`, `list[int]`).
```

(tr-create-diff)=

## Differentiability

Tesseracts can expose endpoints for computing derivatives, making it possible to compose multiple Tesseracts into automatically differentiable workflows for shape optimization, model calibration, and more.

The {py:class}`tesseract_core.runtime.Differentiable` type annotation marks which inputs can be differentiated with respect to, and which outputs can be differentiated. All `Differentiable` outputs are considered differentiable with respect to all `Differentiable` inputs. Passing a non-differentiable field (e.g., `jac_inputs=["non_differentiable_arg"]`) raises a validation error before the endpoint runs.

For example:

```python
from tesseract_core.runtime import Differentiable, Float64


class InputSchema(BaseModel):
    x: Differentiable[Float64]
    r: Differentiable[Array[(3, 3), Float32]]
    s: float

class OutputSchema(BaseModel):
    a: Differentiable[Float64]
    b: int
```

Here, output `a` can be differentiated with respect to the scalar `x` and the matrix `r`, but not `s`.

```{warning}
`Differentiable` can only wrap {py:class}`tesseract_core.runtime.Array` types (including rank-0 aliases like {py:class}`Float64 <tesseract_core.runtime.Float64>`). Using it on Python base types (e.g., `Differentiable[float]`) will raise an error.
```

Beyond marking fields, you also need to implement the derivative logic. With autodiff frameworks like JAX or PyTorch, these are usually one-liners. See [Differentiable Programming](tr-autodiff) for details on implementing `jacobian`, `jacobian_vector_product`, and `vector_jacobian_product`.

[^1]: "A Tesseract's input schema" refers to the input of its `apply` function. The other endpoints (`jacobian`, `jacobian_vector_product`, ...) are derivatives of `apply`, and their schemas are derived automatically.
