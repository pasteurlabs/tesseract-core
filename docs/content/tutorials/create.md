(tr-create)=

# Creating Tesseracts

This page walks through creating your own Tesseract: defining its schemas and endpoints, running and testing it, using arrays and nested data, and making it differentiable. We recommend reading the [Get Started page](tr-quickstart) first.

## Initialize a new Tesseract

Run the following to initialize everything needed to define a new Tesseract in the current directory:

% invisible-code-block: bash
%
% $ cd "$DOC_TMPDIR"

```bash
$ tesseract init --name my_tesseract
```

This creates three files:

- `tesseract_api.py` — Python module defining the Tesseract's computations.
- `tesseract_config.yaml` — metadata (name, version), how to install the Tesseract's requirements, and options for packaging it as a container image (base image, custom build steps, external data).
- `tesseract_requirements.txt` — Python dependencies in [pip requirements file format](https://pip.pypa.io/en/stable/reference/requirements-file-format/).

Use `--target-dir [DIRECTORY]` to create these files elsewhere. Other useful options:

- `--recipe` — use a ready-made template for common scenarios (e.g., generating gradient endpoints from JAX functions).
- `--help` — list all available options and recipes.

## Define a simple Tesseract

The generated `tesseract_api.py` contains boilerplate code to guide you. Let's walk through it section by section, implementing a simple `helloworld` Tesseract that accepts a `name` and returns `"Hello {name}!"`.

The first section defines the input and output schemas:

% invisible-code-block: python
%
% from pydantic import BaseModel

```python
class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    pass
```

Input and output schemas are defined as [Pydantic](https://docs.pydantic.dev/latest/) models[^1]. For `helloworld`, we need a string in and a string out:

```{literalinclude} ../../../examples/helloworld/tesseract_api.py
:language: python
:start-at: from pydantic
:end-before: def apply
```

Field descriptions are optional but recommended — they appear in the auto-generated docs and schemas. You can also use default values, validators, and other Pydantic features (see [Pydantic docs](https://docs.pydantic.dev/latest/)).

Below the schemas, you'll find the required endpoints:

```python
def apply(inputs: InputSchema) -> OutputSchema:
    ...
```

Currently, only `apply` is required. This is where you define the Tesseract's core computation. For `helloworld`:

```{literalinclude} ../../../examples/helloworld/tesseract_api.py
:language: python
:pyobject: apply
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

```{literalinclude} ../../../examples/helloworld/tesseract_config.yaml
:language: yaml
:end-before: build_config
```

(tr-create-run)=

## Run your Tesseract

% invisible-code-block: bash
%
% $ cp -r "$REPO_ROOT"/examples/helloworld/. .

To call your Tesseract while you work on it, load it with `Tesseract.from_source`. From the directory containing `tesseract_api.py`:

```python
>>> from tesseract_core import Tesseract
>>>
>>> with Tesseract.from_source("tesseract_api.py") as helloworld:
...     helloworld.apply({"name": "Ozzy"})
{'greeting': 'Hello Ozzy!'}
```

On first use, `from_source` creates a virtual environment in `.tesseract-venv` next to `tesseract_api.py` and installs the Tesseract runtime and the requirements from `tesseract_requirements.txt` into it. This needs [uv](https://docs.astral.sh/uv/getting-started/installation/). Later calls reuse the environment until the requirements change.

The Tesseract runs in a separate process that starts when the `with` block is entered and stops when it exits, so changes to `tesseract_api.py` take effect the next time you enter the block. To call the Tesseract from the command line or over HTTP instead, see [Interacting with Tesseracts](tr-interact).

(testing-a-tesseract)=

## Test your Tesseract

Every Tesseract ships with a built-in `test` endpoint for regression testing. It calls one of your endpoints with a given input and checks the result against a known-good output (or against an expected exception), reporting a simple pass/fail.

A test case is a small JSON file. For the `helloworld` example, `test_cases/test_apply.json` looks like this:

```{literalinclude} ../../../examples/helloworld/test_cases/test_apply.json
:language: json
```

From Python, pass the test case to the `test` method, which raises an `AssertionError` if the test fails:

```python
>>> import json
>>>
>>> with open("test_cases/test_apply.json") as f:
...     test_case = json.load(f)
>>>
>>> with Tesseract.from_source("tesseract_api.py") as helloworld:
...     helloworld.test(test_case)
```

From the command line, use `tesseract-runtime test` and pass the file with the `@` prefix. The environment that `from_source` created includes `tesseract-runtime`, which loads `tesseract_api.py` from the current directory unless `TESSERACT_API_PATH` says otherwise:

```bash
$ .tesseract-venv/bin/tesseract-runtime test @test_cases/test_apply.json
{"status":"passed","message":"","endpoint":"apply"}
```

The command prints a JSON result with a `status` of `"passed"`, `"failed"`, or `"error"`. It always exits with code 0 and reports the outcome in the payload, so a test runner must read `status` rather than the exit code.

A spec has a few more knobs:

- **`atol` / `rtol`** — absolute and relative tolerances for numeric comparisons (defaults: `1e-8` and `1e-5`). Loosen these for endpoints whose outputs aren't bit-reproducible.
- **`expected_exception`** (with optional **`expected_exception_regex`**) — assert that the endpoint _raises_ instead of returning. Give the exception's name (e.g. `"pydantic.ValidationError"`) and, optionally, a regex the message must match. Provide exactly one of `expected_outputs` or `expected_exception`.

To run tests without starting a separate process, for example from pytest, `Tesseract.from_tesseract_api("tesseract_api.py")` imports the Tesseract into the current Python process and has the same `test` method. The Tesseract's requirements must then be installed in that process's environment.

```{note}
Over HTTP, the `test` endpoint is only exposed when a Tesseract is [served in debug mode](../how-to/debugging.md). `Tesseract.from_source` and `Tesseract.from_tesseract_api` enable debug mode, and the `tesseract-runtime test` and `tesseract run ... test` commands always have the endpoint. A plain `tesseract-runtime serve` or `tesseract serve` does not expose it.
```

Keeping a `test_cases/` directory next to each Tesseract and running every spec in CI is the pattern the [`cookiecutter-tesseract`](#building-a-multi-tesseract-project) template sets up for you.

## Arrays in the schema

N-dimensional arrays are central to scientific computing. Use the `tesseract_core.runtime.Array` type annotation to define them:

% invisible-code-block: python
%
% from pydantic import Field

```python
from tesseract_core.runtime import Array, Float32, Float64

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

For scalars, use `tesseract_core.runtime.Float32`, `Float64`, `Int32`, etc. (see the [runtime API reference](../reference/tesseract-runtime-api.md) for the full list). Plain `float` works but does not support the [differentiability features](tr-create-diff) described below.

## Nested schemas

Since `InputSchema` and `OutputSchema` are Pydantic `BaseModel`s, they support nesting other models within them:

```python
from tesseract_core.runtime import Int32

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
from tesseract_core.runtime import Differentiable

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

## Package it as an image

To share a Tesseract with people who don't have your environment, to deploy it, or to give it system dependencies that a Python environment can't provide, package it as a container image. This needs [Docker](#installation-docker). Run `tesseract build` from the directory containing `tesseract_api.py`:

```bash
$ tesseract build .
```

The image name comes from `tesseract_config.yaml`. By default, the image is tagged with both the specified version (`1.0.0`) and `latest`, so you can reference it as `helloworld:1.0.0` or `helloworld:latest`. Options that only apply to images, such as the base image, system packages, and data files, go in the `build_config` section of `tesseract_config.yaml` (see the [configuration reference](../reference/config.md)).

To list all locally available Tesseract images:

```bash
$ tesseract list
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ ID                  ┃ Tags                                      ┃ Name       ┃ Version ┃ Description         ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ sha256:d4bdc2c29eb1 │ ['helloworld:1.0.0', 'helloworld:latest'] │ helloworld │ 1.0.0   │ A sample Python app │
└─────────────────────┴───────────────────────────────────────────┴────────────┴─────────┴─────────────────────┘
```

The output is a table of Tesseract images with their ID, tags, name, version, and description. For machine-readable output unaffected by rich formatting use `tesseract list --format json`.

To run the same test cases against the image, use `tesseract run`:

```bash
$ tesseract run helloworld:latest test @test_cases/test_apply.json
{"status":"passed","message":"","endpoint":"apply"}
```

See [Interacting with Tesseracts](tr-interact) for the other ways to call an image, and [Deploying Tesseracts](../how-to/deploy.md) for sharing and running images on other machines.

[^1]: "A Tesseract's input schema" refers to the input of its `apply` function. The other endpoints (`jacobian`, `jacobian_vector_product`, ...) are derivatives of `apply`, and their schemas are derived automatically.
