# Creating Tesseracts

In this page you will find instructions on how to create your own
tesseracts, starting from a very basic example and then building on
top of it with some advanced patterns. All this requires some basic
knowledge of tesseracts, so we suggest you to read at least the
[Quickstart page](tr-quickstart) beforehand.

## Initialize a new Tesseract

In order to start creating a Tesseract in your current directory,
you can run
```bash
$ tesseract init
```
and follow the prompt to specify a name for your tesseract.
(Alternatively, you can use the option `--name` to provide this inline.)
This will create three files in the current directory:
  -  `tesseract_api.py`, a python module where you should implement
     the core computations in the Tesseract.
  -  `tesseract_config.yaml`, a `yaml` file where you can specify metadata, such as Tesseract name and version, various build options, such as
     which base Docker image to use, define custom steps in building the Tesseract, access to external data, and so on.
  -  `tesseract_requirements.txt`, a text file where you can specify the (Python)
     dependencies of your Tesseract. It should be in the
     [requirements file format](https://pip.pypa.io/en/stable/reference/requirements-file-format/).

If you want to create these files in some other path, you can use the `--target-dir [DIRECTORY]`
option.
Other options include:
- `--recipe` allows you to use ready-made templates that generate pre-configured Tesseract configurations for common scenarios (such as generating autodiff endpoints from JAX functions).
- `--help` will print help regarding CLI usage and list all currently available recipes.

## Define a simple Tesseract

The `tesseract_api.py` produced by `tesseract init` contains some boilerplate code
which can guide you. Let's follow it section by section and pretend we want to implement
a very simple `helloworld` Tesseract: one that accepts a string `name` and returns `"Hello {name}!"`.

The first section in `tesseract_api.py` looks like this:

```python
class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    pass
```

This is where you can define the input and output schemas of the Tesseract[^1] via
[pydantic](https://docs.pydantic.dev/latest/). As we want `helloworld` to accept a string
and return one, we can edit this section as follows:

```python
class InputSchema(BaseModel):
    name: str = Field(
        description="Name of the person you want to greet."
    )

class OutputSchema(BaseModel):
    greeting: str = Field(description="A greeting!")
```

Providing field descriptions is not mandatory, but if you do,
they will be included in live docs and in the generated schemas
themselves. This is useful to end users, so in general we recommend to
write them. You can also set default values, validators (both for each field individually
and at the model level), and so on. Have a look at
[pydantic's docs](https://docs.pydantic.dev/latest/) to know more.

Just below the schemas you will find the section where required
endpoints are defined:

```python
def apply(inputs: InputSchema) -> OutputSchema:
    ...
```

These must always be present, with no exception, for every Tesseract. Right now,
only `apply` is required.

In the `apply` function instead we define the calculation we want our Tesseract
to implement. In `helloworld`'s case, this is simply:

```python
def apply(inputs: InputSchema) -> OutputSchema:
    """Greet a person whose name is given as input."""
    return OutputSchema(greeting=f"Hello {inputs.name}")
```

```{note}
The docstring of `apply` (as well as all others that you implement in
`tesseract_api.py`) will be available to your Tesseract's users.
```

The last section in `tesseract_api.py` contains templates for optional endpoints:

```python
# def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
#     return {}

...
```

You can leave it untouched for this example, as the operation we are
implementing is not differentiable.

Finally, we can set the name of this Tesseract and its version in
`tesseract_config.yaml`.

```yaml
name: "helloworld"
version: "1.0.0"
description: "A sample Python app"
```

If you followed all these steps, congratulations! 🎉 You are ready to
build your first Tesseract.

## Build a Tesseract

In order to build a Tesseract, you can use the `tesseract build` command.
For the `helloworld` Tesseract we defined above, assuming that our
current directory is where `tesseract_api.py` is located,
the full command would be:

```
$ tesseract build . -t v1.0.0
```

This is to be interpreted as "build the Tesseract which is located in
the current directory, and tag it as `v1.0.0`". The name of the Tesseract
is defined in the `tesseract_config.yaml` file, and it is `helloworld`, so the full
name of the tesseract we just built is `helloworld:v1.0.0`.

It is not mandatory to provide a tag for a Tesseract; in case you don't do it, it is tagged
as `latest`, and the Tesseract that was previously tagged as `latest` loses that tag.


### Viewing Built Tesseracts
In order to view all locally available Tesseracts, you can run the following command:

```bash
$ tesseract list
```

The output will be a table of Tesseracts images with their ID, name, version, and description:

```bash
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID                  ┃ Tags                  ┃ Name       ┃ Version ┃ Description                               ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ sha256:d4bdc2c29eb1 │ ['helloworld:latest'] │ helloworld │ 1.0.0   │ A sample Python app                       │
└─────────────────────┴───────────────────────┴────────────┴─────────┴───────────────────────────────────────────┘
```

## Arrays in the schema

In scientific computing, one of the most important data
structures are N-dimensional arrays. For these, you should use
the `tesseract_core.runtime.Array` type annotation:

```python
from tesseract_core.runtime import Array, Float64

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

The first parameter of `Array` is the `shape`, while the second is the `dtype`, for both of them you can use the same convention as numpy's `ndarray`.
Within a Tesseract, the variables marked as `Array` will be cast to `numpy.ndarray` objects of the given `dtype` and `shape`, so you can rely
on numpy's broadcasting rules and operators. In this example, `r @ x + s` would be a valid expression to use in `apply` and similar endpoints, which
corresponds to multiplying the `r` matrix with the `x` vector, and then adding the scalar `s` (broadcasted to match the vector's dimension) to that.

For scalar values you can use `tesseract_core.runtime.Float32`, `tesseract_core.runtime.Float64`, `tesseract_core.runtime.Int32`,
and so on (see [tesseract_core.runtime API documentation](../api/tesseract-runtime-api.md) for a comprehensive list).
You could use just `float`, but you would not be able to make use of the autodiff features which
we show in [the Differentiability section](tr-create-diff).

## Nested schemas

As both `InputSchema` and `OutputSchema` are pydantic `BaseModel`s,
they support nesting other `BaseModels` within them. This can be useful
to create data structures that are convenient to work with:

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

(tr-create-diff)=
## Differentiability

A key feature of Tesseracts is their ability to expose endpoints for calculating various kinds of derivatives when the operation they implement is differentiable, which in turn makes it possible to combine multiple Tesseracts into automatically differentiable workflows! This is advantageous in multiple contexts: shape optimization, model calibration, and so on.

Keeping with one of Tesseract's key foci being *validation*, the type annotation {py:class}`tesseract_core.runtime.Differentiable` is introduced to mark outputs that can be differentiated, and inputs that can be differentiated with respect to.
All outputs marked as `Differentiable` will be considered differentiable with respect to
all inputs marked as `Differentiable`.
Attempting to differentiate (with respect to) an output/input (e.g. by passing `jac_inputs=["non_differentiable_arg"]` to the `jacobian` endpoint) will raise a validation error even before the endpoint is invoked.

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

Here, it will be possible in principle to differentiate `a` in the Tesseract's output with respect to the scalar
parameter `x` and with respect to each of the components of the matrix `r` -- but not with respect to `s`.

```{warning}
`Differentiable` can only be used on {py:class}`tesseract_core.runtime.Array` types, which includes aliases for
rank 0 tensors like {py:class}`Float64 <tesseract_core.runtime.Float64>`. Do not use it on
Python base types -- things like `Differentiable[float]` will trigger errors.
```

Aside from marking the parameters with respect to which your Tesseract is differentiable, one also
must implement the logic for how the derivatives shall be calculated. If you are using an autodiff
framework like `jax` or `pytorch`, these implementations will mostly be one-liners, but you are free
in general to implement whatever method works best for you.
Check the [page on Autodiff](tr-autodiff) for more details on how to implement the differential
endpoints like `jacobian`, `jacobian_vector_product`, and so on.


[^1]: We often refer to "a Tesseract's input schema" to indicate the input of the Tesseract's
`apply` function. This should not lead to confusion, as there is only one core
functionality in a Tesseract, the one implemented in `apply`. All the other
functions in a Tesseract (`jacobian`, `jacobian_vector_product`, ...) are just derivatives of `apply`, and their schemas are adapted automatically
from the schema of `apply`.
