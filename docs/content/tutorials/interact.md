(tr-interact)=

# Interacting with Tesseracts

You've built a Tesseract — now let's put it to work. This page walks through
calling a Tesseract from all three interfaces (CLI, REST API, and Python SDK),
computing derivatives, and reading a Tesseract's schema so you know exactly what
it expects. We recommend working through [Get Started](tr-quickstart) and
[Creating Tesseracts](tr-create) first.

We'll use the `vectoradd` Tesseract from the [examples](https://github.com/pasteurlabs/tesseract-core/tree/main/examples/vectoradd),
which computes `s·a + b` for two vectors `a`, `b` and a scalar `s`. Build it the
same way you built `helloworld`:

```bash
$ tesseract build examples/vectoradd
```

## See what's running

Some ways of calling a Tesseract (the REST API, or a served Python session)
need a running container. To list every Tesseract currently running on your
machine:

```bash
$ tesseract ps
```

The output is a table showing each container's ID, name, version, host port,
project ID, and description:

```bash
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID           ┃ Name      ┃ Version ┃ Host Port ┃ Project ID             ┃ Description                               ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 997fca92ea37 │ vectoradd │ 1.2.3   │ 56434     │ tesseract-afn60xa27hih │ Simple tesseract that adds two vectors.\n │
└──────────────┴───────────┴─────────┴───────────┴────────────────────────┴───────────────────────────────────────────┘
```

Two columns you'll reach for often:

- **Host Port** — the port to use when calling the Tesseract's REST endpoints.
- **Project ID** — pass this to `tesseract teardown` to stop all containers in that project.

## Invoke the `apply` endpoint

Every Tesseract's core operation is `apply` — for `vectoradd` it's the vector
sum, but in general this could be a neural network forward pass, a simulation
step, or any other computation. Let's call it with `a = (1, 2, 3)` and
`b = (4, 5, 6)`, which should give us `(5, 7, 9)`.

Pick whichever interface fits how you're working — all three return the same
result:

::::{tab-set}
:::{tab-item} CLI
:sync: cli

```bash
$ tesseract run vectoradd apply @examples/vectoradd/example_inputs.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":[5.0,7.0,9.0],"encoding":"json"}}}
```

The `@` prefix tells the CLI to read the input payload from a file:

```{literalinclude} ../../../examples/vectoradd/example_inputs.json
:caption: example_inputs.json
```

For small payloads, you can also pass JSON inline:

```bash
$ tesseract run vectoradd apply '{"inputs": {"a": ..., "b": ...}}'
```

:::
:::{tab-item} REST API
:sync: http

First make sure the Tesseract is running (check `tesseract ps`), or launch it with:

```bash
$ tesseract serve vectoradd
```

Then post to its `/apply` endpoint, using the host port from `tesseract ps`:

```bash
$ curl http://<tesseract-address>:<port>/apply \ # Replace with actual address
  -H "Content-Type: application/json" \
  -d @examples/vectoradd/example_inputs.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":[5.0,7.0,9.0],"encoding":"json"}}}
```

The payload posted to `/apply` is:

```{literalinclude} ../../../examples/vectoradd/example_inputs.json
:caption: example_inputs.json
```

:::
:::{tab-item} Python
:sync: python

```python
>>> import numpy as np
>>> from tesseract_core import Tesseract
>>>
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = np.array([4.0, 5.0, 6.0])
>>>
>>> with Tesseract.from_image(image="vectoradd") as vectoradd:
...     vectoradd.apply({"a": a, "b": b})
{'result': array([5., 7., 9.])}
```

The [Tesseract](#tesseract_core.Tesseract) context manager starts the container
locally and tears it down when the context exits, so you don't have to manage
its lifecycle by hand.

```{tip}
To connect to a Tesseract that's already running elsewhere, use `Tesseract.from_url(...)` instead.
```

:::
::::

Notice the shape of the result. Over the CLI and REST API, arrays come back in a
structured JSON form that records the `dtype`, `shape`, and `encoding` alongside
the raw `buffer` — this is what makes Tesseract outputs self-describing across
languages. The Python SDK unpacks that form into a NumPy array for you.

## Compute derivatives

`vectoradd` is _differentiable_: it was built with `Differentiable` inputs and
outputs, so it exposes derivative endpoints in addition to `apply`. (If you
haven't yet met differentiable Tesseracts, the
[Differentiability](tr-create-diff) section of the previous page introduces
them.)

Let's compute the Jacobian of `result` with respect to `a` at the same point,
`a = (1, 2, 3)`, `b = (4, 5, 6)`. Since `result = s·a + b` with `s = 1`, we
expect the 3×3 identity matrix:

::::{tab-set}
:::{tab-item} CLI
:sync: cli

```bash
$ tesseract run vectoradd jacobian @examples/vectoradd/example_jacobian_inputs.json
{"result":{"a":{"object_type":"array","shape":[3,3],"dtype":"float64","data":{"buffer":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"encoding":"json"}}}}
```

:::
:::{tab-item} REST API
:sync: http

```bash
$ curl -d @examples/vectoradd/example_jacobian_inputs.json \
  -H "Content-Type: application/json" \
  http://<tesseract-address>:<port>/jacobian
{"result":{"a":{"object_type":"array","shape":[3,3],"dtype":"float64","data":{"buffer":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"encoding":"json"}}}}
```

:::
:::{tab-item} Python
:sync: python

```python
>>> import numpy as np
>>> from tesseract_core import Tesseract
>>>
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = np.array([4.0, 5.0, 6.0])
>>>
>>> with Tesseract.from_image("vectoradd") as vectoradd:
...     vectoradd.jacobian({"a": a, "b": b}, jac_inputs=["a"], jac_outputs=["result"])
{'result': {'a': array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])}}
```

:::
::::

A derivative endpoint takes the same inputs as `apply`, plus two extra fields
that say _what_ to differentiate: `jac_inputs` (which inputs to differentiate
with respect to) and `jac_outputs` (which outputs to differentiate). For the CLI
and REST API, those live in the payload:

```{literalinclude} ../../../examples/vectoradd/example_jacobian_inputs.json
:caption: example_jacobian_inputs.json
```

Only fields marked `Differentiable` may appear in `jac_inputs` / `jac_outputs`;
asking for a non-differentiable field raises a validation error before the
endpoint runs. Alongside `jacobian`, Tesseracts can expose the
`jacobian_vector_product` (JVP) and `vector_jacobian_product` (VJP) endpoints —
these are what [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) and
[Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch) use under the
hood to make gradients flow through your Tesseract automatically. See
[Differentiable Programming](tr-autodiff) for what each one computes.

Not every Tesseract implements every endpoint. To check which ones a Tesseract
supports:

```python
>>> with Tesseract.from_image("vectoradd") as vectoradd:
...     print(vectoradd.available_endpoints)
['apply', 'jacobian', 'health']
```

`vectoradd` implements `jacobian` but not JVP or VJP, so those don't appear.

Equivalently, run `tesseract apidoc vectoradd` or open the `/docs` endpoint of a
running Tesseract in your browser.

## Read a Tesseract's schema

Because each Tesseract has its own input/output signature, you'll often want to
discover that signature programmatically — for validation, for building a UI, or
just to remind yourself what a Tesseract expects. Every Tesseract can emit its
schema in [OpenAPI](https://swagger.io/specification/) format:

::::{tab-set}
:::{tab-item} CLI
:sync: cli

```bash
$ tesseract run vectoradd openapi-schema
```

:::
:::{tab-item} REST API
:sync: http

```bash
$ curl <tesseract-address>:<port>/openapi.json
```

:::
:::{tab-item} Python
:sync: python

```python
>>> from tesseract_core import Tesseract
>>> with Tesseract.from_image("vectoradd") as vectoradd:
...     schema = vectoradd.openapi_schema
```

:::
::::

The OpenAPI schema covers every endpoint, though they're all derived from the
`apply` input/output schemas. It's built for machines; for a human-readable
view, run `tesseract apidoc vectoradd` or open a running Tesseract's `/docs`
endpoint.

## What's next

You can now build a Tesseract and drive it through all three interfaces,
including its derivatives. From here:

- [](../how-to/pipelines.md) — chain several Tesseracts into a larger differentiable workflow.
- [](../concepts/differentiable-programming.md) — how the `jacobian`, JVP, and VJP endpoints relate, and how to implement them.
- [](../demo/demo.md) — full end-to-end demos: data assimilation, shape optimization, and more.
