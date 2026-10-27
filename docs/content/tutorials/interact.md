(tr-interact)=

# Interacting with Tesseracts

This page shows how to call a Tesseract through all three interfaces (CLI, REST
API, and Python SDK), compute its derivatives, and read its schema so you know
exactly what it expects. We recommend working through [Get started](tr-quickstart)
and [Creating Tesseracts](tr-create) first.

We'll use the `vectoradd` Tesseract from the {gh-tree}`examples <examples/vectoradd>`,
which computes `s·a + b` for two vectors `a`, `b` and a scalar `s`. As in
[Get started](tr-quickstart), commands on this page run from the directory that
contains `examples/`.

## Invoke the `apply` endpoint

Every Tesseract's core operation is `apply`. For `vectoradd` it's the vector
sum, but in general this could be a neural network forward pass, a simulation
step, or any other computation. Let's call it with `a = (1, 2, 3)` and
`b = (4, 5, 6)`, which should give us `(5, 7, 9)`.

Pick whichever interface fits how you're working. All three return the same
result:

::::{tab-set}
:::{tab-item} CLI
:sync: cli

The `tesseract-runtime` CLI runs a Tesseract in your current environment, so
the runtime (`pip install "tesseract-core[runtime]"`) and the Tesseract's
requirements have to be installed there. Point it at the Tesseract with
`TESSERACT_API_PATH`:

```bash
$ export TESSERACT_API_PATH=examples/vectoradd/tesseract_api.py
$ tesseract-runtime apply @examples/vectoradd/example_inputs.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":[5.0,7.0,9.0],"encoding":"json"}}}
```

The `@` prefix tells the CLI to read the input payload from a file:

```{literalinclude} ../../../examples/vectoradd/example_inputs.json
:caption: example_inputs.json
```

For small payloads, you can also pass JSON inline:

% skip: next "abbreviated payload"

```bash
$ tesseract-runtime apply '{"inputs": {"a": ..., "b": ...}}'
```

:::
:::{tab-item} REST API
:sync: http

With the runtime installed as in the CLI tab, start a server for the Tesseract:

% invisible-code-block: bash
%
% $ tesseract-runtime serve --port 8080 > /dev/null 2>&1 & echo $! > "$DOC_TMPDIR/serve.pid"
% $ for i in $(seq 60); do curl -sf http://127.0.0.1:8080/health > /dev/null && break; sleep 0.5; done

% skip: next "long-running server, started in the background above"

```bash
$ export TESSERACT_API_PATH=examples/vectoradd/tesseract_api.py
$ tesseract-runtime serve --port 8080
```

Then, from a second terminal, post to its `/apply` endpoint:

```bash
$ curl http://127.0.0.1:8080/apply \
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
>>> with Tesseract.from_source("examples/vectoradd/tesseract_api.py") as vectoradd:
...     vectoradd.apply({"a": a, "b": b})
{'result': array([5., 7., 9.])}
```

`Tesseract.from_source` starts the Tesseract in a separate process when the
`with` block is entered and stops it when the block exits. On first use, it
also creates an environment for the Tesseract (see
[Creating Tesseracts](tr-create-run)).

```{tip}
To connect to a Tesseract that's already running elsewhere, use `Tesseract.from_url(...)` instead.
```

:::
::::

Notice the shape of the result. Over the CLI and REST API, arrays come back in a
structured JSON form that records the `dtype`, `shape`, and `encoding` alongside
the raw `buffer`, which makes Tesseract outputs self-describing across
languages. The Python SDK unpacks that form into a NumPy array for you.

## Compute derivatives

`vectoradd` is _differentiable_: its schemas mark inputs and outputs as
`Differentiable`, so it exposes derivative endpoints in addition to `apply`.
(If you haven't yet met differentiable Tesseracts, the
[Differentiability](tr-create-diff) section of the previous page introduces
them.)

Let's compute the Jacobian of `result` with respect to `a` at the same point,
`a = (1, 2, 3)`, `b = (4, 5, 6)`. Since `result = s·a + b` with `s = 1`, we
expect the 3×3 identity matrix:

::::{tab-set}
:::{tab-item} CLI
:sync: cli

```bash
$ tesseract-runtime jacobian @examples/vectoradd/example_jacobian_inputs.json
{"result":{"a":{"object_type":"array","shape":[3,3],"dtype":"float64","data":{"buffer":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"encoding":"json"}}}}
```

:::
:::{tab-item} REST API
:sync: http

```bash
$ curl http://127.0.0.1:8080/jacobian \
  -H "Content-Type: application/json" \
  -d @examples/vectoradd/example_jacobian_inputs.json
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
>>> with Tesseract.from_source("examples/vectoradd/tesseract_api.py") as vectoradd:
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
`jacobian_vector_product` (JVP) and `vector_jacobian_product` (VJP) endpoints.
These are what [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) and
[Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch) use under the
hood to make gradients flow through your Tesseract automatically. See
[Differentiable programming basics](tr-autodiff) for what each one computes.

Not every Tesseract implements every endpoint. To check which ones a Tesseract
supports:

```python
>>> with Tesseract.from_source("examples/vectoradd/tesseract_api.py") as vectoradd:
...     print(vectoradd.available_endpoints)
['apply', 'jacobian', 'health', 'test']
```

`vectoradd` implements `jacobian` but not JVP or VJP, so those don't appear.
(`test` is listed because `from_source` serves Tesseracts in debug mode, which
enables the [`test` endpoint](testing-a-tesseract).)

A served Tesseract also has human-readable API docs at its `/docs` endpoint.
With the server from the REST API tab running, open
`http://127.0.0.1:8080/docs` in your browser.

## Read a Tesseract's schema

Because each Tesseract has its own input/output signature, you'll often want to
discover that signature programmatically: for validation, for building a UI, or
just to remind yourself what a Tesseract expects. Every Tesseract can emit its
schema in [OpenAPI](https://swagger.io/specification/) format:

::::{tab-set}
:::{tab-item} CLI
:sync: cli

```bash
$ tesseract-runtime openapi-schema
```

:::
:::{tab-item} REST API
:sync: http

```bash
$ curl http://127.0.0.1:8080/openapi.json
```

:::
:::{tab-item} Python
:sync: python

```python
>>> from tesseract_core import Tesseract
>>> with Tesseract.from_source("examples/vectoradd/tesseract_api.py") as vectoradd:
...     schema = vectoradd.openapi_schema
```

:::
::::

The OpenAPI schema covers every endpoint, though they're all derived from the
`apply` input/output schemas. It's built for machines; for a human-readable
view, open a served Tesseract's `/docs` endpoint.

% invisible-code-block: bash
%
% $ kill "$(cat "$DOC_TMPDIR/serve.pid")"

## Call a Tesseract image

Tesseracts are often shared as container images, which bundle the Tesseract with
its environment and any system dependencies. With [Docker](#installation-docker)
installed, build an image from the Tesseract's folder:

```bash
$ tesseract build examples/vectoradd
```

The image is named after the Tesseract, and the same three interfaces work
against it:

::::{tab-set}
:::{tab-item} CLI
:sync: cli

```bash
$ tesseract run vectoradd apply @examples/vectoradd/example_inputs.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":[5.0,7.0,9.0],"encoding":"json"}}}
```

:::
:::{tab-item} REST API
:sync: http

`tesseract serve` starts a container in the background:

```bash
$ tesseract serve vectoradd
```

To list every Tesseract container running on your machine:

```bash
$ tesseract ps
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID           ┃ Name      ┃ Version ┃ Host Address    ┃ Container Name         ┃ Description                             ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 997fca92ea37 │ vectoradd │ 1.2.3   │ 127.0.0.1:56434 │ tesseract-afn60xa27hih │ Simple tesseract that adds two vectors. │
└──────────────┴───────────┴─────────┴─────────────────┴────────────────────────┴─────────────────────────────────────────┘
```

Use the **Host Address** to call the container's endpoints:

```bash
$ curl http://<tesseract-address>:<port>/apply \
  -H "Content-Type: application/json" \
  -d @examples/vectoradd/example_inputs.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":[5.0,7.0,9.0],"encoding":"json"}}}
```

Pass the **Container Name** to `tesseract teardown` to stop the container.

:::
:::{tab-item} Python
:sync: python

```python
>>> with Tesseract.from_image("vectoradd") as vectoradd:
...     vectoradd.apply({"a": a, "b": b})
{'result': array([5., 7., 9.])}
```

:::
::::

To view an image's API docs in your browser, run `tesseract apidoc vectoradd`.
[Deploying Tesseracts](../how-to/deploy.md) covers pushing images to a registry
and running them on other machines.

## What's next

You can now call a Tesseract through all three interfaces, from source or as an
image, and compute its derivatives. From here:

- [](../how-to/pipelines.md) — chain several Tesseracts into a larger differentiable workflow.
- [](../concepts/differentiable-programming.md) — how the `jacobian`, JVP, and VJP endpoints relate, and how to implement them.
- [](../demo/demo.md) — full end-to-end demos: data assimilation, shape optimization, and more.
