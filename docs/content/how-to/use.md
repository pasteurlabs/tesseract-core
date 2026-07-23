# Interacting with Tesseracts

## Viewing running Tesseracts

To list all running Tesseracts:

```bash
$ tesseract ps
```

The output is a table showing each container's ID, name, version, host port, project ID, and description:

```bash

┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID           ┃ Name      ┃ Version ┃ Host Port ┃ Project ID             ┃ Description                               ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 997fca92ea37 │ vectoradd │ 1.2.3   │ 56434     │ tesseract-afn60xa27hih │ Simple tesseract that adds two vectors.\n │
└──────────────┴───────────┴─────────┴───────────┴────────────────────────┴───────────────────────────────────────────┘
```

- **Host Port** — the port to use when calling the Tesseract's endpoints.
- **Project ID** — pass this to `tesseract teardown` to stop all containers in that project.

## Invoking a Tesseract

Every Tesseract's core operation is `apply` — this could be a neural network forward pass, a simulation step, or any other computation.

::::{tab-set}
:::{tab-item} CLI
:sync: cli

```bash
$ tesseract run vectoradd apply @examples/vectoradd/example_inputs.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":[5.0,7.0,9.0],"encoding":"json"}}}
```

Where `example_inputs.json` contains:

```{literalinclude} ../../../examples/vectoradd/example_inputs.json
:caption: example_inputs.json
```

The `@` prefix tells the CLI to read input from a file. For small payloads, you can also pass JSON inline:

```bash
$ tesseract run vectoradd apply '{"inputs": {"a": ..., "b": ...}}'
```

:::
:::{tab-item} REST API
:sync: http
Make sure the Tesseract is running (`docker ps`), or launch it with `tesseract serve vectoradd`.

Then call its `/apply` endpoint:

```bash
$ curl http://<tesseract-address>:<port>/apply \ # Replace with actual address
  -H "Content-Type: application/json" \
  -d @examples/vectoradd/example_inputs.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":[5.0,7.0,9.0],"encoding":"json"}}}
```

Where the payload posted to `/apply` is:

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
>>>     vectoradd.apply({"a": a, "b": b})
{'result': array([5., 7., 9.])}
```

The [Tesseract](#tesseract_core.Tesseract) context manager starts the container locally and tears it down when the context exits.

```{tip}
To connect to a remote Tesseract, use `Tesseract.from_url(...)`.
```

:::
::::

This Tesseract returns the vector sum `a + b` in the `result` output field.

## Optional endpoints and differentiation

If a Tesseract is differentiable, its derivative endpoints work the same way. For example, computing the Jacobian of `result` with respect to `a` at $a = (1,2,3)$, $b = (4,5,6)$:

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

The payload we posted contains information about which inputs and outputs we want to consider
when computing derivatives:

```{literalinclude} ../../../examples/vectoradd/example_jacobian_inputs.json
:caption: example_jacobian_inputs.json
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
>>>     vectoradd.jacobian({"a": a, "b": b}, jac_inputs=["a"], jac_outputs=["result"])
{'result': {'a': array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])}}
```

:::
::::

The output is a 3x3 identity matrix, as expected.

To check which endpoints a Tesseract supports, use the `/docs` endpoint of a running Tesseract, `tesseract apidoc`, or the [Python SDK](#tesseract_core.Tesseract.available_endpoints):

```python
>>> with Tesseract.from_image("vectoradd") as vectoradd:
...     print(vectoradd.available_endpoints)
['apply', 'jacobian', 'health']
```

## OpenAPI schemas for programmatic parsing

Each Tesseract has a unique input/output signature. To retrieve the schema programmatically:

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
>>>     schema = vectoradd.openapi_schema
```

:::
::::

Schemas are returned in [OpenAPI](https://swagger.io/specification/) format, designed for programmatic parsing. For a human-readable view, use the `/docs` endpoint of a running Tesseract or run `tesseract apidoc <tesseract-name>`.

The OpenAPI schema includes all endpoints, though they are all derived from the `/apply` input/output schemas.
