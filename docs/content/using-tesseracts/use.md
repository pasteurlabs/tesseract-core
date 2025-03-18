# Interacting with Tesseracts

## Viewing Running Tesseracts
In order to view all Tesseracts that are running, you can run the following command:

```bash
$ tesseract ps
```

The output will be a table of Tesseracts containers with their ID, name, version, host port, project id, and description:

```bash

┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID           ┃ Name      ┃ Version ┃ Host Port ┃ Project ID             ┃ Description                               ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 997fca92ea37 │ vectoradd │ 1.2.3   │ 56434     │ tesseract-afn60xa27hih │ Simple tesseract that adds two vectors.\n │
└──────────────┴───────────┴─────────┴───────────┴────────────────────────┴───────────────────────────────────────────┘
```

The `host port` is the port used in the Tesseract address to interact with the different endpoints of the hosted Tesseract.
The `project id` is what is used in the `tesseract teardown` command and will kill all Tesseracts associated with that project id.

## Invoking a Tesseract


The main operation which a Tesseract implements is called `apply`. This could be
a forward pass of a neural network applied to some input data,
a simulation of a quantity of interest which accepts its initial conditions as input,
and so on.

::::{tab-set}
:::{tab-item} CLI
:sync: cli
```bash
$ tesseract run vectoradd apply @examples/vectoradd/example_inputs.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":[5.0,7.0,9.0],"encoding":"json"}}}
```

Where the `example_inputs.json` passed as input just contains the following:
```{literalinclude} ../../../examples/vectoradd/example_inputs.json
:caption: example_inputs.json
```

Notice the `@` before the filename in the command payload. This tells the CLI to read the file and
use it as input to the Tesseract. You can also provide a JSON string in place of this:
```bash
$ tesseract run vectoradd apply '{"inputs": {"a": ..., "b": ...}}'
```
This can be useful for small input payloads, but it can become quite cumbersome very quickly.

:::
:::{tab-item} REST API
:sync: http
Make sure that the Tesseract is running as a service (`docker ps`). Otherwise, launch
it via `tesseract serve vectoradd`.

You can then simply curl its `/apply` endpoint
```bash
$ curl http://<tesseract-address>:<port>/apply \ # Replace with actual address
  -H "Content-Type: application/json" \
  -d @examples/vectoradd/example_inputs.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":[5.0,7.0,9.0],"encoding":"json"}}}
```

Where the payload `example_inputs.json` we POST to the `/apply` endpoint is the following:
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

The [Tesseract](#tesseract_core.Tesseract) context manager will spin up
a Tesseract locally, and tear it down once the context is exited.

```{tip} You can also instantiate a Tesseract object which connects to
a remote Tesseract via `Tesseract(url=...)`.```
:::
::::

This Tesseract, which accepts two vectors, is returning an object which has the vector sum
`a + b` in the `result` output field.

If the Tesseract you are using is differentiable, the endpoints which return derivatives can be called in a similar
fashion; for instance, here is how one would calculate the Jacobian of the `result` output field, partial only
on the `a` vector, at $a = (1,2,3)$, $b = (4,5,6)$:

::::{tab-set}
:::{tab-item} CLI
:sync: cli
```bash
$ tesseract run vectoradd jacobian @examples/vectoradd/example_jacobian_inputs.json
{"result":{"a":{"object_type":"array","shape":[3,3],"dtype":"float64","data":{"buffer":[[3.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,3.0]],"encoding":"json"}}}}
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
Notice that the payload we posted contains information about which inputs and outputs we want to consider
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

Now the output is a 3x3 matrix, as expected.

To know which endpoints among `jacobian`, `jacobian_vector_product`, or `vector_jacobian_product` are
available in a given Tesseract, you can check which commands appear in `tesseract run <tesseract-name>[:tag] --help`, or
look at the docs at the `/docs` endpoint of a running Tesseract service (or use `tesseract apidoc`).

## Input and output schemas

As they wrap arbitrary computation, each Tesseract has a unique input/output signature.
To make it easier to programmatically know each specific Tesseract's input and output schema,
you can use the following:

::::{tab-set}
:::{tab-item} CLI
:sync: cli
For the input schema:

```bash
$ tesseract run vectoradd input-schema
```

For the output schema:

```bash
$ tesseract run vectoradd output-schema
```
:::
:::{tab-item} REST API
:sync: http
For the input schema:

```bash
$ curl <tesseract-address>:<port>/input_schema
```

For the output schema:

```bash
$ curl <tesseract-address>:<port>/output_schema
```
:::
:::{tab-item} Python
:sync: python
For the input schema:

```python
>>> from tesseract_core import Tesseract
>>> with Tesseract(image="vectoradd") as vectoradd:
>>>     vectoradd.input_schema
```

For the output schema:

```python
>>> from tesseract_core import Tesseract
>>> with Tesseract(image="vectoradd") as vectoradd:
>>>     vectoradd.output_schema
```
:::
::::

Schemas are returned in the [JSON Schema](https://json-schema.org/) format,
which is intended for programmatic parsing rather than human
readability. If you want a more human-readable version of the schema,
your best option is instead to look at the docs in the `/docs` endpoint of a Tesseract
service.

The schemas returned by the command above are the ones of the `/apply` endpoint. The schema
for differential endpoints, like `jacobian`, are derived from it -- see the
[page on Autodiff](tr-autodiff) for more details.

You can also get the OpenAPI schema for the whole API of each Tesseract via:

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
>>> with Tesseract(image="vectoradd") as vectoradd:
>>>     vectoradd.openapi_schema
```

If you only care about which endpoints are available, you can use the [available_endpoints](#tesseract_core.Tesseract.available_endpoints)
attribute:

```python
>>> with Tesseract(image="vectoradd") as vectoradd:
>>>     vectoradd.available_endpoints
['apply', 'jacobian', 'health', 'input_schema', 'output_schema', 'abstract_eval']
```
:::
::::
