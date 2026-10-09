(tr-quickstart)=

# Get Started

## Quick install

```{note}
This requires Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/). Docker is only needed for the [last part](#get-started-build), which builds a container image. See the [installation guide](../introduction/installation.md) for details.
```

```bash
$ pip install tesseract-core
```

## Hello Tesseract

A Tesseract is a folder with a `tesseract_api.py` that defines its endpoints. Here, we'll run and invoke a simple Tesseract that greets you by name.

### Run your first Tesseract

Download the {download}`Tesseract examples </downloads/examples.zip>` and unpack the archive. From that directory, you can invoke the `helloworld` example via the [Python SDK](../reference/tesseract-api.md), the CLI, or the REST API:

::::{tab-set}
:::{tab-item} Python SDK
:sync: python

```python
>>> from tesseract_core import Tesseract
>>>
>>> with Tesseract.from_source("examples/helloworld/tesseract_api.py") as helloworld:
...     helloworld.apply({"name": "Osborne"})
{'greeting': 'Hello Osborne!'}
```

`Tesseract.from_source` serves the Tesseract from a separate process. On first use, it builds an environment for the Tesseract from its `tesseract_config.yaml` and `tesseract_requirements.txt`, and reuses that environment while the requirements stay unchanged.

:::
:::{tab-item} CLI
:sync: cli

The `tesseract-runtime` CLI runs a Tesseract in your current environment, so the runtime and the Tesseract's requirements have to be installed there:

```bash
$ pip install "tesseract-core[runtime]"
$ export TESSERACT_API_PATH=examples/helloworld/tesseract_api.py
$ tesseract-runtime apply '{"inputs": {"name": "Osborne"}}'
{"greeting":"Hello Osborne!"}
```

:::
:::{tab-item} REST API
:sync: http

With the runtime installed as in the CLI tab, serve the Tesseract and send it requests:

```bash
$ export TESSERACT_API_PATH=examples/helloworld/tesseract_api.py
$ tesseract-runtime serve --port 8080 &
$ curl -d '{"inputs": {"name": "Osborne"}}' \
       -H "Content-Type: application/json" \
       http://127.0.0.1:8080/apply
{"greeting":"Hello Osborne!"}
```

:::
::::

```{tip}
Having trouble? Check [common issues](#installation-issues) for solutions.
```

(get-started-build)=

### Build a container image

To share a Tesseract with someone who doesn't have your environment, or to deploy it, build it into a container image. This step requires [Docker](#installation-docker):

```bash
$ tesseract build examples/helloworld
 [i] Building image ...
 [i] Built image sha256:95e0b89e9634, ['helloworld:latest']
```

The image exposes the same endpoints, and the [`tesseract` CLI](../reference/tesseract-cli.md) can run or serve it:

::::{tab-set}
:::{tab-item} CLI
:sync: cli

```bash
$ tesseract run helloworld apply '{"inputs": {"name": "Osborne"}}'
{"greeting":"Hello Osborne!"}
```

:::
:::{tab-item} REST API
:sync: http

```bash
$ tesseract serve -p 8080 helloworld
 [i] Waiting for Tesseract containers to start ...
 [i] Container ID: 2587deea2a2efb6198913f757772560d9c64cf8621a6d1a54aa3333a7b4bcf62
 [i] Name: tesseract-uum375qt6dj5-sha256-9by9ahsnsza2-1
 [i] Entrypoint: ['tesseract-runtime', 'serve']
 [i] View Tesseract: http://127.0.0.1:56489/docs
 [i] Docker Compose Project ID, use it with 'tesseract teardown' command: tesseract-u7um375qt6dj5
{"project_id": "tesseract-u7um375qt6dj5", "containers": [{"name": "tesseract-uum375qt6dj5-sha256-9by9ahsnsza2-1", "port": "8080"}]}%

$ curl -d '{"inputs": {"name": "Osborne"}}' \
       -H "Content-Type: application/json" \
       http://127.0.0.1:8080/apply
{"greeting":"Hello Osborne!"}

$ tesseract teardown tesseract-u7um375qt6dj5
 [i] Tesseracts are shutdown for Project name: tesseract-u7um375qt6dj5
```

:::
:::{tab-item} Python SDK
:sync: python

```python
>>> from tesseract_core import Tesseract
>>>
>>> with Tesseract.from_image("helloworld") as helloworld:
...     helloworld.apply({"name": "Osborne"})
{'greeting': 'Hello Osborne!'}
```

:::
::::

Each built Tesseract auto-generates CLI and REST API docs. To view them:

::::{tab-set}
:::{tab-item} CLI
:sync: cli

```bash
$ tesseract run helloworld --help
```

:::
:::{tab-item} REST API
:sync: http

```bash
$ tesseract apidoc helloworld
 [i] Waiting for Tesseract containers to start ...
 [i] Serving OpenAPI docs for Tesseract helloworld at http://127.0.0.1:59569/docs
 [i]   Press Ctrl+C to stop
```

:::
::::

```{figure} /img/apidoc-screenshot.png
:scale: 33%

The OpenAPI docs for the `helloworld` Tesseract, documenting its endpoints and valid inputs / outputs.
```

(getting-started)=

## Under the hood

The `helloworld` folder contains three files:

```bash
$ tree examples/helloworld
examples/helloworld
├── tesseract_api.py
├── tesseract_config.yaml
└── tesseract_requirements.txt
```

These are all that's needed to define a Tesseract.

### `tesseract_api.py`

This file defines the Tesseract's input and output schemas, along with the endpoint functions: `apply`, `abstract_eval`, `jacobian`, `jacobian_vector_product`, and `vector_jacobian_product` (see [endpoints](../reference/endpoints.md)). Only `apply` is required.

```{literalinclude} ../../../examples/helloworld/tesseract_api.py
:pyobject: InputSchema
```

```{literalinclude} ../../../examples/helloworld/tesseract_api.py
:pyobject: OutputSchema
```

```{literalinclude} ../../../examples/helloworld/tesseract_api.py
:pyobject: apply
```

```{tip}
For a Tesseract that has all optional endpoints implemented, check out the [Univariate example](../examples/building-blocks/univariate.md).
```

(quickstart-tr-config)=

### `tesseract_config.yaml`

Contains metadata such as the Tesseract's name, description, version, and build configuration.

```{literalinclude} ../../../examples/helloworld/tesseract_config.yaml

```

### `tesseract_requirements.txt`

Lists the Python packages needed to run the Tesseract, whether in the environment `from_source` builds or in a container image, in [pip requirements file format](https://pip.pypa.io/en/stable/reference/requirements-file-format/).

```{note}
This file is optional. `tesseract_api.py` can invoke functions written in any language. In that case, use the `build_config` section in [`tesseract_config.yaml`](quickstart-tr-config) to provide data files and install dependencies.
```

```{literalinclude} ../../../examples/helloworld/tesseract_requirements.txt

```

## Features and limitations

::::{tab-set}
:::{tab-item} Features

- **Self-documenting** — Tesseracts announce their interfaces, so users can inspect them without reading source code and perform static validation without running the code.
- **Auto-validating** — Input data is automatically validated against the schema, so internal logic can assume the data is in the expected format.
- **Autodiff-native** — Tesseracts support [differentiable programming](../concepts/differentiable-programming.md) and integrate as native operations in PyTorch and JAX — but exposing derivatives is _strictly optional_.
- **Batteries included** — Every Tesseract comes with a CLI, a REST API, and a Python SDK, and runs as a subprocess, a container, or a remote service.

:::
:::{tab-item} Limitations

- **Python as glue** — Tesseracts may use any software under the hood, but they always use Python as glue between the runtime and the wrapped functionality. Support for Python projects is more mature than other languages.
- **Single entrypoint** — Each Tesseract has a single `apply` entrypoint. To expose N functions, create N Tesseracts.
- **Context-free** — Tesseracts are not aware of outer-loop orchestration or runtime details.
- **Runtime overhead** — Calls usually cross a process boundary, which costs milliseconds, so Tesseracts suit components whose calls take much longer than that (see [performance](../concepts/performance.md)).

:::
::::

## Next steps

Depending on your needs:

- [](../tutorials/create.md) — define schemas, implement endpoints, and build Tesseracts
- [](../tutorials/interact.md) — invoke Tesseracts, compute derivatives, and read their schemas

- [](../how-to/check-gradients.md) — verify a Tesseract's derivatives against finite differences

Or jump into the [demos](../demo/demo.md), which differentiate through Fortran, PyTorch, and coupled solvers end to end.
