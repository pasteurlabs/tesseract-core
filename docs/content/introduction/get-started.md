(tr-quickstart)=

# Get Started

```{note}
Make sure you have a working [installation](installation.md) before proceeding.
```

## Hello Tesseract

The [`tesseract` CLI](../api/tesseract-cli.md) builds Tesseracts as Docker containers from `tesseract_api.py` files. Here, we'll build and invoke a simple Tesseract that greets you by name.

### Build your first Tesseract

Download the {download}`Tesseract examples </downloads/examples.zip>` and run the following command from where you unpacked the archive:

```bash
$ tesseract build examples/helloworld
 [i] Building image ...
 [i] Built image sha256:95e0b89e9634, ['helloworld:latest']
```

```{tip}
Having trouble? Check [common issues](#installation-issues) for solutions.
```

Your first Tesseract is now available as a Docker image on your system.

### Run your Tesseract

You can interact with any built Tesseract via the CLI, the REST API, or the [Python SDK](../api/tesseract-api.md):

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

$ # The port at which your Tesseract will be served is random if `--port` is not specified;
$ # specify the one you received from `tesseract serve` output in the next command.
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
>>>     helloworld.apply({"name": "Osborne"})
{'greeting': 'Hello Osborne!'}
```

:::
::::

```{tip}
For faster iteration during development, you can run Tesseracts without building containers. See [Debugging and Development](../misc/debugging.md) for details.
```

Each Tesseract auto-generates CLI and REST API docs. To view them:

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

The folder passed to `tesseract build` contains three files:

```bash
$ tree examples/helloworld
examples/helloworld
├── tesseract_api.py
├── tesseract_config.yaml
└── tesseract_requirements.txt
```

These are all that's needed to define a Tesseract.

### `tesseract_api.py`

This file defines the Tesseract's input and output schemas, along with the endpoint functions: `apply`, `abstract_eval`, `jacobian`, `jacobian_vector_product`, and `vector_jacobian_product` (see [endpoints](../api/endpoints.md)). Only `apply` is required.

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

Lists the Python packages needed to build and run the Tesseract, in [pip requirements file format](https://pip.pypa.io/en/stable/reference/requirements-file-format/).

```{note}
This file is optional. `tesseract_api.py` can invoke functions written in any language. In that case, use the `build_config` section in [`tesseract_config.yaml`](quickstart-tr-config) to provide data files and install dependencies.
```

```{literalinclude} ../../../examples/helloworld/tesseract_requirements.txt

```

## Next steps

Depending on your needs:

- [](../creating-tesseracts/create.md) — define schemas, implement endpoints, and build Tesseracts
- [](../using-tesseracts/use.md) — invoke Tesseracts and work with their outputs

Or jump into end-to-end tutorials:

- [JAX Rosenbrock function minimization](https://si-tesseract.discourse.group/t/jax-based-rosenbrock-function-minimization/48)
- [PyTorch Rosenbrock function minimization](https://si-tesseract.discourse.group/t/pytorch-based-rosenbrock-function-minimization/44)
- [JAX RBF fitting with autodiff](https://si-tesseract.discourse.group/t/jax-auto-diff-templates-gaussian-radial-basis-function-fitting/51)
