(tr-quickstart)=
# Get started

```{note}
Please ensure you have a working [installation](installation.md) before proceeding with this tutorial.
```

## Hello Tesseract

The [`tesseract` command line application](../api/tesseract-cli.md) provides tools to build Tesseracts as Docker containers from ``tesseract_api.py`` files. Here, we'll use it to build and invoke a simple Tesseract that greets you by name.

### Building your first Tesseract

Download the {download}`Tesseract examples </downloads/examples.zip>` and run the following command from where you unpacked the archive:

```bash
$ tesseract build examples/helloworld
 [i] Building image ...
 [i] Built image sha256:95e0b89e9634, ['helloworld:latest']
```

```{tip}
Having issues building Tesseracts? Check out [common issues](#installation-issues) with the installation process.
```

Congratulations! You've just built your first Tesseract, which is now available as a Docker image on your system.

### Running your Tesseract

You can interact with any built Tesseract via the command line interface (CLI), the REST API, or through the [Python API](../api/tesseract-api.md) (which uses CLI / REST under the hood). Try the commands below to see your Tesseract in action:

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
 [i] View Tesseract: http://localhost:56489/docs
 [i] Docker Compose Project ID, use it with 'tesseract teardown' command: tesseract-u7um375qt6dj5
{"project_id": "tesseract-u7um375qt6dj5", "containers": [{"name": "tesseract-uum375qt6dj5-sha256-9by9ahsnsza2-1", "port": "8080"}]}%

$ # The port at which your Tesseract will be served is random if `--port` is not specified;
$ # specify the one you received from `tesseract serve` output in the next command.
$ curl -d '{"inputs": {"name": "Osborne"}}' \
       -H "Content-Type: application/json" \
       http://localhost:8080/apply
{"greeting":"Hello Osborne!"}

$ tesseract teardown tesseract-9hj8fyxrx073
 [i] Tesseracts are shutdown for Project name: tesseract-9hj8fyxrx073
```
:::
:::{tab-item} Python API
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

Now, have a look at the (auto-generated) CLI and REST API docs for your Tesseract:

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
 [i] Serving OpenAPI docs for Tesseract helloworld at http://localhost:59569/docs
 [i]   Press Ctrl+C to stop
```
:::
::::

```{figure} /img/apidoc-screenshot.png
:scale: 33%

The OpenAPI docs for the `vectoradd` Tesseract, documenting its endpoints and valid inputs / outputs.
```

## Let's peek under the hood

The folder passed to `tesseract build` contains the files needed to build and run the Tesseract:

```bash
$ tree examples/helloworld
examples/helloworld
├── tesseract_api.py
├── tesseract_config.yaml
└── tesseract_requirements.txt
```

These files are all that's needed to define a Tesseract.

### `tesseract_api.py`

 The `tesseract_api.py` file defines the Tesseract's input and output schemas, and the functions that are being called when we invoke `tesseract run <funcname>`. These are,
 `apply`, `jacobian`, `jacobian_vector_product`, and `vector_jacobian_product`. Out of all of the endpoints you
 can implement, only `apply` is required for a Tesseract to work.

```{literalinclude} ../../../examples/helloworld/tesseract_api.py
:pyobject: InputSchema
```

```{literalinclude} ../../../examples/helloworld/tesseract_api.py
:pyobject: OutputSchema
```

```{literalinclude} ../../../examples/helloworld/tesseract_api.py
:pyobject: apply
```

(quickstart-tr-config)=
### `tesseract_config.yaml`

`tesseract_config.yaml` contains the Tesseract's metadata, such as its name, description, version, and build configuration.

```{literalinclude} ../../../examples/helloworld/tesseract_config.yaml
```

### `tesseract_requirements.txt`

`tesseract_requirements.txt` lists the Python packages required to build and run the Tesseract.

```{note}
The `tesseract_requirements.txt` file is optional. In fact, `tesseract_api.py` is free to invoke functions that are not written in Python at all. In this case, use the `build_config` section in [`tesseract_config.yaml`](quickstart-tr-config) to provide data files and install the necessary dependencies.
```

```{literalinclude} ../../../examples/helloworld/tesseract_requirements.txt
```


## The journey continues...

Now, you're ready to learn more, depending on your needs:

- [](../creating-tesseracts/create.md)
- [](../using-tesseracts/use.md)
