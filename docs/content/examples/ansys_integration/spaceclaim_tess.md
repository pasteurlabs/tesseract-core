# SpaceClaim Tesseract

## Context
Within the Ansys product collection there are many great ways to generate geometries for Computer Aided Engineering (CAE) simulations. Additionally, complex CAD models are often imported from parametric CAD software and require pre-processing by e.g. extracting a fluid volume for simulatuion, or naming domain faces such that appropriate boundary conditions can be applied.

SpaceClaim is commonly used to perform these pre-procsesing actions, and additionally can be used to generate geometry. Here the use of SpaceClaim and automated SpaceClaim scripts (`.scscript`) will be demonstrated from within a Tesseract.

## What is different about this Tesseract?

Tesseracts are most commonly used in their self-contained built form; however, in this case we want SpaceClaim to be called from within the Tesseract, and cannot containerize SpaceClaim itself. To allow us to use SpaceClaim (or any other propriatary software that cannot be containorized) we will be demonstrating a runtime Tesseract with the `serve` functionality from `tesseract-core[runtime]`. This will allow us to setup a Tesseract on a machine with Ansys products and licensing, and then make requests to this Tesseract via HTTP.

## Setting up a Tesseract Runtime Server

When creating a Tesseract you should have a Tesseract directory with three files like so:

```bash
$ tree examples/helloworld
examples/helloworld
├── tesseract_api.py
├── tesseract_config.yaml
└── tesseract_requirements.txt
```

You can learn about Tesseract basics [here](../../introduction/get-started.md). Normally this directory would be passed to `tesseract build`, but instead we are going to make use of `tesseract-runtime` which will provide us an interface with the Tesseract:

```bash
pip install tesseract-core[runtime]
```

With this installed, and a open port of your choice, from within the Tesseract directory we can execute:

```bash
tesseract-runtime serve --port <port_number> --host 0.0.0.0
```

The result should be an active Tesseract Runtime Server that we can now make requests too:

```bash
$ tesseract-runtime serve --port 443 --host 0.0.0.0
INFO:     Started server process [14888]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:443 (Press CTRL+C to quit)
```

## Example Tesseract (`examples/ansys_integration/spaceclaim_tess`)

For this example we are looking at the SpaceX Grid Fin geometry demonstrated in this [demo](https://si-tesseract.discourse.group/c/showcase/11). This specific example requires some dependencies, so if you would like to follow along copy the files from `examples/ansys_integration/spaceclaim_tess` and create a new python environment, then:

```bash
pip install tesseract-core[runtime] trimesh
```



```{literalinclude} ../../../../examples/ansys_integration/spaceclaim_tess/tesseract_api.py
:language: python
:pyobject: run_spaceclaim
```

Why spaceclaim

Wrapping products that cannot be containorized

Runtime tesseracts with serve

Connecting to a runtime tesseract

How we choose inputs and outputs

See how this example is used in a DEMO
