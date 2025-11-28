# Wrapping SpaceClaim as a Tesseract

This guide outlines how to wrap Ansys SpaceClaim as a Tesseract. For this, we will use [non-containerized execution](#tr-without-docker) to start an HTTP server that dispatches requests to SpaceClaim through SpaceClaim scripts (`.scscript`).

```{seealso}
The full code for this Tesseract can be found under `demo/_showcase/ansys-shapeopt/spaceclaim` in the [Tesseract Core repository](https://github.com/pasteurlabs/tesseract-core/tree/main/demo/_showcase/ansys-shapeopt/spaceclaim).
```

## Why SpaceClaim as a Tesseract?

Complex CAD models imported from parametric CAD software often require pre-processing before they can be fed into a simulator, such as extracting a fluid volume or naming domain faces such that appropriate boundary conditions can be applied.

SpaceClaim is commonly used to generate parametric geometries and perform pre-processing actions on them. In this example we demonstrate the use of SpaceClaim as a geometry engine within Tesseract-driven processing pipelines. This unlocks powerful applications operating on real-world CAD geometries.

```{figure} ../../../img/spaceclaim_tesseract_workflow.png

Architecture of the SpaceClaim Tesseract we are going to implement.
```

## Core concepts

### Folder structure

When creating a new Tesseract you should have a directory with three files like so:

```bash
$ tesseract init --name spaceclaim --target-dir ./spaceclaim
$ tree ./spaceclaim
spaceclaim
├── tesseract_api.py
├── tesseract_config.yaml
└── tesseract_requirements.txt
```

```{seealso}
If this doesn't look familiar you can learn more about Tesseract basics [here](../../introduction/get-started.md).
```

To wrap SpaceClaim as a Tesseract, we will have to implement each of these files:

- `tesseract_api.py` contains all logic on accepted inputs / outputs and dispatches calls to SpaceClaim.
- `tesseract_config.yaml` is not relevant here, since we will be invoking the Tesseract without containerization (see below).
- `tesseract_requirements.txt` contains additional Python requirements to run the Tesseract.

### Non-containerized usage via `tesseract-runtime`

```{note}
**Tesseract without containerization** --- Tesseracts are most commonly used in the form of Docker containers. This is not a neccessary requirement, and any object that adheres to the Tesseract interface is a valid Tesseract (see also [](#tr-without-docker)).
```

SpaceClaim is typically running directly on a Windows host machine, so instead of using Docker to build an image and spin up a container, we leverage the [Tesseract runtime CLI](../../api/tesseract-runtime-cli) to serve the SpaceClaim Tesseract on bare metal and expose SpaceClaim's functionality over HTTP.

So instead of using the more common `tesseract build`, we install and use the `tesseract-runtime` CLI application which will provide us an interface with the Tesseract:

```bash
$ pip install tesseract-core[runtime]
```

```{warning}
Windows is officially only supported via Windows Subsystem for Linux (WSL), see [](#windows-support).
Make sure to use an appropriate WSL setup when running into issues.
```

Now with an a open port of your choice, and from within the Tesseract directory, we can execute:

```bash
$ pip install ./tesseract-requirements.txt
$ tesseract-runtime serve --host 0.0.0.0 --port port_number
```

The result is a Tesseract Runtime Server.

```bash
$ tesseract-runtime serve --host 0.0.0.0 --port 443
INFO:     Started server process [14888]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:443 (Press CTRL+C to quit)
```

## Implementing the Tesseract

### `tesseract_requirements.txt`

Besides a SpaceClaim installation and a recent Python version, this Tesseract only requires one dependency, the Python package `trimesh`. We add it to `tesseract_requirements.txt`:

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/spaceclaim/tesseract_requirements.txt
:language: text
```

Since we are running the Tesseract in a non-containerized way, all requirements need to be installed into the current Python interpreter:

```bash
$ pip install -r tesseract_requirements.txt
```

### `tesseract_api.py` --- Input and output schema

When using SpaceClaim as a geometry engine, the goal is typically to map design parameters in the parametric CAD model to a surface mesh. Here, we are creating a SpaceClaim Tesseract that operates on a grid fin geometry with a number of parameters representing the position of bars and their thickness.

```{seealso}
The explanation and intuation behind the inputs is explained further in the [demo](https://si-tesseract.discourse.group/c/showcase/11).
```

#### Input schema

This Tesseract accepts multiple goemetry parameters to create `N` grid fin geometries simulatanously. That way, we hide the startup latency of SpaceClaim when requesting a large number of geometries.

The `InputSchema` class looks like this:

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/spaceclaim/tesseract_api.py
:language: python
:pyobject: InputSchema
```

#### Output schema

The output of the Tesseract is a list of `TriangularMesh` objects representing the `N` grid fin meshes:

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/spaceclaim/tesseract_api.py
:language: python
:pyobject: OutputSchema
```

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/spaceclaim/tesseract_api.py
:language: python
:pyobject: TriangularMesh
```

## `tesseract_api.py` --- `apply`

The `apply` function that we are invoking with the above command builds each of the grid fin geometries and extracts the mesh data from the `trimesh` objects.

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/spaceclaim/tesseract_api.py
:language: python
:pyobject: apply
```

To build the geometries we first prepare the SpaceClaim `.scscript` by replacing placeholder values with the user inputs via string substitution. SpaceClaim is then run, outputting `.stl` meshes that are read with `trimesh`.

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/spaceclaim/tesseract_api.py
:language: python
:pyobject: build_geometries
```

The `.scscript` preperation is unique to this grid fin example, with the user input values being processed into dictionaries that are then used within the string substitution. For a different geometry one would have to create their own `.scscript` and dictionaries with all the neccessary inputs required.

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/spaceclaim/tesseract_api.py
:language: python
:pyobject: _prep_scscript
```

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/spaceclaim/tesseract_api.py
:language: python
:pyobject: _find_and_replace_keys_in_archive
```

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/spaceclaim/tesseract_api.py
:language: python
:pyobject: _safereplace
```

Once the `.scscript` is ready the final step is to run SpaceClaim. Here it is easy to see how this process could be extended to any software that is running on the host machine. For example Ansys Fluent could also be wrapped in a runtime Tesseract, potentially using an adjoint solver to produce gradient information, allowing the Tesseract to be differentiable.

```{literalinclude} ../../../../demo/_showcase/ansys-shapeopt/spaceclaim/tesseract_api.py
:language: python
:pyobject: run_spaceclaim
```

## Invoking the Tesseract

Now that we have defined the Tesseract we can use it. From within the Tesseract's root directory, start the runtime server with a port of your choice:

```bash
$ tesseract-runtime serve --host 0.0.0.0 --port 443
```

We can now test the Tesseract manually by sending an HTTP request for two grid fin geometries either from the same computer, as shown here, or via the network. __Make sure to change the URL IP and port to reflect your setup, along with the SpaceClaim.exe path__:

```bash
# Bash
$ curl -d '{
  "inputs": {
    "differentiable_parameters": [
    [200, 600, 0, 3.14, 0.39, 3.53, 0.79, 3.93, 1.18, 4.32, 1.57, 4.71, 1.96, 5.11, 2.36, 5.50, 2.75, 5.89],
    [400, 400, 0, 3.14, 0.39, 3.53, 0.79, 3.93, 1.18, 4.32, 1.57, 4.71, 1.96, 5.11, 2.36, 5.50, 2.75, 5.89]
    ],
    "non_differentiable_parameters": [
      [800, 100],
      [800, 100]
    ],
    "string_parameters": [
      "F:\\Ansys installations\\ANSYS Inc\\v241\\scdm\\SpaceClaim.exe",
      "geometry_generation.scscript"
    ]
  }
}' \
-H "Content-Type: application/json" \
http://127.0.0.1:443/apply
```

Or:

```powershell
# Windows PowerShell
curl -Method POST `
     -Uri "http://127.0.0.1:443/apply" `
     -ContentType "application/json" `
     -Body '{"inputs":{"differentiable_parameters":[[200,600,0,3.14,0.39,3.53,0.79,3.93,1.18,4.32,1.57,4.71,1.96,5.11,2.36,5.50,2.75,5.89],[400,400,0,3.14,0.39,3.53,0.79,3.93,1.18,4.32,1.57,4.71,1.96,5.11,2.36,5.50,2.75,5.89]],"non_differentiable_parameters":[[800,100],[800,100]],"string_parameters":["F:\\Ansys installations\\ANSYS Inc\\v241\\scdm\\SpaceClaim.exe","geometry_generation.scscript"]}}'
```

After about ~15 seconds the mesh output is returned and displayed in text form in your terminal. The point coordinates and cells correspond to a grid fin like below (shown with randomised cross beam locations).

```{figure} ../../../img/grid_fin_stl.png

Grid fin geometry shown with randomised beam locations.
```

## Next steps

Invoking SpaceClaim via HTTP is only the start of the Tesseract journey.

For example, by using finite difference approximations under the hood, we can make the resulting geometry [differentiable](../../introduction/differentiable-programming.md) with respect to the design parameters. For a concrete demonstration of end-to-end shape optimization in action, please have a look at our [grid fin optimization showcase](https://si-tesseract.discourse.group/c/showcase/11).
