# SpaceClaim Tesseract

## Context
Complex CAD models are often imported from parametric CAD software and require pre-processing by e.g. extracting a fluid volume for simulatuion, or naming domain faces such that appropriate boundary conditions can be applied.

SpaceClaim is commonly used to perform these pre-processing actions, and additionally can be used to generate geometry. In this example we demonstrate the use of SpaceClaim through SpaceClaim scripts (`.scscript`) within a Tesseract.

## What is different about this Tesseract?

Tesseracts are most commonly used in their self-contained built form; however SpaceClaim is not containerization friendly. Instead we use the Tesseract [`tesseract-core[runtime]`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/tesseract-runtime-cli.html to  [`serve`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/tesseract-runtime-cli.html#tesseract-runtime-serve) the SpaceClaim (or any other proprietary software that cannot be containerized). This will allow us to setup a Tesseract on an e.g. Windows machine with Ansys installed and easily expose its functionality over as an HTTP API.

## Setting up a generic Tesseract Runtime Server

When creating a Tesseract you should have a Tesseract directory with three files like so:

```bash
$ tree examples/helloworld
examples/helloworld
├── tesseract_api.py
├── tesseract_config.yaml
└── tesseract_requirements.txt
```

If this isn't familiar then you can learn about Tesseract basics [here](../../introduction/get-started.md). Normally this directory would be passed to `tesseract build`, but instead we can install and make use of the `tesseract-runtime` CLI application which will provide us an interface with the Tesseract:

```bash
pip install tesseract-core[runtime]
```

Now with an a open port of your choice, and from within the Tesseract directory, we can execute:

```bash
tesseract-runtime serve --port port_number
```

The result is a Tesseract Runtime Server.

```bash
$ tesseract-runtime serve --port 443
INFO:     Started server process [14888]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:443 (Press CTRL+C to quit)
```

## Example SpaceClaim Tesseract (`examples/spaceclaim`)

For this specific example we are looking at the SpaceX Grid Fin geometry shown in this [demo](https://si-tesseract.discourse.group/c/showcase/11). This specific example requires `trimesh` as a dependency. For an easy setup navigate to  `examples/spaceclaim` and install the requirements in your python environment of choice

```bash
pip install -r tesseract_requirements.txt
```

This Tesseract accepts goemetry parameters to create `N` Grid Fin geometries simulatanously. The API was setup this way to hide the startup latency of SpaceClaim when requesting a large number of geometries.

```{literalinclude} ../../../../examples/spaceclaim/tesseract_api.py
:language: python
:pyobject: InputSchema
```

The output of the Tesseract is a list of `TriangularMesh` objects representing the N Grid Fin meshes.

```{literalinclude} ../../../../examples/spaceclaim/tesseract_api.py
:language: python
:pyobject: OutputSchema
```
```{literalinclude} ../../../../examples/spaceclaim/tesseract_api.py
:language: python
:pyobject: TriangularMesh
```

The explanation and intuation behind the inputs is explained further in the [demo](https://si-tesseract.discourse.group/c/showcase/11).

Now that we understand the inputs and outputs of the Tesseract we can use it. From within the Tesseract directory setup the runtime server with a port of your choice:

```bash
tesseract-runtime serve --port 443
```

If we want to manually test the Tesseract it should now be possible to make a HTTP request for two Grid Fin geometries either from the same computer, as shown here, or a seperate one. __Make sure to change the URL IP and port to reflect your setup, along with the SpaceClaim.exe path__:

```bash
#Bash
curl -d '{
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

After about (~15 seconds) the mesh output is returned and displayed in text form in your terminal. The point coordinates and cells correspond to a Grid Fin like below (shown with randomised cross beam locations).

![Example Grid Fin geometry](../../../img/grid_fin_stl.png)

*Figure: Grid Fin geometry shown with randomised beam locations.*

The `apply` function that we are invoking with the above command builds each of the Grid Fin geometries and extracts the mesh data from the `trimesh` objects.

```{literalinclude} ../../../../examples/spaceclaim/tesseract_api.py
:language: python
:pyobject: apply
```

To build the geometries we first prepare the SpaceClaim `.scscript` by replacing placeholder values with the user inputs via string substituation. SpaceClaim is then run, outputting `.stl` meshes that are read with `trimesh`.

```{literalinclude} ../../../../examples/spaceclaim/tesseract_api.py
:language: python
:pyobject: build_geometries
```

The `.scscript` preperation is unique to this Grid Fin example, with the user input values being processed into dictionaries that are then used within the string substituation. For a different geometry one would have to create their own dictionaries with all the neccessary inputs required by their new `.scscript`.

```{literalinclude} ../../../../examples/spaceclaim/tesseract_api.py
:language: python
:pyobject: _prep_scscript
```

```{literalinclude} ../../../../examples/spaceclaim/tesseract_api.py
:language: python
:pyobject: _find_and_replace_keys_in_archive
```

```{literalinclude} ../../../../examples/spaceclaim/tesseract_api.py
:language: python
:pyobject: _safereplace
```

Once the `.scscript` is ready the final step is to run SpaceClaim. Here it is easy to see how this proecss could be extended to any software that cannot be containorized. For example Ansys Fluent could also be wrapped in a Runtime Tesseract, with the Adjoint solver used to produced gradient information allowing the Tesseract to be differentiable.

```{literalinclude} ../../../../examples/spaceclaim/tesseract_api.py
:language: python
:pyobject: run_spaceclaim
```

See this Runtime Tesseract in action in our Grid Fin optimisation [demo](https://si-tesseract.discourse.group/c/showcase/11).
