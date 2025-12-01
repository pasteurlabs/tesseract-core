# Parametric Shape Optimization of Rocket Fins with Ansys SpaceClaim and PyAnsys

This directory contains an example Tesseract configuration and scripts demonstrating how to use Tesseract-JAX with Ansys SpaceClaim and PyMAPDL. The overall workflow is illustrated below and demonstrated within our [rocket fin optimization showcase](https://si-tesseract.discourse.group/t/parametric-shape-optimization-of-rocket-fins-with-ansys-spaceclaim-and-pyansys/109):

![Workflow](imgs/workflow_1.png)

The main entry point of this demo is `optimization.ipynb`. The evolution of the mesh over the optimization on two different initial conditions can be seen here:

| Grid IC | Random IC |
|----------|----------|
| ![Workflow](imgs/mesh_grid_adam.gif)    | ![Workflow](imgs/mesh_rnd_adam.gif)     |

where the loss decay is plotted here:

![Workflow](imgs/loss.png)

In `optimization_os.ipynb` we provide a alternative workflow that is relying on the open source FEM solver [JAX-FEM](https://github.com/deepmodeling/jax-fem), hence does not access to an ANSYS licence. It can be run directly without having to the following installation steps.



| Mesh | Loss |
|----------|----------|
| ![Workflow](imgs/mesh_optim.gif)    | ![Workflow](imgs/loss_os.png)     |


## Get Started

### Prerequisites

A Windows machine A with:
1. Ansys SpaceClaim + MAPDL installed with an active license.
2. Python with a virtual environment (e.g. via conda, venv).
3. Two open ports.
4. Known IP address, obtain it by running

```powershell
(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi","Ethernet" | Where-Object {$_.IPAddress -notlike "169.254.*" -and $_.IPAddress -ne $null}).IPAddress
```

A machine B, ideally running linux, with:
1. Docker installed and running.
2. Python with a virtual environment (e.g. via conda, venv).

### SpaceClaim Tesseract setup

In Windows Powershell, install the the required dependencies by running:

```bash
$ pip install tesseract-core[runtime] trimesh
```

Clone this repository, navigate to `demo/_showcase/ansys-shapeopt/spaceclaim` and start the Tesseract runtime server with:

```bash
$ tesseract-runtime serve --port <port_number_1> --host 0.0.0.0
```
Note that we do not build a Tesseract Docker image for SpaceClaim in this example. Instead, we use an existing SpaceClaim installation directly from the host machine. More details about this Tesseract can be found [here](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/examples/ansys_integration/spaceclaim_tess.html).

### PyMAPDL Server

On machine A, run the following Powershell command to start Ansys with gRPC server enabled:

```powershell
Start-Process -FilePath "F:\ANSYS Inc\v242\ansys\bin\winx64\ANSYS242.exe" -ArgumentList "-grpc", "-port", "<port_number_2>"
```

replace "v242" with your Ansys version and ensure the path is correct. More details about PyMAPDL Tesseract can be found [here](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/examples/ansys_integration/pymapdl_tess.html).

### Building Tesseracts

On machine B, navigate to `demo/_showcase/ansys-shapeopt/` and run

```bash
$ pip install -r requirements.txt
```

and build the needed tesseracts using

```bash
$ tesseract build spaceclaim
$ tesseract build pymapdl
$ tesseract build sdf_fd
```

### Run the notebook

At this stage you are ready to run the main notebook [optimization.ipynb](optimization.ipynb).
