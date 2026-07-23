(deploying-on-hpc)=

# Deploying on HPC with Apptainer

On HPC systems — Slurm clusters, national labs, university systems — a root Docker
daemon is usually unavailable, so Docker cannot be used. Tesseract's **Apptainer
backend** (experimental) brings Tesseracts to these systems using
[Apptainer](https://apptainer.org/) (formerly Singularity), the de-facto rootless
container runtime there.

This guide covers the recommended end-to-end workflow: build in CI, pull on the
cluster, and run or serve inside a Slurm job.

```{note}
The Apptainer backend is experimental. Please share cluster-specific findings on
the [community forum](https://si-tesseract.discourse.group/).
```

## Overview

Apptainer cannot build images from Dockerfiles, so building stays on Docker. The
supported flow is:

1. **Build** the Tesseract with Docker (on a workstation or in CI).
2. **Publish** it — push the OCI image to a registry, _or_ convert it to a single
   SIF file.
3. **Pull / run** natively on the cluster with the Apptainer backend.

Select the backend on the cluster with:

```bash
$ export TESSERACT_CONTAINER_BACKEND=apptainer
```

See the {ref}`backend-capability-matrix` for exactly what the Apptainer backend
supports.

## 1. Build and publish (CI recommended)

The recommended path is a CI pipeline (e.g. GitHub Actions) that builds the image
with Docker and pushes it to a registry your cluster can reach:

```bash
# On a machine with Docker
$ tesseract build my_tesseract --tag 1.0.0
$ docker tag my_tesseract:1.0.0 ghcr.io/my-org/my_tesseract:1.0.0
$ docker push ghcr.io/my-org/my_tesseract:1.0.0
```

Alternatively, on a machine that has _both_ Docker and Apptainer, build directly to
a SIF in the image store:

```bash
$ TESSERACT_CONTAINER_BACKEND=apptainer tesseract build my_tesseract --output-format sif
["my_tesseract:1.0.0", "my_tesseract:latest"]
```

## 2. Pull onto the cluster

On the cluster (login node), pull the published image into the local SIF store:

```bash
$ export TESSERACT_CONTAINER_BACKEND=apptainer
$ tesseract pull docker://ghcr.io/my-org/my_tesseract:1.0.0
["my_tesseract:1.0.0"]
$ tesseract list
```

### Where the SIF store lives

By default images are stored under `~/.local/share/tesseract/images/`. `$HOME` is
often small on clusters, so point the store at scratch or project storage instead:

```bash
$ export TESSERACT_APPTAINER_IMAGE_DIR=/scratch/$USER/tesseract-images
```

Store mutations are file-locked, so parallel jobs pulling into a shared store on a
parallel filesystem are safe.

## 3. Run inside a Slurm job

One-shot execution works exactly like the Docker backend; only the backend selector
differs. Inside a batch script:

```bash
#!/bin/bash
#SBATCH --job-name=tesseract-run
#SBATCH --gpus=1
#SBATCH --time=00:30:00

export TESSERACT_CONTAINER_BACKEND=apptainer
export TESSERACT_APPTAINER_IMAGE_DIR=/scratch/$USER/tesseract-images

tesseract run my_tesseract:1.0.0 apply @inputs.json --output-path ./results
```

You can also pass a direct path to a `.sif` file instead of a store reference:

```bash
$ tesseract run /scratch/$USER/tesseract-images/my_tesseract/1.0.0.sif apply @inputs.json
```

## Serving inside a job

`tesseract serve` starts an Apptainer _instance_ and binds a free host port
directly (Apptainer is host-network only, so there is no port mapping):

```bash
$ tesseract serve my_tesseract:1.0.0
# ... Serving Tesseract at http://127.0.0.1:52664
```

Attach from the same node with the Python SDK:

```python
from tesseract_core import Tesseract

with Tesseract.from_image("my_tesseract:1.0.0") as t:
    result = t.apply({"inputs": ...})
```

```{important}
Apptainer instances are **per-user and per-node**. `tesseract ps` on a login node
will not see instances running on compute nodes — run `ps`, `teardown`, and any SDK
attach on the same node where the Tesseract is served (e.g. from within the same
Slurm allocation).
```

To serve several Tesseracts that call each other, give each its own localhost port
and connect via URLs — user-defined Docker networks and network aliases are not
available under Apptainer.

## GPUs

The Apptainer backend enables GPU access with `--nv` and selects devices via
`CUDA_VISIBLE_DEVICES`, which Slurm typically sets for you when you request GPUs:

```bash
$ tesseract run my_tesseract:1.0.0 apply @inputs.json  # inside a --gpus Slurm job
```

Docker-style per-device requests (`--gpus '"device=0,1"'`) do not exist under
Apptainer; use `CUDA_VISIBLE_DEVICES` instead.

## Unsupported options

These fail with a clear error rather than degrading silently:

- **User-defined networks / aliases** (`--network`, `--network-alias`) — serve on
  distinct localhost ports instead.
- **Port mapping** — the runtime binds host ports directly.
- **Restart policies** — not available.

**Memory limits** (`--memory`) are applied best-effort and require cgroups v2
delegation to take effect.
