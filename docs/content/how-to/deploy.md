# Deploying Tesseracts

Tesseracts built via `tesseract build` are standard Docker images, so they can be shared, pushed, pulled, and deployed like any other container.

## Using Docker tools

Built Tesseracts are Docker images named after the Tesseract. Standard Docker commands work directly:

```bash
# Invoke a Tesseract via `docker run`
$ docker run vectoradd apply --help
Usage: tesseract-runtime apply [OPTIONS] JSON_PAYLOAD

  Apply the Tesseract to the input data.

  Multiplies a vector `a` by `s`, and sums the result to `b`.
  ...
```

```bash
# Push a Tesseract to a container registry
$ docker push vectoradd
...
```

```bash
# Pull a Tesseract from a container registry
$ docker pull mytesseract
...
```

```bash
# Save a pre-built Tesseract image to a tar file
$ docker image save vectoradd -o vectoradd.tar
...
```

```bash
# Spawn a Tesseract server
$ docker run vectoradd serve
...
```

This gives you fine-grained control over Tesseract images and lets you use any container-aware tooling.

```{tip}
Since `podman` has a `docker`-compatible CLI, all commands above work with `podman` as well.
```

## Example: Deploying on [Azure Virtual Machines](https://azure.microsoft.com/en-us/products/virtual-machines)

```{note}
This example assumes you have an Azure account and are familiar with cloud infrastructure. Using Azure VMs is just one of many deployment options. Cloud resources may incur costs.
```

The general process:

1. Push the Tesseract image to Azure Container Registry.
2. Create a virtual machine.
3. Install Docker on the VM.
4. (Optional) Install NVIDIA drivers and the CUDA toolkit.
5. Pull the Tesseract image on the VM.
6. Start the container via `docker run serve`, listening on port `8000`.

A reference script is available here: {download}`create-vm-azure.sh </downloads/create-vm-azure.sh>`.

```{warning}
This script contains placeholders and will not work out of the box. It assumes you have a resource group, VNet, Subnet, and Azure Container Registry already configured.
```

Populate the variables at the top of the script and run it:

```console
$ bash create-vm-azure.sh vectoradd:latest
```

The script assumes Docker is installed locally and authenticated to your Azure Container Registry:

```console
$ az login
$ az acr login --name <registry-name>
```

## Deploying without a container engine

If no container engine is available, you can deploy using `tesseract-runtime serve`. This provides the same functionality as `tesseract serve` and `docker run myimage serve`, and can be queried the same way (e.g., with `curl` or via [`Tesseract.from_url`](#Tesseract.from_url) in the Python SDK).

This approach requires a Python environment with the dependencies from `tesseract_requirements.txt` installed, as documented in [Running Tesseracts without containers](running-without-containers). For a complete HPC example, see the [HPC deployment tutorial](https://si-tesseract.discourse.group/t/deploying-and-interacting-with-tesseracts-on-hpc-clusters-using-tesseract-runtime-serve/104).

Example:

```bash
$ pip install -r tesseract_requirements.txt
$ TESSERACT_API_PATH=/path/to/tesseract_api.py tesseract-runtime serve
```
