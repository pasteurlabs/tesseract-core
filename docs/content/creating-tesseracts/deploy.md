# Deploying Tesseracts

Since Tesseracts built via `tesseract build` are regular Docker images, they can be shared, pushed / pulled, and deployed like any other container.

## Using Docker tools to work with Tesseracts

Built Tesseracts are Docker images of the same name as the Tesseract. You can use any Docker command to work with them. For example:

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

This provides fine-grained control over the Tesseract image, and allows you to use any container-aware tooling to manage and deploy them.

## Example: Deploying a Tesseract on [Azure Virtual Machines](https://azure.microsoft.com/en-us/products/virtual-machines)

```{note}
This example assumes you already have an Azure account and know your way around cloud infrastructure. Using Azure Virtual Machines is just one of many ways to deploy Tesseracts. Accessing cloud resources may incur costs.
```

The general process to deploy a Tesseract on an Azure Virtual Machine is as follows:
1. Push the Tesseract image to Azure Container Registry.
2. Instantiate a new virtual machine.
3. Setup Docker on the VM.
4. Optionally setup Nvidia drivers and CUDA toolkit.
5. Pull the Tesseract Image in the virtual machine.
6. Start the Tesseract container via `docker run serve`, listening on port `8000`.

This process is illustrated within the following Bash script: {download}`create-vm-azure.sh </downloads/create-vm-azure.sh>`.

```{warning}
This script will likely not work out of the box for your setup and contains placeholders for actual resources, endpoints, and credentials. It assumes you have a resource group, VNet, Subnet, and Azure Container Registry set up.
```

Download the script, populate the variables at the beginning about your infrastructure accordingly, and run it:

```console
$ bash create-vm-azure.sh vectoradd:latest
```

This Bash script assumes you have Docker installed and authenticated to your
Azure Container Registry. To login into the Container Registry, use the `az`
CLI:

```console
$ az login
$ az acr login --name <registry-name>
```
