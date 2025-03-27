# Installation

## Basic installation

```{note}
Before proceeding, make sure you have a working installation of Docker ([Docker Desktop](https://www.docker.com/products/docker-desktop/) or [Docker Engine](#installation-docker)) and a modern Python installation (Python 3.10+), ideally in a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
```

The simplest way to install Tesseract Core is via `pip`:

```bash
$ pip install tesseract-core
```

Then, verify everything is working as intended:

```bash
$ tesseract list
```

(installation-docker)=
## Installing Docker

[Docker Desktop](https://www.docker.com/products/docker-desktop/) ships with everything you need to run Tesseract Core, including the Docker Engine CLI, Docker Compose, and Docker Buildx. It also includes a GUI for managing containers and images.
It is available for Windows, macOS, and Linux for Debian and Fedora based distros.

If your system is not supported by Docker Desktop, or you prefer a more minimal setup, you will need to install the [`docker` engine CLI](https://docs.docker.com/engine/install/) together with some required plugins:

1. [`docker-buildx`](https://github.com/docker/buildx)
2. [`docker-compose`](https://github.com/docker/compose)

To use Tesseract without `sudo`, you will need to add your user to the `docker` group. See [Linux post-installation steps for Docker Engine > Manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user), or run:

```bash
$ sudo usermod -aG docker $USER
```

Then, log out and back in to apply the changes.

```{warning}
Using `sudo tesseract` may bypass active virtual environments and shadow the `tesseract` command with [conflicting executables](#exe-conflicts). To avoid this, make sure you're using the correct `tesseract` executable, or add your user to the `docker` group (and omit `sudo`).
```

(installation-runtime)=
## Runtime installation

Invoking the Tesseract Runtime directly without Docker can be useful for debugging during Tesseract creation and non-containerized deployment (see [here](#tr-without-docker)). To install it, run:

```bash
$ pip install tesseract-core[runtime]
```

```{warning}
Some shells use `[` and `]` as special characters, and might error out on the `pip install` line above. If that happens, consider escaping these characters, e.g. `-e .\[dev\]`, or enclosing them in double quotes, e.g. `-e ".[dev]"`.
```

(installation-issues)=
## Common issues

### Windows support

Tesseract is fully supported on Windows via the Windows Subsystem for Linux (WSL). For guidance, please refer to the [official documentation](https://docs.microsoft.com/en-us/windows/wsl/).

(exe-conflicts)=
### Conflicting executables

This is not the only software called "Tesseract". Sometimes, this leads to multiple executables with the same name, for example if you also have [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed. In that case, you may encounter the following error:

```
$ tesseract build examples/vectoradd/ vectoradd

read_params_file: Can't open vectoradd
Error in findFileFormatStream: failed to read first 12 bytes of file
Error during processing.
```

To avoid it, we always recommend to use Tesseract in a separate Python virtual environment. Nevertheless, this error can still happen if you are a `zsh` shell user due to its way of caching paths to executables. If that's the case, consider refreshing the shell's executable cache with

```bash
$ hash -r
```

You can always confirm what executable the command `tesseract` corresponds with

```bash
$ which tesseract
```

### Missing user privileges

If you lack permissions to access the Docker daemon, running e.g. `tesseract build` will result in the following exception:

```bash
$ tesseract build examples/helloworld
RuntimeError: Could not reach Docker daemon, check if it is running. See logs for details.
```

You can resolve this by adding your user to the `docker` group.
See [Linux post-installation steps for Docker Engine > Manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user), or run:

```bash
$ sudo usermod -aG docker $USER
```

Then, log out and back in to apply the changes.

(installation-dev)=
## Development installation

If you would like to install everything you need for dev work on Tesseract itself (editable source, runtime + dependencies for tests), run this instead:

```bash
$ git clone git@github.com:pasteurlabs/tesseract-core.git
$ cd tesseract-core
$ pip install -e .[dev]
$ pre-commit install
```
