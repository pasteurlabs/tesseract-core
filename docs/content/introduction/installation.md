# Installation

## Basic installation

```{note}
Before proceeding, make sure you have:
- A working installation of Docker ([Docker Desktop](https://www.docker.com/products/docker-desktop/) or [Docker Engine](#installation-docker))
- Python 3.10+, ideally in a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
```

Install Tesseract Core via `pip`:

```bash
$ pip install tesseract-core
```

Then, verify the installation:

```bash
$ tesseract list
```

If the output is an empty table, that's expected — the CLI is working correctly, you just don't have any Tesseracts built yet.

(installation-docker)=

## Installing Docker

[Docker Desktop](https://www.docker.com/products/docker-desktop/) ships with everything you need, including the Docker Engine CLI, Docker Compose, and Docker Buildx.
It is available for Windows, macOS, and Linux (Debian- and Fedora-based distros).

If your system is not supported by Docker Desktop, or you prefer a more minimal setup, install the [`docker` engine CLI](https://docs.docker.com/engine/install/) together with [`docker-buildx`](https://github.com/docker/buildx).

### Running Docker without `sudo`

To use Tesseract without `sudo`, add your user to the `docker` group (see [Docker post-install docs](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)):

```bash
$ sudo usermod -aG docker $USER
```

Then log out and back in to apply the changes.

```{warning}
Using `sudo tesseract` may bypass your virtual environment and shadow the `tesseract` command with [conflicting executables](#exe-conflicts). Instead, add your user to the `docker` group and omit `sudo`.
```

(installation-podman)=

## Using alternative container engines

The container engine can be configured with the `TESSERACT_DOCKER_EXECUTABLE` environment variable. Any engine with a `docker`-compatible CLI (e.g. `podman`) is supported.

```bash
$ export TESSERACT_DOCKER_EXECUTABLE=podman
$ echo "export TESSERACT_DOCKER_EXECUTABLE=podman" >> ~/.bashrc
```

(installation-runtime)=

## Runtime installation

Installing the Tesseract Runtime directly (without Docker) is useful for debugging during Tesseract creation and for non-containerized deployment (see {ref}`running-without-containers`). To install it:

```bash
$ pip install tesseract-core[runtime]
```

```{tip}
Some shells treat `[` and `]` as special characters. If the command above fails, escape them (`tesseract-core\[runtime\]`) or use quotes (`"tesseract-core[runtime]"`).
```

(installation-issues)=

## Common issues

(windows-support)=

### Windows support

Tesseract is fully supported on Windows via the Windows Subsystem for Linux (WSL). See the [official WSL documentation](https://docs.microsoft.com/en-us/windows/wsl/) for setup instructions.

(exe-conflicts)=

### Conflicting executables

Other software shares the "Tesseract" name (notably [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)). If both are installed, you may see errors like:

```
$ tesseract build examples/vectoradd/ vectoradd

read_params_file: Can't open vectoradd
Error in findFileFormatStream: failed to read first 12 bytes of file
Error during processing.
```

We recommend always using a dedicated Python virtual environment. If you use `zsh`, you may also need to refresh the shell's executable cache:

```bash
$ hash -r
```

To confirm which executable `tesseract` resolves to:

```bash
$ which tesseract
```

### Missing user privileges

If you lack permissions to access the Docker daemon, commands like `tesseract build` will fail:

```bash
$ tesseract build examples/helloworld
RuntimeError: Could not reach Docker daemon, check if it is running. See logs for details.
```

Resolve this by adding your user to the `docker` group as described in [Running Docker without sudo](#installation-docker).

(installation-dev)=

## Development installation

To install everything needed for development on Tesseract Core itself (editable source, runtime, and test dependencies):

```bash
$ git clone git@github.com:pasteurlabs/tesseract-core.git
$ cd tesseract-core
$ pip install -e .[dev]
$ pre-commit install
```
