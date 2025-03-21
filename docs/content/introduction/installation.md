# Installation

## Dependencies

Tesseract Core depends on Docker.
If you are on macOS, installing Docker Desktop includes several plugins for extended functionality needed by Tesseract Core.
If you are on Linux and / or prefer to work via CLIs, you will need to install these plugin packages in addition to `docker`.
These are 

1. [`docker-buildx`](https://github.com/docker/buildx)
2. [`docker-compose`](https://github.com/docker/compose)

## Basic installation

```{note}
Before proceeding, make sure you have a [working installation of Docker](https://docs.docker.com/engine/install/) and a modern Python installation (Python 3.10+), ideally in a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
```

The simplest way to install Tesseract Core is via `pip`:

```bash
$ pip install tesseract-core
```

Then, verify everything is working as intended:

```bash
$ tesseract list
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

### User privileges

If you are on Linux and not a regular Docker user, or are used to running Docker via the command line using `sudo`, you may find that `tesseract build` results in the following exception

```
$ tesseract build examples/helloworld
RuntimeError: Could not reach Docker daemon, check if it is running. See logs for details.
```

Your natural inclination may be to run this with elevated privileges, however if you do this it's possible that your system will ignore the `tesseract` command in your virtual environment, and instead execute [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

```
$ sudo tesseract build examples/helloworld
Error opening data file /usr/share/tessdata/eng.traineddata
Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory.
Failed loading language 'eng'
...
```

You can resolve this by omitting the `sudo` command, and instead adding your user to the `docker` group.
See [Linux post-installation steps for Docker Engine > Manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

(installation-dev)=
## Development installation

If you would like to install everything you need for dev work on Tesseract itself (editable source, runtime + dependencies for tests), run this instead:

```bash
$ git clone git@github.com:pasteurlabs/tesseract-core.git
$ cd tesseract-core
$ pip install -e .[dev]
$ pre-commit install
```
