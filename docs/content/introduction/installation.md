# Installation

## Basic installation

```{note}
Before proceeding, make sure you have a [working installation of Docker](https://docs.docker.com/engine/install/) and a modern Python installation (Python 3.10+).
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

Windows and WSL users can encounter an issue while trying to build a Tesseract:

```bash
$ tesseract build examples/vectoradd/ vectoradd

Uncaught error: [Errno 20] Not a directory: ...
# similarly
NotADirectoryError: [WinError 267] The directory name is invalid: ...
```

This error is caused by the fact that Tesseract uses symlinks, which are not properly supported by Git for Windows. See [this SuperUser thread](https://superuser.com/questions/1713099/symbolic-link-does-not-work-in-git-over-windows) for more information.

We are looking to improve support for Windows users, but in meantime consider one of the following workarounds:
* Instead of using a Windows-side Git client, use a Unix-based Git client when cloning `tesseract`, and clone to a Unix filesystem. Simply put, clone from your WSL shell.
* Git clone with `git clone -c core.symlinks=true <repository_url>`, as suggested [here](https://www.scivision.dev/git-windows-symlink/).

### Conflicting executables

"Tesseract" is widely known term, and other software projects adopted it too. This sometimes can lead to multiple executables with the same name, as can happen if you also have [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed. In that case, you may encounter following error:

```
$ tesseract build examples/vectoradd/ vectoradd

read_params_file: Can't open vectoradd
Error in findFileFormatStream: failed to read first 12 bytes of file
Error during processing.
```

To avoid it, we always recommend to use Tesseract in a separate Python virtual environment. Nevertheless, this error can still happen if you are a `zsh` shell user due to its way of caching paths to executables. If that's the case, consider refreshing the shell's hash with

```bash
$ hash -r
```

You can always confirm what executable the command `tesseract` corresponds with

```bash
$ which tesseract
```

(installation-dev)=
## Development installation

If you would like to install everything you need for dev work on Tesseract itself (editable source, runtime + dependencies for tests), run this instead:

```bash
$ git clone git@github.com:pasteurlabs/tesseract-core.git
$ cd tesseract-core
$ pip install -e .[dev]
$ pre-commit install
```
