# Installing local Python modules into a Tesseract

## Context
Sometimes it might be necessary to bundle local Python modules into a Tesseract.

There are 2 ways to do this:

1. Make them a proper Python package with `pyproject.toml` and add the local path to the `tesseract_requirements.txt` file.
Both absolute and relative paths work, but in case they are relative paths, they should be
relative to the Tesseract's root folder (i.e., the one which contains the `tesseract_api.py` file).
2. Just put them as `.py` files next to `tesseract_api.py` and add them to `build_config.package_data` (see also [packagedata.md]) in `tesseract_config.yaml` to make sure they're being included in container builds.

## Example Tesseract

Here is an example Tesseract that highlights both ways to include dependencies.

```{literalinclude} ../../../../examples/localpackage/tesseract_api.py
:language: python
```

The custom package `helloworld` is shipped with a file `helloworld.py` and a `pyproject.toml` that ensures it can be built as a Python package.

```{literalinclude} ../../../../examples/localpackage/helloworld/helloworld.py
:language: python
```

```{literalinclude} ../../../../examples/localpackage/helloworld/pyproject.toml
:language: toml
```

Then, it can be added as a local dependency to `tesseract_requirements.txt` by passing a relative path:

```{literalinclude} ../../../../examples/localpackage/tesseract_requirements.txt
:language: text
```

The local module `goodbyeworld.py` is just a single Python file:

```{literalinclude} ../../../../examples/localpackage/goodbyeworld.py
:language: python
```

To ensure it gets added to built Tesseracts, we have to register it as `package_data`:

```{literalinclude} ../../../../examples/localpackage/tesseract_config.yaml
:language: yaml
```

Now we are ready to build the Tesseract. This Tesseract will accept a name like "Tessie" as an input and return a message:

```bash
$ tesseract build examples/localpackage
$ tesseract run localpackage apply '{"inputs": {"name": "Tessie"}}'
{"message":"Hello Tessie!\nGoodbye Tessie!"}
```

This confirms that both custom dependencies are available to the Tesseract container.

To run the Tesseract without containerization, we have to make sure that `helloworld` is installed into our dev environment:

```bash
$ pip install examples/localpackage/helloworld
$ TESSERACT_API_PATH=examples/localpackage/tesseract_api.py tesseract-runtime apply '{"inputs": {"name": "Tessie"}}'
{"message":"Hello Tessie!\nGoodbye Tessie!"}
```

## Advanced pattern: injecting private dependencies as local wheels

In case a Tesseract depends on private packages that are not accessible from the public Python package index (PyPI), injecting them as local files can be a useful way to side-step authentication issues.

This is especially powerful in conjunction with `pip download` (e.g. from a private PyPI registry) to obtain a pre-built wheel:

```bash
$ pip download cowsay==6.1
Collecting cowsay==6.1
  Obtaining dependency information for cowsay==6.1 from https://files.pythonhosted.org/packages/f1/13/63c0a02c44024ee16f664e0b36eefeb22d54e93531630bd99e237986f534/cowsay-6.1-py3-none-any.whl.metadata
  Downloading cowsay-6.1-py3-none-any.whl.metadata (5.6 kB)
Downloading cowsay-6.1-py3-none-any.whl (25 kB)
Saved ./cowsay-6.1-py3-none-any.whl
Successfully downloaded cowsay
```

We can then specify it as a local dependency in `tesseract_requirements.txt` by adding the following line:
```
./cowsay-6.1-py3-none-any.whl
```

Finally, let's build the Tesseract, and verify it works
```bash
$ tesseract build mytess
 [i] Building image ...
 [i] Built image sha256:7d024, ['mytess:latest']

$ tesseract run mytess apply '{"inputs": {"message": "Hello, World!"}}'
{"out":"  _____________\n| Hello, World! |\n  =============\n             \\\n              \\\n                ^__^\n                (oo)\\_______\n                (__)\\       )\\/\\\n                    ||----w |\n                    ||     ||"}
```
