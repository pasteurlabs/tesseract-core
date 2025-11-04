# Installing local dependencies into a Tesseract

## Context
Sometimes it might be necessary to bundle local dependencies into a Tesseract;
this can be done by simply adding their path to the `tesseract_requirements.txt` file.
Both absolute and relative paths work, but in case they are relative paths, they should be
relative to the Tesseract's root folder (i.e., the one which contains the `tesseract_api.py` file)

## Example Tesseract
Here's an example: let's initialize an empty tesseract via
```bash
$ tesseract init --name cowsay
 [i] Initializing Tesseract cowsay in directory: .
 [i] Writing template tesseract_api.py to tesseract_api.py
 [i] Writing template tesseract_config.yaml to tesseract_config.yaml
 [i] Writing template tesseract_requirements.txt to tesseract_requirements.txt
 ```

 And let's then edit the `tesseract_api.py` file to read

```{literalinclude} ../../../../examples/conda/tesseract_api.py
:language: python
```


This Tesseract will accept a message like "Hello, world!" as an input and return
```{literalinclude} ../../../../examples/conda/expected_output.txt
:language: text
```

but in order to do so, it will need the dependency `cowsay` installed. We could just
add `cowsay` to the `tesseract_requirements.txt` file, but that would install it from PyPI every
time. Let's instead download it via pip download and then bundle it into a Tesseract; in order to
do the former, we can run:
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

```{note}
If you have additional Python modules in the same directory as `tesseract_api.py`, you can import them directly without any special configuration. The directory containing `tesseract_api.py` is automatically added to `PYTHONPATH` in the container environment.
```

Finally, let's build the Tesseract, and verify it works
```bash
$ tesseract build .
 [i] Building image ...
 [i] Built image sha256:7d024, ['cowsay:latest']

$ tesseract run install_tarball apply '{"inputs": {"message": "Hello, World!"}}'
{"out":"  _____________\n| Hello, World! |\n  =============\n             \\\n              \\\n                ^__^\n                (oo)\\_______\n                (__)\\       )\\/\\\n                    ||----w |\n                    ||     ||"}
```
