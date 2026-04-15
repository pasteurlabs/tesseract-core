# `PathReference`

## Context

Tesseract that mounts input and output directories as datasets.
To be used for Tesseracts with large inputs and/or outputs.

## Example Tesseract (`examples/pathreference`)

Using `InputPathReference` and `OutputPathReference` you can
include references to files or directories in the `InputSchema` and `OutputSchema` of a Tesseract.
The path reference schemas make sure that a path exists (either locally or in the Tesseract)
and resolve paths correctly in both `tesseract-runtime` and `tesseract run` calls.

```{literalinclude} ../../../../examples/pathreference/tesseract_api.py
:pyobject: InputSchema
:language: python
```

```{literalinclude} ../../../../examples/pathreference/tesseract_api.py
:pyobject: OutputSchema
:language: python
```

```{literalinclude} ../../../../examples/pathreference/tesseract_api.py
:pyobject: apply
:language: python
```

For the `tesseract-runtime` command, paths are relative to the local input/output paths:

```bash
tesseract-runtime apply \
    --input-path ./testdata \
    --output-path ./output \
    '{"inputs": {"paths": ["sample_0.json", "sample_1.json"]}}'
```

For the `tesseract run` command, the path
reference schemas resolve to the mounted input/output folders inside the
Tesseract:

```bash
tesseract run pathreference apply \
    --input-path ./testdata \
    --output-path ./output \
    '{"inputs": {"paths": ["sample_2.json", "sample_3.json"]}}'
```

For the Python SDK usage examples see `test_tesseract.py`.
