# `FileReference`

## Content

Tesseract that mounts input and output directories as datasets.
To be used for Tesseracts with large inputs and/or outputs.

## Example Tesseract (`examples/filereference`)

Using `InputFileReference` and `OutputFileReference` you can
include references to files in the `InputSchema` and `OuputSchema` of a Tesseract.
The file reference schemas make sure that a file exists (either locally or in the Tesseract)
and resolve paths correctly in both `tesseract-runtime` and `tesseract run` calls.

```{literalinclude} ../../../../examples/filereference/tesseract_api.py
:pyobject: InputSchema
:language: python
```

```{literalinclude} ../../../../examples/filereference/tesseract_api.py
:pyobject: OutputSchema
:language: python
```

```{literalinclude} ../../../../examples/filereference/tesseract_api.py
:pyobject: apply
:language: python
```

For the `tesseract-runtime` command, paths are relative to the local input/output paths:

```bash
tesseract-runtime apply \
    --input-path ./testdata \
    --output-path ./output \
    '{"inputs": {"data": ["sample_0.json", "sample_1.json"]}}'
```

For the `tesseract run` command, the file
reference schemas resolve to the mounted input/output folders inside the
Tesseract:

```bash
tesseract run filereference apply \
    --input-path ./testdata \
    --output-path ./output \
    '{"inputs": {"data": ["sample_2.json", "sample_3.json"]}}'
```

For the Python API usage examples see `test_tesseract.py`.
