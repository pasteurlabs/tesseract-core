# File IO with `InputPath` / `OutputPath`

[View on GitHub](https://github.com/pasteurlabs/tesseract-core/tree/main/examples/file_io)

## Context

Instead passing file constents to the input payload, a Tesseract can
declare `InputPath` / `OutputPath` fields that refer to files or directories on
the `--input-path` / `--output-path` mounts. This is useful when inputs or
outputs are large on disk, or consist of many files.

## Example Tesseract (`examples/file_io`)

Using `InputPath` and `OutputPath` you can
include references to files or directories in the `InputSchema` and `OutputSchema` of a Tesseract.
The schemas make sure that a path exists (either locally or in the Tesseract)
and resolve paths correctly in both `tesseract-runtime` and `tesseract run` calls.

```{literalinclude} ../../../../examples/file_io/tesseract_api.py
:pyobject: InputSchema
:language: python
```

```{literalinclude} ../../../../examples/file_io/tesseract_api.py
:pyobject: OutputSchema
:language: python
```

```{literalinclude} ../../../../examples/file_io/tesseract_api.py
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
tesseract run file_io apply \
    --input-path ./testdata \
    --output-path ./output \
    '{"inputs": {"paths": ["sample_2.json", "sample_3.json"]}}'
```

## What `InputPath` / `OutputPath` do in a schema

`InputPath` and `OutputPath` are Pydantic-aware `Path` subclasses that carry validation logic. Use `InputPath` in your `InputSchema` and `OutputPath` in your `OutputSchema`.

**`InputPath` fields** — caller sends a relative string, `apply` receives an absolute `Path`:

```
caller sends → "sample_8.json"
  → built-in   →  Path("/tesseract/input_data/sample_8.json")   (resolved + existence check)
  → apply sees →  Path("/tesseract/input_data/sample_8.json")
```

- Rejects any path that would escape `input_path` (path traversal protection).
- Raises `ValidationError` if the resolved path does not exist.

**`OutputPath` fields** — `apply` returns an absolute `Path`, caller receives a relative string:

```
apply returns  →  Path("/tesseract/output_data/sample_8.copy")
  → built-in         →  Path("sample_8.copy")                          (existence check + prefix stripped)
  → caller receives  →  "sample_8.copy"
```

- Raises `ValidationError` if the path does not exist inside `output_path`.

## Composing user-defined validators

Use `Annotated` with `AfterValidator` to attach custom validators to a path
reference. Validators always receive **absolute paths** for both input and
output types:

```python
from typing import Annotated
from pydantic import AfterValidator
from tesseract_core.runtime.experimental import InputPath, OutputPath

def has_bin_sidecar(path: Path, info) -> Path:
    """Check that any binref JSON has its .bin sidecar present."""
    if path.is_file():
        name = bin_reference(path)
        if name is not None:
            bin = path.parent / name
            assert bin.exists(), f"Expected .bin file for json {path} not found at {bin}"
    elif path.is_dir():
        return path
    else:
        raise ValueError(f"{path} does not exist.")
    return path

CheckedInputPath = Annotated[InputPath, AfterValidator(has_bin_sidecar)]
CheckedOutputPath = Annotated[OutputPath, AfterValidator(has_bin_sidecar)]

class InputSchema(BaseModel):
    paths: list[CheckedInputPath]

class OutputSchema(BaseModel):
    paths: list[CheckedOutputPath]
```

**Input fields** — built-in resolves first, then user validators run (on absolute paths):

```
caller sends → "sample_8.json"
  → built-in         →  Path("/tesseract/input_data/sample_8.json")   (resolved + existence check)
  → has_bin_sidecar  →  Path("/tesseract/input_data/sample_8.json")   (checks .bin sidecar present)
  → apply receives   →  Path("/tesseract/input_data/sample_8.json")
```

**Output fields** — user validators run on absolute paths, built-in strips on serialization:

```
apply returns  →  Path("/tesseract/output_data/sample_8.copy")
  → built-in         →  Path("/tesseract/output_data/sample_8.copy")   (existence check)
  → has_bin_sidecar  →  Path("/tesseract/output_data/sample_8.copy")   (checks .bin sidecar present)
  → caller receives  →  "sample_8.copy"                                (prefix stripped on serialization)
```
