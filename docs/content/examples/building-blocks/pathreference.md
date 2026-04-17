# `PathReference`

## Context

Tesseract that mounts input and output directories as datasets.
To be used for Tesseracts with large inputs and/or outputs.

## Example Tesseract (`examples/pathreference`)

Using `InputPath` and `OutputPath` you can
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

## What `InputPath` / `OutputPath` do in a schema

`*Path`s are `Annotated[Path, AfterValidator(...)]` types that carry validation logic. Use `InputPath` in your `InputSchema` and `OutputPath` in your `OutputSchema`.

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

Use `compose_validator()` to attach custom `AfterValidator`s to a path reference.
User validators always receive **absolute paths**, regardless of whether the field
is an input or output:

```python
from tesseract_core.runtime.experimental import (
    InputPath, OutputPath, compose_validator,
)

def has_bin_sidecar(path: Path) -> Path:
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

InputPath = compose_validator(InputPath, AfterValidator(has_bin_sidecar))
OutputPath = compose_validator(OutputPath, AfterValidator(has_bin_sidecar))

class InputSchema(BaseModel):
    paths: list[InputPath]

class OutputSchema(BaseModel):
    paths: list[OutputPath]
```

**Input fields** — built-in resolves first, then user validators run (on absolute paths):

```
caller sends → "sample_8.json"
  → built-in         →  Path("/tesseract/input_data/sample_8.json")   (resolved + existence check)
  → has_bin_sidecar  →  Path("/tesseract/input_data/sample_8.json")   (checks .bin sidecar present)
  → apply receives   →  Path("/tesseract/input_data/sample_8.json")
```

**Output fields** — user validators run first (on absolute paths), then built-in strips:

```
apply returns  →  Path("/tesseract/output_data/sample_8.copy")
  → has_bin_sidecar  →  Path("/tesseract/output_data/sample_8.copy")   (checks .bin sidecar present)
  → built-in         →  Path("sample_8.copy")                          (existence check + prefix stripped)
  → caller receives  →  "sample_8.copy"
```
