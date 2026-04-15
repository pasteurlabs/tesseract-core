# Path Reference Example

A Tesseract that copies files and directories from `input_path` to `output_path`.
It demonstrates how to use `InputPathReference` / `OutputPathReference` in Tesseract
schemas and how to compose custom Pydantic validators on top of the built-in
path-handling behaviour.

## What `InputPathReference` / `OutputPathReference` do in a schema

`*PathReference`s are `Annotated[Path, AfterValidator(...)]` types that carry validation logic. Use `InputPathReference` in your `InputSchema` and `OutputPathReference` in your `OutputSchema`.

**`InputPathReference` fields** — caller sends a relative string, `apply` receives an absolute `Path`:

```
caller sends       →  "sample_8.json"
built-in resolves  →  Path("/tesseract/input_data/sample_8.json")   (checked: exists)
apply sees         →  Path("/tesseract/input_data/sample_8.json")
```

- Rejects any path that would escape `input_path` (path traversal protection).
- Raises `ValidationError` if the resolved path does not exist.

**`OutputPathReference` fields** — `apply` returns an absolute `Path`, caller receives a relative string:

```
apply returns      →  Path("/tesseract/output_data/sample_8.copy")
built-in strips    →  Path("sample_8.copy")                          (checked: exists)
caller receives    →  "sample_8.copy"
```

- Raises `ValidationError` if the path does not exist inside `output_path`.

## Composing user-defined validators

Use `compose_validator()` to attach custom `AfterValidator`s to a path reference.
User validators always receive **absolute paths**, regardless of whether the field
is an input or output:

```python
from tesseract_core.runtime.experimental import (
    InputPathReference, OutputPathReference, compose_validator,
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

InputPath = compose_validator(InputPathReference, AfterValidator(has_bin_sidecar))
OutputPath = compose_validator(OutputPathReference, AfterValidator(has_bin_sidecar))

class InputSchema(BaseModel):
    paths: list[InputPath]

class OutputSchema(BaseModel):
    paths: list[OutputPath]
```

**Input fields** — built-in resolves first, then user validators run (on absolute paths):

```
"sample_8.json"
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
