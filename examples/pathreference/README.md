# Path Reference Example

A Tesseract that copies files and directories from `input_path` to `output_path`.
It demonstrates how to use `Path` in Tesseract schemas and how to compose custom
Pydantic validators on top of the built-in path-handling behaviour.

## What `Path` does in a schema

When you annotate a field with `Path`, the schema generation layer automatically
injects path-handling validators at runtime.

**Input `Path` fields**

- Accept a _relative_ path string from the caller.
- Resolve it to an absolute path under the configured `--input-path`.
- Reject any path that would escape `input_path` (path traversal protection).
- Raise `FileNotFoundError` if the resolved path does not exist.
- Accept both files **and** directories (use `InputFileReference` for files only).

**Output `Path` fields**

- Accept the absolute path your `apply` function produces (e.g. `output_path / name`).
- Strip the `output_path` prefix, returning a _relative_ path to the caller.
- Raise `ValueError` if the path does not exist inside `output_path`.
- Accept both files **and** directories (use `OutputFileReference` for files only).

So from the caller's perspective, both inputs and outputs are relative path strings;
the runtime handles all absolute-path resolution transparently.

## Composing user-defined validators

`AfterValidator`s placed on a `Path`-annotated field are preserved, and in both
cases the user validator receives an already-resolved **absolute** `Path`:

```python
def has_bin_sidecar(path: Path) -> Path:
    """Check that any binref JSON has its .bin sidecar present."""
    if path.is_file():
        name = bin_reference(path)
        if name is not None:
            bin = path.parent / name
            assert bin.exists(), f"Expected .bin file for json {path} not found at {bin}"
    return path

class InputSchema(BaseModel):
    paths: list[Annotated[Path, AfterValidator(has_bin_sidecar)]]
```

The built-in path validators run at different points depending on direction:

**Input fields** — built-in validator runs **first**, user validators run after:

1. Raw string (e.g. `"sample_8.json"`) is resolved to an absolute path and checked
   for existence by the built-in input validator.
2. The resolved absolute `Path` is passed to `has_bin_sidecar`, which checks that
   the referenced `.bin` sidecar is present beside it.

**Output fields** — user validators run **first**, built-in validator runs after:

1. The absolute `Path` returned by `apply` (e.g. `output_path / "sample_8.copy"`)
   is passed to `has_bin_sidecar`, which checks the `.bin` sidecar was also copied.
2. The built-in output validator then confirms the path exists inside `output_path`
   and strips the prefix, returning a relative path to the caller.

This example uses output validators to confirm that `apply` copied the sidecar
`.bin` file alongside each JSON file.

## Test data

The test dataset (`test_cases/testdata/`) contains:

| File                                                               | Array encoding                                  |
| ------------------------------------------------------------------ | ----------------------------------------------- |
| `sample_0.json`, `sample_3.json`, `sample_6.json`, `sample_9.json` | `json` (inline)                                 |
| `sample_1.json`, `sample_4.json`, `sample_7.json`                  | `base64` (inline)                               |
| `sample_2.json`, `sample_5.json`, `sample_8.json`                  | `binref` (references the shared `.bin` sidecar) |
| `sample_dir/`                                                      | directory containing `data.json`                |

`generate_data.py` re-creates this dataset using a fixed RNG seed.

## Running

```bash
# local (no Docker)
uv run python test_tesseract.py

# build Docker image first, then re-run
uv run tesseract build .
uv run python test_tesseract.py
```
