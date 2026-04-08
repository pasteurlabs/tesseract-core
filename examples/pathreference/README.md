# Path Reference Example

A Tesseract that copies files and directories from `input_path` to `output_path`.
It demonstrates how to use `Path` in Tesseract schemas and how to compose custom
Pydantic validators on top of the built-in path-handling behaviour.

## What `Path` does in a schema

When you annotate a field with `Path`, the schema generation layer automatically
replaces it with `InputPathReference` on inputs and `OutputPathReference` on outputs.

```python
class InputSchema(BaseModel):
    paths: list[Path]          # → list[InputPathReference] at runtime

class OutputSchema(BaseModel):
    paths: list[Path]          # → list[OutputPathReference] at runtime
```

**`InputPathReference`** (inputs)

- Accepts a _relative_ path string from the caller.
- Resolves it to an absolute path under the configured `--input-path`.
- Rejects any path that would escape `input_path` (path traversal protection).
- Raises `FileNotFoundError` if the resolved path does not exist.
- Accepts both files **and** directories (use `InputFileReference` for files only).

**`OutputPathReference`** (outputs)

- Accepts the absolute path your `apply` function produces (e.g. `output_path / name`).
- Strips the `output_path` prefix, returning a _relative_ path to the caller.
- Raises `ValueError` if the path does not exist inside `output_path`.
- Accepts both files **and** directories (use `OutputFileReference` for files only).

So from the caller's perspective, both inputs and outputs are relative path strings;
the runtime handles all absolute-path resolution transparently.

## Composing user-defined validators

`AfterValidator`s placed on a `Path`-annotated field are preserved and run _after_
the built-in path resolution. The user validator therefore always receives an
already-resolved, validated absolute `Path`:

```python
def has_bin_sidecar(path: Path) -> Path:
    """Check that any binref JSON has its .bin sidecar present."""
    if path.is_file():
        name = bin_reference(path)
        if name is not None:
            bin = path.parent / name
            assert bin.exists(), f"Expected .bin file for json {json} not found at {bin}"
    return path

class InputSchema(BaseModel):
    paths: list[Annotated[Path, AfterValidator(has_bin_sidecar)]]
```

Execution order for each element of `paths`:

1. Raw string (e.g. `"sample_8.json"`) is validated by `InputPathReference`:
   resolves → `/abs/input_path/sample_8.json`, checks it exists.
2. The resolved `Path` is passed to `next_to_binary_path`, which reads the JSON
   and checks that the referenced `.bin` sidecar is present beside it.

The same pattern applies to `OutputSchema`: the validator runs after
`OutputPathReference` has verified the output file exists and stripped the prefix.
This example uses it to confirm that `apply` also copied the sidecar `.bin` file.

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
