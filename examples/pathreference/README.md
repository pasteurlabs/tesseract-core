# Path Reference Example

A Tesseract that copies files and directories from `input_path` to `output_path`.
It demonstrates how to use `InputPathReference` / `OutputPathReference` in Tesseract
schemas and how to compose custom Pydantic validators on top of the built-in
path-handling behaviour.

## What `InputPathReference` / `OutputPathReference` do in a schema

These are `Annotated[Path, AfterValidator(...)]` types that carry their validation
logic directly. Use `InputPathReference` in your `InputSchema` and
`OutputPathReference` in your `OutputSchema`.

**`InputPathReference` fields** — caller sends a relative string, `apply` receives an absolute `Path`:

```
caller sends       →  "sample_8.json"
built-in resolves  →  Path("/tesseract/input_data/sample_8.json")   (checked: exists)
apply sees         →  Path("/tesseract/input_data/sample_8.json")
```

- Rejects any path that would escape `input_path` (path traversal protection).
- Raises `FileNotFoundError` if the resolved path does not exist.

**`OutputPathReference` fields** — `apply` returns an absolute `Path`, caller receives a relative string:

```
apply returns      →  Path("/tesseract/output_data/sample_8.copy")
built-in strips    →  Path("sample_8.copy")                          (checked: exists)
caller receives    →  "sample_8.copy"
```

- Raises `ValueError` if the path does not exist inside `output_path`.

## Composing user-defined validators

`AfterValidator`s placed on an `InputPathReference` or `OutputPathReference` field
are preserved. Because the built-in validator is the **innermost** `AfterValidator`,
it always runs first:

```python
def has_bin_sidecar(path: Path) -> Path:
    """Check that any binref JSON has its .bin sidecar present."""
    if path.is_file():
        name = bin_reference(path)
        if name is not None:
            bin = path.parent / name
            assert bin.exists(), f"Expected .bin file for json {path} not found at {bin}"
    else:
        raise ValueError(f"{path} does not exist or is not a file.")
    return path

class InputSchema(BaseModel):
    paths: list[Annotated[InputPathReference, AfterValidator(has_bin_sidecar)]]

class OutputSchema(BaseModel):
    paths: list[Annotated[OutputPathReference, AfterValidator(has_bin_sidecar)]]
```

**Input fields** — built-in validator runs **first**, user validators run after (on absolute paths):

```
"sample_8.json"
  → built-in         →  Path("/tesseract/input_data/sample_8.json")   (resolved + existence check)
  → has_bin_sidecar  →  Path("/tesseract/input_data/sample_8.json")   (checks .bin sidecar present)
  → apply receives   →  Path("/tesseract/input_data/sample_8.json")
```

**Output fields** — built-in validator runs **first**, user validators run after (on relative paths):

```
apply returns  →  Path("/tesseract/output_data/sample_8.copy")
  → built-in         →  Path("sample_8.copy")                          (existence check + prefix stripped)
  → has_bin_sidecar  →  Path("sample_8.copy")                          (user validation on relative path)
  → caller receives  →  "sample_8.copy"
```

## Test data

The test dataset (`test_cases/testdata/`) contains:

| File                                                               | Array encoding                                  |
| ------------------------------------------------------------------ | ----------------------------------------------- |
| `sample_0.json`, `sample_3.json`, `sample_6.json`, `sample_9.json` | `json` (inline)                                 |
| `sample_1.json`, `sample_4.json`, `sample_7.json`                  | `base64` (inline)                               |
| `sample_2.json`, `sample_5.json`, `sample_8.json`                  | `binref` (references the shared `.bin` sidecar) |
| `sample_dir/`                                                      | directory containing `data.json`                |

`generate_data.py` re-creates this dataset using a fixed RNG seed.
