# Sidecar model outputs with `Ref`

[View on GitHub](https://github.com/pasteurlabs/tesseract-core/tree/main/examples/ref_output)

## Context

The `json+binref` output format already keeps array _buffers_ out of the
response payload by writing them to `.bin` files. `Ref` applies the same idea
one level up the tree: it takes a nested Pydantic model — typically your own
domain object holding several arrays — and writes the whole model to its own
`.json` file, leaving only a relative path in the main payload.

This is useful when a Tesseract returns many structured objects (time steps,
cases, samples) and callers want to fetch them selectively rather than parse
one large response.

## Example Tesseract (`examples/ref_output`)

`Frame` is an ordinary user-defined model with two differentiable array leaves
and a plain string leaf. Wrapping it in `Ref[Frame]` is the only change needed:

```{literalinclude} ../../../../examples/ref_output/tesseract_api.py
:pyobject: Frame
:language: python
```

```{literalinclude} ../../../../examples/ref_output/tesseract_api.py
:pyobject: OutputSchema
:language: python
```

Run with an output path and the `json+binref` format, the response carries only
the sidecar paths:

```bash
tesseract-runtime \
    --output-path ./output \
    --output-format json+binref \
    apply '{"inputs": {"scale": 2.0, "n_nodes": 4, "n_frames": 3}}'
```

```json
{
  "frames": [
    { "object_type": "ref", "path": "frame_000.json" },
    { "object_type": "ref", "path": "frame_001.json" },
    { "object_type": "ref", "path": "frame_002.json" }
  ]
}
```

Each entry is self-describing, in the same way an encoded array carries
`"object_type": "array"`. That is what lets a client resolve refs without
knowing the Tesseract's schema (see [Reading refs back](reading-refs-back)).

```
output/
├── 49aa7689-….bin        # array buffers, shared across all frames
├── frame_000.json
├── frame_001.json
└── frame_002.json
```

Each sidecar is a normal encoded payload, whose arrays are in turn binrefs into
the shared buffer:

```json
{
  "name": "step_1",
  "displacement": {
    "object_type": "array",
    "shape": [4, 3],
    "dtype": "float32",
    "data": { "buffer": "49aa7689-….bin:80", "encoding": "binref" }
  },
  "pressure": {
    "object_type": "array",
    "shape": [4],
    "dtype": "float64",
    "data": { "buffer": "49aa7689-….bin:128", "encoding": "binref" }
  }
}
```

## When refs are written

Whether refs become files depends on the **output format**, not on the
transport: `Ref` writes a sidecar when the serialization context carries a
`base_dir`, which is what `json+binref` sets. That applies over HTTP too — a
served Tesseract writes sidecars into its per-request `run_<id>/` directory
under `--output-path`, next to the `.bin` buffer they point into.

With the `json` and `json+base64` formats there is no output directory to
write to, so refs are serialized **inline** and the same Tesseract still
returns a self-contained response:

```bash
tesseract-runtime apply '{"inputs": {"scale": 2.0, "n_nodes": 2, "n_frames": 1}}'
```

```json
{
  "frames": [
    {
      "name": "step_0",
      "displacement": {
        "object_type": "array",
        "shape": [2, 3],
        "dtype": "float32",
        "data": {
          "buffer": [
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0]
          ],
          "encoding": "json"
        }
      },
      "pressure": {
        "object_type": "array",
        "shape": [2],
        "dtype": "float64",
        "data": { "buffer": [0.0, 2.0], "encoding": "json" }
      }
    }
  ]
}
```

Validation accepts both forms, so a payload produced either way can be read
back. A ref is resolved against `base_dir`, the file is read, and the wrapped
model's own validators run on its contents — array shapes and dtypes inside a
sidecar are checked exactly as they would be inline. A bare path string is
accepted too, so hand-written payloads that just name the file keep working.

(reading-refs-back)=

## Reading refs back

The Python SDK resolves refs for you, provided it knows where the output
directory is:

```python
tess = Tesseract.from_image("ref_output", output_path="./output",
                            output_format="json+binref")
out = tess.apply({"scale": 2.0, "n_nodes": 2, "n_frames": 2})

out["frames"][1]["displacement"]   # -> np.ndarray, loaded from the sidecar
```

The client has no access to your `OutputSchema`, so it recognises a ref by its
`object_type` marker — exactly how it recognises encoded arrays — loads the
sidecar, and decodes the arrays inside it (including nested refs). Without an
`output_path` it raises rather than handing back an unresolvable path.

## Naming sidecar files

Sidecars are named `<prefix>_<index>.json`, where the prefix comes from the
first of these that is usable:

1. the prefix passed as `Ref[Frame, "frame"]` → `frame_000.json`;
2. the model's own `name` field → `step_0_000.json`;
3. the model's class name → `Frame_000.json`.

Failing all three, files are named with a UUID, matching how `json+binref`
names its `.bin` files. Because every name carries a running index, sidecars in
one payload can never collide — a `name` field need not be unique. A `name`
that is missing or not filename-safe (a path separator, say) quietly falls
through to the next option rather than sinking the whole response; a prefix
passed to `Ref` is developer-supplied, so an unusable one raises.

Counters restart on each dump, so two dumps into the _same_ directory overwrite
each other. A served Tesseract avoids this by writing into a per-request
`run_<id>/` directory.

```{note}
Linters read a bare string in a subscript as a forward reference, so
`Ref[Frame, "frame"]` may need a `# noqa: F821`. The same applies to Tesseract's
existing `Array[(2, 3), "float32"]` form.
```

```python
class OutputSchema(BaseModel):
    frames: list[Ref[Frame, "frame"]]
```

## Refs and differentiation

`Ref` is a thin `Annotated` wrapper, which keeps it transparent to schema
generation. The arrays inside a wrapped model stay visible at their natural
paths, so nothing has to be re-declared:

- `abstract_eval` replaces the arrays inside `Frame` with `ShapeDType` on its own.
- `jac_outputs` / `jvp_outputs` / `vjp_outputs` accept `frames.[0].displacement`
  and `frames.[1].pressure` as usual.

Gradient payloads are flat `{path: array}` mappings keyed by paths into the
output tree, so they never traverse a ref — no ref-specific handling is needed
in your gradient endpoints, and none of the array data is written to sidecars
there.

```{note}
`Ref` is part of `tesseract_core.runtime.experimental` and its API may change.
```
