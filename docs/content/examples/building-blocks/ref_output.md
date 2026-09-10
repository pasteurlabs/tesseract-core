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
{ "frames": ["frame_000.json", "frame_001.json", "frame_002.json"] }
```

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
  "name": "frame_001",
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

`Ref` writes a sidecar only when the serialization context carries a
`base_dir`, which is what `json+binref` sets. With the `json` and
`json+base64` formats there is nowhere to write, so refs are serialized
**inline** and the same Tesseract keeps working over plain HTTP:

```bash
tesseract-runtime apply '{"inputs": {"scale": 2.0, "n_nodes": 2, "n_frames": 1}}'
```

```json
{"frames": [{"name": "frame_000", "displacement": {...}, "pressure": {...}}]}
```

Validation accepts both forms, so a payload produced either way can be read
back. A path is resolved against `base_dir`, the file is read, and the wrapped
model's own validators run on its contents — array shapes and dtypes inside a
sidecar are checked exactly as they would be inline.

## Naming sidecar files

By default sidecars get UUID names, matching how `json+binref` names `.bin`
files. Give the wrapped model a `__ref_name__()` method to derive readable
names from the data instead. The stem must be unique across the payload;
duplicates and unsafe names are rejected rather than silently clobbered.

```{literalinclude} ../../../../examples/ref_output/tesseract_api.py
:pyobject: Frame.__ref_name__
:language: python
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
