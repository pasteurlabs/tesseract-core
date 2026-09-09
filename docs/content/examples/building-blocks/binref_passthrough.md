# Returning on-disk arrays with `BinrefArray`

## Context

A transient solver often produces more output than fits comfortably in memory: a
2D field snapshotted at every timestep is a 3D array that grows as the simulation
runs. Streaming those snapshots to disk during the solve keeps peak memory
bounded — but returning them from a Tesseract normally means reading the whole
trajectory _back_ into memory just to hand it to the runtime, which then
serializes it out to disk again.

`BinrefArray` removes that round-trip. The solver writes each array straight to a
binref buffer on disk and returns a lightweight reference. When the client
negotiates `json+binref` output, the on-disk bytes are forwarded **verbatim** —
never read back into the server's memory. For any other format (`json`,
`base64`) the buffer is loaded once and encoded like a normal array, so the field
behaves identically to a plain `Array` either way.

```{note}
`BinrefArray` and `BinrefWriter` live in `tesseract_core.runtime.experimental`.
```

```{warning}
The memory saving only applies to `json+binref` output. If a client requests
`json` or `base64` from a `BinrefArray`-backed field, the buffer **must** be read
into memory to inline it — exactly the cost the reference exists to avoid. The
result is still correct, and the runtime emits a `RuntimeWarning` so the load is
not silent, but a Tesseract that returns arrays too large to fit in memory should
be served (and requested) with `json+binref`.
```

## Example Tesseract (`examples/binref_passthrough`)

The output fields are ordinary `Array` types — nothing about the schema signals
that the data lives on disk:

```{literalinclude} ../../../../examples/binref_passthrough/tesseract_api.py
:pyobject: OutputSchema
:language: python
```

`apply` runs a tiny explicit heat-diffusion solver, checkpointing the field at
every timestep. Each snapshot is written to disk as it is computed and never held
in memory beyond the current step:

```{literalinclude} ../../../../examples/binref_passthrough/tesseract_api.py
:pyobject: apply
:language: python
```

Serve it with `json+binref` output and the `.bin` files are written to the
mounted `--output-path`; the client reads them back from there:

```bash
tesseract run binref_passthrough apply \
    --output-path ./output \
    --output-format json+binref \
    '{"inputs": {"size": 16, "steps": 20}}'
```

Nothing changes for the caller — the SDK decodes the references into NumPy arrays
transparently:

```python
from tesseract_core import Tesseract

with Tesseract.from_image(
    "binref_passthrough", output_format="json+binref", output_path="./output"
) as sim:
    out = sim.apply({"size": 16, "steps": 20})

out["final"]          # a NumPy array
len(out["trajectory"]) # steps + 1 snapshots
```

## Constructing a `BinrefArray`

`BinrefArray` has no public constructor; pick a named one depending on where the
buffer comes from.

**`BinrefArray.write(arr)`** — write a NumPy array to its own buffer. Use it for
one-off outputs:

```python
final = BinrefArray.write(field)
```

**`BinrefWriter`** — pack many arrays into a few shared, rotating buffers instead
of one file per array. Use it when a Tesseract emits many small arrays (like a
per-timestep trajectory):

```python
with BinrefWriter() as checkpoints:
    trajectory = [checkpoints.write(field)]
    for _ in range(steps):
        field = step(field)
        trajectory.append(checkpoints.write(field))
```

**`BinrefArray.from_file(path, shape, dtype)`** — reference a buffer that some
other code already wrote, without going through NumPy at all. This is the path
for compiled solvers that emit their own binary output:

```python
run_compiled_solver(out="field.bin")  # writes into the output directory
final = BinrefArray.from_file("field.bin", shape=(size, size), dtype="float64")
```

The buffer path is resolved against the served `--output-path`, so it must be
relative to that directory (a bare filename works) or an absolute path. The data
must be C-contiguous, row-major, and match the declared `shape` and `dtype`; a
mismatch is caught when the output is validated.

## Differentiable outputs

Because a `BinrefArray` is fed to an ordinary `Array` field, it composes with
`Differentiable` with no extra work — a large gradient buffer can be forwarded
from disk exactly like any other output:

```python
from tesseract_core.runtime import Array, Differentiable, Float64

class OutputSchema(BaseModel):
    grad: Differentiable[Array[(None,), Float64]]

def apply(inputs):
    ...
    return OutputSchema(grad=BinrefArray.write(gradient))
```
