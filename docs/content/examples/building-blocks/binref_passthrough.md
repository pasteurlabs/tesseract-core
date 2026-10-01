# Returning on-disk arrays with `BinrefArray`

{gh-tree}`View on GitHub <examples/binref_passthrough>`

## Context

A transient solver often produces more output than fits comfortably in memory: a
2D field snapshotted at every timestep is a 3D array that grows as the simulation
runs. Streaming those snapshots to disk during the solve keeps peak memory
bounded. Returning them from a Tesseract, however, normally means reading the
whole trajectory _back_ into memory just to hand it to the runtime, which then
serializes it out to disk again.

`BinrefArray` removes that round-trip. The solver writes each array straight to a
binref buffer on disk and returns a lightweight reference. When the client
negotiates `json+binref` output, the server forwards the on-disk bytes
**verbatim** without reading them back into memory. For any other format (`json`,
`base64`) the buffer is loaded once and encoded like a normal array.

```{note}
`BinrefArray` and `BinrefWriter` live in `tesseract_core.runtime.experimental`.
```

```{warning}
The memory saving only applies to `json+binref` output. If a client requests
`json` or `base64` from a `BinrefArray`-backed field, the buffer **must** be read
into memory to inline it. The result is still correct, and the runtime emits a
`RuntimeWarning`. A Tesseract that returns arrays too large to fit in memory
should be served (and requested) with `json+binref`.
```

## Example Tesseract (`examples/binref_passthrough`)

The output fields are ordinary `Array` types, so nothing about the schema signals
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

Nothing changes for the caller. The SDK decodes the references into NumPy arrays
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

`BinrefArray` has no public constructor. Pick a named one depending on where the
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
checkpoints = BinrefWriter()
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

The buffer path must be relative to the served `--output-path` (a bare filename
works) and must not lead outside it. `from_file` also checks that the file exists
and holds at least `prod(shape) * dtype.itemsize` bytes past the given offset.

```{warning}
Because the buffer is forwarded without being read, the runtime **cannot** check
that the bytes on disk actually match what you declared. Beyond the file size, it
only validates the `shape` and `dtype` you passed to `from_file` against the
field's declared type, and a mismatch there raises when the output is validated.
The bytes themselves are trusted. The data must be **C-contiguous and
row-major**. Otherwise, the client silently reinterprets it as the wrong array.
Writing the buffer with the matching NumPy
`dtype` and `np.ascontiguousarray` (or via `BinrefArray.write`) avoids this.
```

## Forwarding gradients from an AD endpoint

Because `BinrefArray` is fed to an ordinary `Array` field, it works with the
gradient endpoints without special support. The arrays those endpoints return are
ordinary arrays too, so they can also be forwarded from disk. This helps when the
Jacobian is as large as the output or larger, since a solver that writes its
adjoint or tangent fields to disk can hand them straight back.

Mark the differentiable input and output on the schema:

```python
from tesseract_core.runtime import Array, Differentiable, Float64

class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float64]]

class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float64]]

def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(y=BinrefArray.write(solve(inputs.x)))
```

A `jacobian` endpoint returns `{output: {input: partial}}`, where each partial is
a normal array and can therefore be a `BinrefArray`:

```python
def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    dy_dx = compute_jacobian(inputs.x)  # solver streams this to disk
    return {"y": {"x": BinrefArray.write(dy_dx)}}
```

The same holds for the tangents and cotangents returned by
`jacobian_vector_product` and `vector_jacobian_product`.
