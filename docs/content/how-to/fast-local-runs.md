---
og:title: "Fast same-machine runs with shared-memory binref"
og:description: "Exchange large arrays with a served Tesseract through shared memory instead of base64 over HTTP, for much lower overhead on same-machine workloads."
---

# Fast same-machine runs with shared-memory binref

When the client and a served Tesseract run on the same machine, most of the
per-call overhead for large arrays comes from moving array bytes over HTTP. The
default `json+base64` encoding copies every array into the request or response
body, base64-encodes it, and copies it out again on the other side. For arrays
in the tens or hundreds of megabytes, that encode/transfer/decode round-trip
dominates the call.

You can avoid it. Point a served Tesseract at a shared-memory directory
(`/dev/shm` on Linux) and use `json+binref` encoding: arrays are written to
`.bin` files in that directory, and only lightweight file references travel over
HTTP. Because `/dev/shm` is a `tmpfs` shared between the host and the container,
the array data never leaves memory and is never copied through the socket.

```{note}
This is a Linux optimization. It relies on `/dev/shm` (a shared-memory `tmpfs`)
being available and bind-mounted into the container. It is most useful when the
client and the Tesseract share a host; for remote Tesseracts, array data has to
cross the network regardless, so a compact wire encoding is what matters instead
(see {doc}`/content/reference/array-encodings`).
```

## Basic usage

Pass shared-memory directories as the input and output paths and select
`json+binref` as the output format:

```python
import tempfile
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

shm = Path("/dev/shm")
x = np.random.default_rng(0).standard_normal(10_000_000)

# Use TemporaryDirectory so the scratch dirs (and any .bin files left in them)
# are removed on exit, rather than accumulating on /dev/shm.
with (
    tempfile.TemporaryDirectory(prefix="tess_in_", dir=shm) as input_dir,
    tempfile.TemporaryDirectory(prefix="tess_out_", dir=shm) as output_dir,
    Tesseract.from_image(
        "my-tesseract",
        input_path=input_dir,
        output_path=output_dir,
        output_format="json+binref",
    ) as t,
):
    result = t.apply({"x": x})
    # result["y"] is a NumPy array, decoded from a .bin file on /dev/shm
```

The `output_format` argument controls only how arrays are exchanged internally;
`apply` still returns ordinary NumPy arrays, so the rest of your code does not
change. `input_path` and `output_path` are bind-mounted into the container.

Input `.bin` files written for each request are cleaned up after that request.
Output `.bin` files, however, are written into `output_path` by the server and
are _not_ removed automatically — the directory is yours to manage. Wrapping the
scratch directories in `tempfile.TemporaryDirectory()`, as above, is the easiest
way to guarantee they are cleared when you are done.

```{tip}
`/dev/shm` is a `tmpfs` with its own size cap (often half of host RAM by
default), separate from any container memory limit, and it holds all data in
memory. Large arrays and repeated runs fill it up, so remove scratch directories
when you are done and size `/dev/shm` for your largest expected payload.
```

## The `binref_pool` fast path

By default, each request writes its input arrays to freshly allocated `.bin`
files. Allocating and faulting in fresh pages for a large array is a significant
part of the remaining cost. The opt-in `binref_pool=True` flag addresses this:

```python
with Tesseract.from_image(
    "my-tesseract",
    input_path=input_dir,
    output_path=output_dir,
    output_format="json+binref",
    binref_pool=True,
) as t:
    result = t.apply({"x": x})
```

With the pool enabled, the client reuses a small set of pre-faulted,
memory-mapped input buffers instead of writing a fresh file per request, and (on
POSIX platforms) decodes outputs as zero-copy memory-mapped views rather than
eager copies. On the shared-memory setup above, this roughly halves the
remaining overhead for large arrays.

Enabling the pool emits a one-time warning reminding you that the scratch `.bin`
files are not cleaned up automatically, since the pool makes it easy to run many
large requests in a row. Manage the scratch directories as shown in
[Basic usage](#basic-usage) above.

Two things to keep in mind when opting in:

- **Decoded results are read-only views** backed by the output files, valid only
  until the next request or until teardown. If you need to keep a result across
  calls or mutate it in place, copy it first (for example `np.array(result["y"])`),
  or leave the pool off. On non-POSIX platforms the eager, owned, writable decode
  is used regardless, so this caveat does not apply there.
- **The pool holds some resident memory** until the Tesseract is torn down.

The pool has no effect for output formats other than `json+binref`.

## When this helps

The benefit grows with array size. The table below shows median per-call
overhead for a no-op Tesseract (which only decodes inputs and encodes outputs)
across encodings, measured on one bare-metal Linux machine with Docker and
loopback networking. Absolute numbers depend heavily on hardware; treat them as
illustrative of the _shape_ of the trade-off, not as guarantees.

| Array size (float64)  | HTTP + base64 | HTTP + shmem binref | HTTP + shmem binref, `binref_pool` |
| --------------------- | ------------- | ------------------- | ---------------------------------- |
| 1,000 (~8 kB)         | ~3.8 ms       | ~4.5 ms             | ~4.4 ms                            |
| 100,000 (~0.8 MB)     | ~9.2 ms       | ~5.7 ms             | ~4.9 ms                            |
| 10,000,000 (~76 MB)   | ~1,130 ms     | ~206 ms             | ~87 ms                             |
| 100,000,000 (~760 MB) | ~13,170 ms    | ~1,990 ms           | ~880 ms                            |

For small arrays, base64 is competitive or slightly faster: the payload is tiny,
so HTTP round-trip latency dominates and the extra file handling of binref is
pure overhead. The crossover in this measurement is around 100 kB. For large
arrays the shared-memory path pulls far ahead, and the pool roughly halves the
overhead again.

If your arrays are small, or your Tesseract runs remotely, stay with the default
`json+base64`. Reach for shared-memory binref when you are passing large arrays
to a Tesseract on the same machine and per-call overhead is limiting you.

For the broader picture of where overhead comes from and how to reason about it,
see {doc}`/content/concepts/performance`. For encoding-format details, see
{doc}`/content/reference/array-encodings`.
