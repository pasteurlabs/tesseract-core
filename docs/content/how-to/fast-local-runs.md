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

You can avoid that with `json+binref` encoding, which writes arrays to `.bin`
files in a scratch directory and sends only lightweight file references over
HTTP, so array data is never copied through the socket. If that directory is on
a shared-memory `tmpfs` (`/dev/shm` on Linux), the array data also never leaves
memory.

```{note}
This only helps when the client and the Tesseract share a host. For remote
Tesseracts, array data has to cross the network regardless, so a compact wire
encoding is what matters instead (see
{doc}`/content/reference/array-encodings`).
```

On Windows there is no shared-memory filesystem, but `json+binref` still works
on an ordinary directory and should still outperform `json+base64` for large arrays
by keeping array data out of the HTTP body.

## Basic usage

Select `json+binref` as the output format. `Tesseract.from_source` and
`Tesseract.from_image` take the same arguments:

::::{tab-set}
:::{tab-item} from_source
:sync: source

```python
import numpy as np
from tesseract_core import Tesseract

x = np.random.default_rng(0).standard_normal(10_000_000)

with Tesseract.from_source(
    "/path/to/tesseract_api.py", output_format="json+binref"
) as t:
    result = t.apply({"x": x})
    # result["y"] is a NumPy array, decoded from a .bin file
```

:::
:::{tab-item} from_image
:sync: image

```python
import numpy as np
from tesseract_core import Tesseract

x = np.random.default_rng(0).standard_normal(10_000_000)

with Tesseract.from_image("my-tesseract", output_format="json+binref") as t:
    result = t.apply({"x": x})
    # result["y"] is a NumPy array, decoded from a .bin file
```

:::
::::

The `output_format` argument controls only how arrays are exchanged internally;
`apply` still returns ordinary NumPy arrays, so the rest of your code does not
change. Scratch directories for the `.bin` files are created in the system's
temporary directory and removed when the Tesseract object is garbage collected.
For a containerized Tesseract, they are bind-mounted into the container.

This is enough to get most of the benefit. Measured on an Apple silicon laptop,
a 40 MB `float64` round trip takes ~26 ms this way against ~168 ms for the
default `json+base64`.

## Using shared memory

To keep the `.bin` files in memory, pass directories on a shared-memory
filesystem as the input and output paths:

::::{tab-set}
:::{tab-item} from_source
:sync: source

```python
import tempfile
from pathlib import Path

shm = Path("/dev/shm")

# Use TemporaryDirectory so the scratch dirs (and any .bin files left in them)
# are removed on exit, rather than accumulating on /dev/shm.
with (
    tempfile.TemporaryDirectory(prefix="tess_in_", dir=shm) as input_dir,
    tempfile.TemporaryDirectory(prefix="tess_out_", dir=shm) as output_dir,
    Tesseract.from_source(
        "/path/to/tesseract_api.py",
        input_path=input_dir,
        output_path=output_dir,
        output_format="json+binref",
    ) as t,
):
    result = t.apply({"x": x})
```

:::
:::{tab-item} from_image
:sync: image

```python
import tempfile
from pathlib import Path

shm = Path("/dev/shm")

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
```

:::
::::

```{note}
For a containerized Tesseract, shared memory needs a Linux host. On macOS and
Windows, the container runs inside a Linux VM, so the bind mounts cross the VM
boundary and the container does not see the host's `tmpfs` as the same memory.
`json+binref` on an ordinary directory still works there. The size cap of
`/dev/shm` is separate from any container memory limit set with `memory=...`.
```

Input `.bin` files written for each request are cleaned up after that request.
Output `.bin` files, however, are written into `output_path` by the server and
are _not_ removed automatically. Wrapping the
scratch directories in `tempfile.TemporaryDirectory()`, as above, is the easiest
way to guarantee they are cleared when you are done.

### How big does it need to be?

**At least the size of your inputs plus your outputs.** Both files exist at the
same time, so a 200 MB input with a 200 MB output needs 400 MB free. With
`experimental_binref_pool` enabled, an input-sized buffer stays allocated
between calls.

`/dev/shm` is a `tmpfs` with its own size cap (often half of host RAM by
default), and it holds all data in memory. Large arrays and repeated runs fill it
up, so remove scratch directories when you are done and size it for your largest
expected payload.

```{warning}
Allocating too much memory relative to your RAM can crash your machine
(we have observed crashes on macOS when using a 4GB hfs on a 16GB machine).
With very large payloads it may be safer to stick with `json+binref` on an
ordinary disk-backed directory.
```

### Shared memory on macOS

On Linux, `/dev/shm` is already present,
but macOS requires creating a shared-memory filesystem, either through tmpfs:

```bash
mkdir -p /tmp/tess-shm
sudo mount_tmpfs -s 1g /tmp/tess-shm   # a cap, not a reservation
```

Or hfs, which we have measured to be about 15% faster than `tmpfs` but requires more conscious cleanup:

```bash
disk=$(hdiutil attach -nomount ram://2097152)   # 1 GB
newfs_hfs -v tess-shm "$disk"
mkdir -p /tmp/tess-shm && mount -t hfs "$disk" /tmp/tess-shm
# when finished:  umount /tmp/tess-shm && hdiutil detach "$disk"
```

Then use `/tmp/tess-shm` in place of `/dev/shm` in the `from_source` example above. A containerized Tesseract can't share this memory with the host (see the note above).

## The `experimental_binref_pool` fast path

By default, each request writes its input arrays to freshly allocated `.bin`
files. Allocating and faulting in fresh pages for a large array is a significant
part of the remaining cost. The opt-in `experimental_binref_pool=True` flag
addresses this:

::::{tab-set}
:::{tab-item} from_source
:sync: source

```python
with Tesseract.from_source(
    "/path/to/tesseract_api.py",
    input_path=input_dir,
    output_path=output_dir,
    output_format="json+binref",
    experimental_binref_pool=True,
) as t:
    result = t.apply({"x": x})
```

:::
:::{tab-item} from_image
:sync: image

```python
with Tesseract.from_image(
    "my-tesseract",
    input_path=input_dir,
    output_path=output_dir,
    output_format="json+binref",
    experimental_binref_pool=True,
) as t:
    result = t.apply({"x": x})
```

:::
::::

With the pool enabled, the client reuses a small set of pre-faulted,
memory-mapped input buffers instead of writing a fresh file per request, and
decodes outputs as zero-copy memory-mapped views rather than eager copies. On a
shared-memory setup, this roughly halves the remaining overhead for large arrays
(see the table below). On the Apple silicon laptop from the basic example, with
the scratch directories on a shared-memory mount, the same 40 MB round trip
takes ~20 ms.

The server's output `.bin` files are unlinked as soon as they are decoded, so
they do not accumulate on disk: each result array keeps its own memory map (and
thus the backing data) alive, and the space is reclaimed once you drop the array.

Two things to keep in mind when opting in:

- **Decoded results are read-only views** backed by memory-mapped files. If you
  need to mutate a result in place, copy it first (for example
  `np.array(result["y"])`), or leave the pool off.
- **The pool holds some resident memory** until the Tesseract is torn down, and
  each result you keep pins its backing pages until it is dropped.

The pool needs the binref directory to be **memory-backed**. That is what makes
its memory-mapped writes free; an ordinary disk-backed directory does not benefit
from this. It has no effect for output formats other than `json+binref`. With
`Tesseract.from_source`, the pool is available on any POSIX platform. With
`Tesseract.from_image`, it raises an error on anything but Linux, for the same
reason as shared memory above.

## When this helps

The benefit grows with array size. The table below shows median per-call
overhead for a no-op Tesseract (which only decodes inputs and encodes outputs)
across encodings, measured on one bare-metal Linux machine with Docker and
loopback networking. Absolute numbers depend heavily on hardware; treat them as
illustrative of the _shape_ of the trade-off, not as guarantees.

| Array size (float64)  | HTTP + base64 | HTTP + shmem binref | HTTP + shmem binref, `experimental_binref_pool` |
| --------------------- | ------------- | ------------------- | ----------------------------------------------- |
| 1,000 (~8 kB)         | ~3.8 ms       | ~4.5 ms             | ~4.4 ms                                         |
| 100,000 (~0.8 MB)     | ~9.2 ms       | ~5.7 ms             | ~4.9 ms                                         |
| 10,000,000 (~76 MB)   | ~1,130 ms     | ~206 ms             | ~87 ms                                          |
| 100,000,000 (~760 MB) | ~13,170 ms    | ~1,990 ms           | ~880 ms                                         |

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
