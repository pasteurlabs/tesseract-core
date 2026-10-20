---
og:title: "Keep GPU arrays on the device with the cuda_ipc transport"
og:description: "Exchange GPU arrays with a served Tesseract by CUDA IPC handle instead of copying them through host memory, from the Python SDK, Tesseract-JAX, Tesseract-Torch, or raw HTTP."
---

# Keep GPU arrays on the device with `cuda_ipc`

By default, a GPU array that crosses into or out of a served Tesseract takes a
round trip through host memory. It is copied off the device, encoded into the
request or response body, decoded on the other side, and copied back onto the
GPU. For large arrays in a loop, such as an optimizer or a training step calling
a GPU-resident Tesseract, that round trip can dominate the cost of each call.

The `cuda_ipc` GPU transport avoids it. Instead of the array bytes, the request
or response carries a 64-byte [CUDA IPC](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)
handle that the other process uses to map the same device memory. Shapes,
dtypes, and schema validation stay in the JSON payload as usual, and CPU arrays
in the same payload are still encoded according to `output_format`.

```{warning}
The GPU transport is experimental. Its interface may change or be removed in
future releases.
```

## Requirements

- **Linux with an NVIDIA GPU.** The transport is built on CUDA IPC.
- **Client and Tesseract on the same machine, seeing the same GPU.** IPC handles
  are meaningless across hosts. The two processes may number the GPU differently,
  for example when a container sees only one of the host's GPUs.
- **A shared IPC namespace.** When the Tesseract runs in a container, it needs
  `--ipc=host`. The SDK adds this for you (see below).
- **GPU arrays on both sides.** The client passes arrays that live on a CUDA
  device, and the Tesseract returns arrays that live on the device. CPU arrays
  are unaffected.
- **Matching versions.** tesseract-core 1.15.0 or later on both sides, since
  earlier runtimes use a different wire format for `cuda_ipc`. For JAX and
  PyTorch, tesseract-jax 0.5.0 and tesseract-torch 0.2.0 or later.

## Write a Tesseract that returns GPU arrays

The runtime exports any output array that lives in GPU memory and exposes
`__cuda_array_interface__`, which includes CuPy, PyTorch, and JAX arrays. Inputs
that arrive by handle reach your `apply` function as framework-agnostic device
arrays that expose both `__cuda_array_interface__` and DLPack, so adopt them with
your framework before computing:

```python
import cupy as cp
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Float32


class InputSchema(BaseModel):
    a: Array[(None,), Float32] = Field(description="An arbitrary vector.")
    b: Array[(None,), Float32] = Field(description="Same shape as a.")
    s: float = Field(description="A scalar.", default=3.0)


class OutputSchema(BaseModel):
    result: Array[(None,), Float32] = Field(description="s * a + b, on the GPU.")


def apply(inputs: InputSchema) -> OutputSchema:
    # Inputs may arrive as NumPy arrays (host encodings) or as device arrays
    # (cuda_ipc). cp.asarray keeps device arrays on the GPU and moves host
    # arrays onto it.
    a = cp.asarray(inputs.a)
    b = cp.asarray(inputs.b)
    return OutputSchema(result=inputs.s * a + b)
```

With PyTorch, adopt inputs with `torch.as_tensor(x, device="cuda")` or
`torch.from_dlpack(x)`. With JAX, use `jnp.asarray(x)` or `jnp.from_dlpack(x)`.
Tesseracts built on the JAX recipe (`tesseract init --recipe jax`) adopt their
inputs this way already. Avoid `np.asarray` and anything else that goes through
`__array__`, since the device arrays support it for convenience by copying to
host.

Input dtypes must match the schema exactly. The host encodings cast an input to
the dtype its schema declares, but `cuda_ipc` does not cast on the device and
rejects a mismatched input instead.

The repository contains complete examples for CuPy, PyTorch, and JAX in
`examples/_gpu_cupy`, `examples/_gpu_torch`, and `examples/_gpu_jax`.

## Serve with the GPU transport

### From Python

Pass `gpu_transport="cuda_ipc"` when you create the Tesseract. For a container,
also give it GPU access:

```python
from tesseract_core import Tesseract

with Tesseract.from_image(
    "my-gpu-tesseract", gpus=["all"], gpu_transport="cuda_ipc"
) as tess:
    ...
```

The SDK runs the container with `--ipc=host` whenever the GPU transport is
enabled, and raises a `ValueError` if no GPUs were requested.

Without a container, `Tesseract.from_source` serves the Tesseract in a
subprocess. By default, that subprocess runs in an environment built from the
Tesseract's own requirements. To share your environment instead, including its
CUDA-enabled libraries, point it at your interpreter:

```python
import sys

with Tesseract.from_source(
    "path/to/tesseract_api.py",
    python_executable=sys.executable,
    gpu_transport="cuda_ipc",
) as tess:
    ...
```

`gpu_transport` can also be set through `runtime_config={"gpu_transport": "cuda_ipc"}`.
If both are given, the keyword argument wins.

A Tesseract created with `gpu_transport="cuda_ipc"` asks for the transport on
every call. To check that it works between the two processes, call
`tess.resolve_gpu_transport()`, which exchanges a small test array with the
server and raises a `RuntimeError` with the reason if the exchange fails.
Tesseract-JAX and Tesseract-Torch run this check before their first call with
GPU arrays.

### From the command line

`tesseract serve` enables the transport through the `TESSERACT_GPU_TRANSPORT`
environment variable:

```bash
$ tesseract serve my-gpu-tesseract --gpus all -e TESSERACT_GPU_TRANSPORT=cuda_ipc
```

As with the SDK, the container is started with `--ipc=host`. Enabling the
transport doesn't make the server use it for every request. Each request asks for
it, as `Tesseract.from_url` clients and [raw HTTP](#raw-http) clients do below.
A Tesseract serving the GPU transport must run with a single worker (the
default), and asking for more raises an error.

## Call it

### Python SDK

When the inputs you pass to `apply` (or any other endpoint) live on the GPU, the
client sends them by handle. GPU outputs come back as device arrays that expose
`__cuda_array_interface__` and DLPack, so any framework can adopt them without a
copy:

```python
import cupy as cp

a = cp.arange(10_000_000, dtype=cp.float32)
b = cp.ones_like(a)

with Tesseract.from_image(
    "my-gpu-tesseract", gpus=["all"], gpu_transport="cuda_ipc"
) as tess:
    out = tess.apply({"a": a, "b": b, "s": 3.0})["result"]

result = cp.asarray(out)      # stays on the GPU
host = out.copy_to_host()     # explicit copy to a NumPy array
```

For a Tesseract that someone else serves, `Tesseract.from_url` takes no transport
argument. Ask the client which transport works, then request it with
`with_encoding`:

```python
tess = Tesseract.from_url("http://localhost:8000")
transport = tess.resolve_gpu_transport()  # "cuda_ipc" if it works from here, else "none"
out = tess.with_encoding(gpu_transport=transport).apply({"a": a, "b": b})
```

`resolve_gpu_transport` returns `"none"` with a warning when the server offers
`cuda_ipc` but it doesn't work from your process, for example because the server
runs on another machine.

### Tesseract-JAX

`apply_tesseract` uses the transport of the Tesseract you pass it. Under
`jax.jit` on a CUDA backend, the call lowers to a native path that hands XLA's
device buffers to the Tesseract by handle and copies the results into XLA's
output buffers on the device. Gradients work as usual, since the VJP and JVP
calls go through the same path:

```python
import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

with Tesseract.from_image(
    "my-gpu-tesseract", gpus=["all"], gpu_transport="cuda_ipc"
) as tess:

    def loss(a, b):
        out = apply_tesseract(tess, {"a": a, "b": b})
        return jnp.sum(out["result"])

    grad_a = jax.jit(jax.grad(loss))(a, b)
```

A Tesseract that asks for no transport, such as one from `Tesseract.from_url`,
uses `cuda_ipc` if the server offers it and it works from your process, and
copies through the host otherwise. To always copy through the host, pass
`tess.with_encoding(gpu_transport="none")`.

The native path is a compiled extension that ships with Tesseract-JAX, and it is
used only when the call is compiled for a CUDA device. If the extension is
unavailable, a Tesseract that asks for `cuda_ipc` raises an error instead of
silently falling back to the host round trip, and one that asks for no transport
falls back with a warning.

```{warning}
If JAX itself runs on the CPU, the GPU transport has no effect and the call takes
the host round trip without an error. This happens with a CPU-only jaxlib, with
`JAX_PLATFORMS=cpu`, or when JAX fails to load its CUDA plugin, in which case it
falls back to the CPU with only a warning at startup. Check that
`jax.default_backend()` returns `"gpu"` before relying on the GPU transport.
```

### Tesseract-Torch

`apply_tesseract` from Tesseract-Torch picks the transport the same way as
Tesseract-JAX. CUDA tensors are sent by handle, GPU outputs come back as CUDA
tensors on their device, and autograd works as usual:

```python
import torch
from tesseract_torch import apply_tesseract

a = torch.randn(1_000_000, device="cuda", requires_grad=True)
b = torch.randn(1_000_000, device="cuda")

with Tesseract.from_image(
    "my-gpu-tesseract", gpus=["all"], gpu_transport="cuda_ipc"
) as tess:
    out = apply_tesseract(tess, {"a": a, "b": b})
    out["result"].sum().backward()
```

Calls with only CPU tensors always copy through the host, and anything that comes
through the host arrives on the CPU. As with Tesseract-JAX, pass
`tess.with_encoding(gpu_transport="none")` to always copy through the host.

(raw-http)=

### Raw HTTP

A client that speaks HTTP directly asks for the transport per request with a
`gpu_transport` parameter on the `Accept` header, next to the usual CPU array
encoding:

```
Accept: application/json+base64; gpu_transport=cuda_ipc
```

Without that parameter, or with `gpu_transport=none`, the response carries host
copies even when the server has the transport enabled. A server without the
transport enabled rejects `cuda_ipc` inputs with a 422 error.

The server keeps the GPU memory behind a response's handles until the client says
it has copied the arrays out. The SDK does this for you. A raw HTTP client sends
the `Tesseract-Exports-Done` header on a later request, naming the IDs that
earlier responses returned in their `Tesseract-Exports` header, and exports nobody
claims are released after 5 minutes. See the
{doc}`array encodings reference </content/reference/array-encodings>` for the
details. Resolving a `cuda_ipc` handle requires calling into the CUDA runtime, so
in practice this is for clients that already work with CUDA.

(two-jax-processes)=

## Two JAX processes on one GPU

By default, each JAX process preallocates 75% of the GPU's memory when it first
uses the device. With a JAX client and a JAX Tesseract on the same GPU, the
second process then runs out of memory. Disable preallocation, or lower the
fraction, before JAX initializes in either process:

```python
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# or: os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
```

A Tesseract started with `from_source` inherits the client's environment
variables, so setting the variable in the client before serving covers both
processes. For a container, pass it with `environment=` (or `-e` on the command
line).

## Limitations

- **A single worker.** A Tesseract serving the GPU transport runs one worker
  process, which can serve concurrent requests.
- **Full synchronization around each call.** Tesseract-JAX synchronizes the CUDA
  stream before and after each dispatch, so GPU work doesn't overlap with a
  Tesseract call.
- **Some copies remain.** Each side copies received arrays into memory it owns,
  and memory from CUDA's virtual memory management API (JAX's default
  allocator, for example) is copied into an exportable buffer before export.
  These copies run on the device at memory bandwidth, so they are much cheaper
  than a round trip through the host. To avoid allocating these buffers on every
  call, each process keeps idle ones for reuse, up to a quarter of the GPU's
  memory by default (set `TESSERACT_CUDA_IPC_POOL_FRACTION` to change it).
- **A fixed cost per call.** Each call still pays for an HTTP request and
  validation, and for exchanging handles on both sides, several milliseconds in
  total. For small arrays, below about 30,000 float32 elements in our measurements, the host round trip can be faster. See
  {doc}`/content/concepts/performance`.

## Troubleshooting

**`gpu_transport='cuda_ipc' requires GPU access, but no GPUs were requested`.**
Pass `gpus=["all"]` (or specific GPU IDs) to `from_image`, or `--gpus all` to
`tesseract serve`.

**`gpu_transport='cuda_ipc' was requested, but it does not work between this
process and the Tesseract`.** The error names the reason. Common causes are a
Tesseract on another machine and a container started without access to the same
GPU.

**`the native GPU FFI shim is unavailable`.** Your Tesseract-JAX install lacks
its compiled extension. Reinstall a wheel for your platform, or build from source
with a C++ compiler available.

**The GPU transport is enabled, but calls are no faster.** Check that your
arrays actually live on the GPU on both sides. Common causes are inputs created
on the CPU, a Tesseract that computes with NumPy, and adopting inputs with
`np.asarray`. Setting `TESSERACT_FORBID_DEVICE_HOST_COPY=1` in the client makes
the SDK and a `from_source` Tesseract raise on any implicit copy of a GPU array to
the host. With Tesseract-JAX, first check that `jax.default_backend()` returns
`"gpu"` in the calling process. If JAX runs on the CPU, the native path is never
used and nothing raises. Once JAX runs on the GPU, setting
`TESSERACT_JAX_DEBUG_CHECK_DEVICE_PTRS=1` makes any host pointer at the native
boundary raise an error, which pinpoints where an array left the device.

**Two JAX processes, and one fails with an out-of-memory error at startup.** See
[Two JAX processes on one GPU](#two-jax-processes).
