---
og:title: "Keep GPU arrays on the device with the cuda_ipc transport"
og:description: "Exchange GPU arrays with a served Tesseract by CUDA IPC handle instead of copying them through host memory, from the Python SDK, Tesseract-JAX, Tesseract-Torch, or raw HTTP."
---

<!--
TODO before release:
- Fill in the minimum versions below once tesseract-core (with #669 and #781),
  tesseract-jax 0.5.0, and tesseract-torch 0.2.0 are released.
- Verify on a GPU box: a JAX Tesseract adopting handle-backed inputs with
  jnp.from_dlpack (the Tesseract-JAX GPU tests serve a CuPy Tesseract only).
- Verify on a GPU box: whether Tesseract-JAX/Torch with gpu_transport work against
  a client from Tesseract.from_url (from_url takes no gpu_transport argument).
-->

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
  are meaningless across hosts or devices.
- **A shared IPC namespace.** When the Tesseract runs in a container, it needs
  `--ipc=host`. The SDK adds this for you (see below).
- **GPU arrays on both sides.** The client passes arrays that live on a CUDA
  device, and the Tesseract returns arrays that live on the device. CPU arrays
  are unaffected.
- **Recent versions.** tesseract-core **TODO**, tesseract-jax **TODO**, and
  tesseract-torch **TODO** or later.

## Write a Tesseract that returns GPU arrays

The runtime exports any output array that lives in GPU memory, whether it
exposes `__cuda_array_interface__` (CuPy, PyTorch, Numba) or DLPack on a CUDA
device (JAX). Inputs that arrive by handle reach your `apply` function as
framework-agnostic device arrays that expose both protocols, so adopt them with
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
`torch.from_dlpack(x)`. With JAX, use `jnp.from_dlpack(x)`. Avoid `np.asarray`
and anything else that goes through `__array__`, since the device arrays support
it for convenience by copying to host.

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
subprocess that shares your environment, including its CUDA libraries:

```python
with Tesseract.from_source("path/to/tesseract_api.py", gpu_transport="cuda_ipc") as tess:
    ...
```

`gpu_transport` can also be set through `runtime_config={"gpu_transport": "cuda_ipc"}`.
If both are given, the keyword argument wins.

### From the command line

`tesseract serve` enables the transport through the `TESSERACT_GPU_TRANSPORT`
environment variable:

```bash
$ tesseract serve my-gpu-tesseract --gpus all -e TESSERACT_GPU_TRANSPORT=cuda_ipc
```

As with the SDK, the container is started with `--ipc=host`. Outputs then leave
the Tesseract by handle unless a request asks otherwise (see [Raw HTTP](#raw-http)
below).

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

### Tesseract-JAX

Pass `gpu_transport="cuda_ipc"` to `apply_tesseract`. Under `jax.jit` on a CUDA
backend, the call then lowers to a native path that hands XLA's device buffers
to the Tesseract by handle and copies the results into XLA's output buffers on
the device. Gradients work as usual, since the VJP and JVP calls go through the
same path:

```python
import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

with Tesseract.from_image(
    "my-gpu-tesseract", gpus=["all"], gpu_transport="cuda_ipc"
) as tess:

    def loss(a, b):
        out = apply_tesseract(tess, {"a": a, "b": b}, gpu_transport="cuda_ipc")
        return jnp.sum(out["result"])

    grad_a = jax.jit(jax.grad(loss))(a, b)
```

The native path is a compiled extension that ships with Tesseract-JAX. If it is
unavailable, or JAX sees no CUDA device, a call that asks for the GPU transport
raises an error instead of silently falling back to the host round trip. Omit
`gpu_transport` to use the host path explicitly.

### Tesseract-Torch

Pass `gpu_transport="cuda_ipc"` to `apply_tesseract`. CUDA tensors are sent by
handle and GPU outputs are adopted back as CUDA tensors through DLPack, with
autograd working as usual:

```python
import torch
from tesseract_torch import apply_tesseract

a = torch.randn(1_000_000, device="cuda", requires_grad=True)
b = torch.randn(1_000_000, device="cuda")

with Tesseract.from_image(
    "my-gpu-tesseract", gpus=["all"], gpu_transport="cuda_ipc"
) as tess:
    out = apply_tesseract(tess, {"a": a, "b": b}, gpu_transport="cuda_ipc")
    out["result"].sum().backward()
```

Non-contiguous tensors are made contiguous before they are sent.

(raw-http)=

### Raw HTTP

A client that speaks HTTP directly selects the transport per request with a
`gpu_transport` parameter on the `Accept` header, next to the usual CPU array
encoding:

```
Accept: application/json+base64; gpu_transport=cuda_ipc
```

When the header names no transport, the served Tesseract's `gpu_transport`
setting applies, and `gpu_transport=none` asks for the host round trip even when
the transport is enabled. Resolving a `cuda_ipc` handle requires calling into the
CUDA runtime, so in practice this is for clients that already work with CUDA.

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

A Tesseract started with `from_source` inherits the client's environment, so
setting the variable in the client before serving covers both processes. For a
container, pass it with `environment=` (or `-e` on the command line).

## Limitations

- **One request at a time.** The server keeps the buffers it exported alive until
  the next request arrives, so a Tesseract serving the GPU transport must process
  requests serially. Serve it with a single worker (the default) and don't send
  overlapping requests.
- **C-contiguous arrays with matching dtypes.** The handle refers to a flat byte
  range with no strides, and nothing casts arrays on the device for you.
- **Full synchronization around each call.** Tesseract-JAX synchronizes the CUDA
  stream before and after each dispatch, so GPU work doesn't overlap with a
  Tesseract call.
- **Some copies remain.** Each side copies received arrays into memory it owns,
  and memory from CUDA's virtual memory management API (JAX's default
  allocator, for example) is copied into an exportable buffer before export.
  These copies run on the device at memory bandwidth, so they are much cheaper
  than a round trip through the host.
- **A fixed cost per call.** Each call still pays for an HTTP request and
  validation, a few milliseconds in total. For small arrays, the host round trip
  can be faster. See {doc}`/content/concepts/performance`.

## Troubleshooting

**`gpu_transport='cuda_ipc' requires GPU access, but no GPUs were requested`.**
Pass `gpus=["all"]` (or specific GPU IDs) to `from_image`, or `--gpus all` to
`tesseract serve`.

**`the native GPU FFI shim is unavailable`.** Your Tesseract-JAX install lacks
its compiled extension. Reinstall a wheel for your platform, or build from source
with a C++ compiler available.

**The GPU transport is enabled, but calls are no faster.** Check that your
arrays actually live on the GPU on both sides. Common causes are inputs created
on the CPU, a Tesseract that computes with NumPy, and adopting inputs with
`np.asarray`. With Tesseract-JAX, setting `TESSERACT_JAX_DEBUG_CHECK_DEVICE_PTRS=1`
makes any host pointer at the native boundary raise an error, which pinpoints
where an array left the device.

**Two JAX processes, and one fails with an out-of-memory error at startup.** See
[Two JAX processes on one GPU](#two-jax-processes).
