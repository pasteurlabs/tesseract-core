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

Each call still pays a fixed cost of a few milliseconds for the HTTP request,
validation, and handle exchange. For small arrays, below about 30,000 float32
elements in our measurements, the host round trip can be faster (see
{doc}`/content/concepts/performance`).

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
that arrive by handle reach your endpoints as `IpcDeviceArray` objects, which
expose both `__cuda_array_interface__` and DLPack, so adopt them with your
framework before computing. The [CuPy](https://github.com/pasteurlabs/tesseract-core/tree/main/examples/_gpu_cupy),
[JAX](https://github.com/pasteurlabs/tesseract-core/tree/main/examples/_gpu_jax),
and [PyTorch](https://github.com/pasteurlabs/tesseract-core/tree/main/examples/_gpu_torch)
examples compute `s * a + b` for two float32 vectors `a` and `b`:

::::{tab-set}
:::{tab-item} CuPy
:sync: cupy

```{literalinclude} ../../../examples/_gpu_cupy/tesseract_api.py
:pyobject: apply
:language: python
```

`cp.asarray` keeps device arrays on the GPU and moves host arrays onto it.

:::
:::{tab-item} JAX
:sync: jax

```{literalinclude} ../../../examples/_gpu_jax/tesseract_api.py
:pyobject: apply
:language: python
```

`as_jax_array` adopts device arrays through DLPack and moves host arrays onto the
default device. Tesseracts built on the JAX recipe (`tesseract init --recipe jax`)
use it for all their inputs already.

:::
:::{tab-item} PyTorch
:sync: torch

```{literalinclude} ../../../examples/_gpu_torch/tesseract_api.py
:pyobject: apply
:language: python
```

`torch.as_tensor(x, device="cuda")` and `torch.from_dlpack(x)` both adopt device
arrays without a copy.

:::
::::

Avoid `np.asarray` and anything else that goes through `__array__`. The device
arrays support it for convenience, by copying to host.

```{warning}
Input dtypes must match the schema exactly. The host encodings cast an input to
the dtype its schema declares, but `cuda_ipc` does not cast on the device and
rejects a mismatched input instead.
```

(serve-with-gpu-transport)=

## Serve with GPU transport

Enable the transport where the Tesseract is served:

::::{tab-set}
:::{tab-item} from_image
:sync: from-image

```python
from tesseract_core import Tesseract

tess = Tesseract.from_image(
    "my-gpu-tesseract", gpus=["all"], gpu_transport="cuda_ipc"
)
with tess:
    ...
```

The SDK runs the container with `--ipc=host` whenever the GPU transport is
enabled, and raises a `ValueError` if no GPUs were requested.

:::
:::{tab-item} from_source
:sync: from-source

```python
import os
import sys

# A JAX client and a JAX Tesseract share the GPU, so neither should claim 75% of
# its memory at startup. The Tesseract inherits this variable from the client.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from tesseract_core import Tesseract

tess = Tesseract.from_source(
    "path/to/tesseract_api.py",
    python_executable=sys.executable,
    gpu_transport="cuda_ipc",
)
with tess:
    ...
```

`from_source` serves the Tesseract in a subprocess, without a container. By
default, that subprocess runs in an environment built from the Tesseract's own
requirements. `python_executable=sys.executable` shares your environment
instead, including its CUDA-enabled libraries.

:::
:::{tab-item} tesseract serve
:sync: cli

```bash
$ tesseract serve my-gpu-tesseract --gpus all -e TESSERACT_GPU_TRANSPORT=cuda_ipc
```

As with the SDK, the container is started with `--ipc=host`. Enabling the
transport doesn't make the server use it for every request. Each request asks for
it, as the SDK and HTTP clients [below](#call-it) do.

:::
::::

`gpu_transport` can also be set through `runtime_config={"gpu_transport": "cuda_ipc"}`.
If both are given, the keyword argument wins.

With a JAX client and a JAX Tesseract on one GPU, disable preallocation (or lower
`XLA_PYTHON_CLIENT_MEM_FRACTION`) in both processes before JAX initializes, as
the `from_source` example does. Otherwise the second process runs out of memory
at startup. For a container, pass the variable with `environment=` (or `-e` on
the command line).

```{warning}
A Tesseract serving the GPU transport must run with a single worker (the
default), and asking for more raises an error. That worker can serve concurrent
requests.
```

A Tesseract created with `gpu_transport="cuda_ipc"` asks for the transport on
every call. To check that it works between the two processes, call
`tess.resolve_gpu_transport()`, which exchanges a small test array with the
server and raises a `RuntimeError` with the reason if the exchange fails.
Tesseract-JAX and Tesseract-Torch run this check before their first call with
GPU arrays.

(call-it)=

## Call it

::::{tab-set}
:::{tab-item} Python SDK
:sync: python

When the inputs you pass to `apply` (or any other endpoint) live on the GPU, the
client sends them by handle. GPU outputs come back as `IpcDeviceArray` objects,
which any framework can adopt without a copy:

```python
import cupy as cp

a = cp.arange(10_000_000, dtype=cp.float32)
b = cp.ones_like(a)

tess = Tesseract.from_image(
    "my-gpu-tesseract", gpus=["all"], gpu_transport="cuda_ipc"
)
with tess:
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

:::
:::{tab-item} Tesseract-JAX
:sync: jax

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

tess = Tesseract.from_image(
    "my-gpu-tesseract", gpus=["all"], gpu_transport="cuda_ipc"
)
with tess:

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

:::
:::{tab-item} Tesseract-Torch
:sync: torch

`apply_tesseract` from Tesseract-Torch picks the transport the same way as
Tesseract-JAX. CUDA tensors are sent by handle, GPU outputs come back as CUDA
tensors on their device, and autograd works as usual:

```python
import torch
from tesseract_torch import apply_tesseract

a = torch.randn(1_000_000, device="cuda", requires_grad=True)
b = torch.randn(1_000_000, device="cuda")

tess = Tesseract.from_image(
    "my-gpu-tesseract", gpus=["all"], gpu_transport="cuda_ipc"
)
with tess:
    out = apply_tesseract(tess, {"a": a, "b": b})
    out["result"].sum().backward()
```

Calls with only CPU tensors always copy through the host, and anything that comes
through the host arrives on the CPU. As with Tesseract-JAX, pass
`tess.with_encoding(gpu_transport="none")` to always copy through the host.

:::
:::{tab-item} HTTP
:sync: http

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

:::
::::

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

**Two JAX processes, and one fails with an out-of-memory error at startup.**
Each JAX process preallocates 75% of the GPU's memory by default. Set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` in both, as described under
{ref}`serve-with-gpu-transport`.
