---
orphan: true
og:title: "Tesseract is now GPU-native"
og:description: "A training loop can now call a GPU solver running in another process, forward and backward, without its arrays ever leaving the GPU. Here is what that takes, what it buys, and how it works."
# TODO(publish): set blog_date to the day this goes out.
blog_date: "2026-10-01"
blog_author: "@dionhaefner"
blog_title: "Tesseract is now GPU-native"
blog_description: "A training loop can now call a GPU solver running in another process, forward and backward, without its arrays ever leaving the GPU. Here is what that takes, what it buys, and how it works."
---

# Tesseract is now GPU-native

% TODO(publish): confirm with the Multi-Agent-DPC team (Pietro Zanotta, Dibakar Roy, Honghui Zheng) that they're happy with how their project and its slowdown are described.

The winning entry of our first hackathon, [Multi-Agent-DPC](https://github.com/SOLARIS-JHU/Multi-Agent-DPC) by Pietro Zanotta, Dibakar Roy, and Honghui Zheng, teaches a swarm of mobile heat sources to shape a temperature field. A small policy network decides how each agent moves, and it learns by backpropagating through 300 steps of a differentiable PDE solver. The team wrapped the solver as a [Tesseract](https://github.com/pasteurlabs/tesseract-core) and called it from their JAX training loop through [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), which makes a Tesseract behave like any other differentiable JAX function.

```{figure} ../static/blog/hackathon-multi-agent-dpc.png
:alt: "Diagram of Multi-Agent Differentiable Predictive Control. Each agent's initial condition, tracking reference, and constraints feed a DeepONet neural policy written in JAX. The policies drive an auto-differentiable PDE solver wrapped as a Tesseract, whose output feeds a differentiable loss for reference tracking and constraint penalties. Forward arrows run from the policies through the Tesseract to the loss, and backward arrows carry gradients from the loss through the Tesseract back to the policies."

A neural policy and a PDE solver trained as one differentiable program, with the solver behind a Tesseract. Figure from the [Multi-Agent-DPC repository](https://github.com/SOLARIS-JHU/Multi-Agent-DPC).
```

It worked, but far too slowly to train with, so the team's training script calls the solver directly with the Tesseract path switched off. They published a [branch that reproduces the slowdown](https://github.com/SOLARIS-JHU/Multi-Agent-DPC/tree/repro-performance-issue), and that is where this work started. Two of the causes had simple fixes. Tesseract-JAX called the Tesseract once per rollout, which it can now batch into a single call, and the solver's container installed the CPU build of JAX. The third went deeper. Every array that crossed into or out of the Tesseract left the GPU, was encoded as text into a JSON body, crossed HTTP, and was decoded and copied back onto the GPU on the other side. A training step that never needed to leave the GPU did so twice, once for the forward pass and once for the gradient.

Tesseract-JAX and [Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch) can now hand GPU arrays to a Tesseract in another process without them ever leaving the GPU. Only a 64-byte handle crosses the process boundary, and the data moves within the memory of the GPU that both processes share. With all three fixes, plus a cache for the solver's forward pass, the team's training step runs within 20% of importing the solver directly. For the larger arrays common in 3D simulation, the transport alone is two orders of magnitude faster than the trip through host memory.

```{figure} ../static/blog/gpu-host-vs-ipc.png
:alt: "Two rows comparing how a GPU array reaches a Tesseract in another process. In the top row, labeled host round trip, the array is copied from the training process's GPU to host memory, base64-encoded into JSON, sent over HTTP across the process boundary, decoded, and copied back onto the GPU for the solver process. In the bottom row, labeled cuda_ipc, the training process exports a 64-byte handle that crosses the process boundary inside the JSON request, and the solver process maps it, while the array data moves device-to-device on the one GPU both processes share."

Two ways a GPU array reaches a Tesseract in another process. Through the host, every byte leaves the GPU and is encoded as text. With `cuda_ipc`, only a handle crosses HTTP, and outputs return the same way.
```

## Why put a process boundary in a training loop

If you can import a component, import it. The Multi-Agent-DPC team came to the same conclusion, and [CINOC](https://github.com/SOLARIS-JHU/CINOC), the ICML 2026 paper that grew out of the project, imports its solvers directly. We still use their setup in this post, because with JAX on both sides, any difference in speed comes from the boundary alone. Read their solver as a stand-in for one you can't import.

Often you can't. The solver comes from another team, is pinned to other dependencies, or is written in a framework that doesn't mix with yours. Even JAX within JAX can clash. `jax_enable_x64` applies to a whole process, so a solver that needs float64 gradients forces it on any float32 model sharing that process. Tesseract puts such a component in its own process or container and still carries gradients across. Our guide on [when to use Tesseract](../content/concepts/when-to-use.md) goes through the trade-offs.

% TODO(publish): the when-to-use link resolves only after #821 merges.

On GPUs, this comes up wherever training runs through a differentiable solver:

- **Solver in the loop.** A network learns to correct a coarse simulation, and its loss is computed after unrolling the solver for many steps.
- **Learned closures.** A network supplies a term the simulation can't resolve, such as subgrid-scale stresses in turbulence, and is trained through the simulation that hosts it.
- **Control and reinforcement learning.** A policy is trained against a differentiable simulator, as in Multi-Agent-DPC.
- **Served surrogates.** A trained surrogate runs as a service on the GPU, and an optimizer or pipeline next to it calls it over and over.

In all of these, both halves live on the GPU, and training crosses the boundary thousands of times. Until now, every crossing went through host memory, and the arrays can be large. A single 256³ float32 field takes up 67 MB. Sending one through the host to a Tesseract and getting one back takes over half a second, which can be longer than the rest of the training step. With the GPU transport, it takes about 6 ms.

## Turning it on

`Tesseract.from_source` serves a `tesseract_api.py` in a subprocess. Pointed at your own interpreter, it gives the solver the same CUDA-enabled JAX as your training loop, and `gpu_transport="cuda_ipc"` asks for the GPU transport. Tesseract-JAX picks it up from there:

```python
import os
import sys

# Two JAX processes share one GPU, so neither should claim 75% of it at startup.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from jax.flatten_util import ravel_pytree
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

solver = Tesseract.from_source(
    "tesseracts/solverHeat_decentralized/tesseract_api.py",
    python_executable=sys.executable,
    gpu_transport="cuda_ipc",
)


def loss(params, z_init, xi_init, z_target):
    inputs = {
        "z_init": z_init,
        "xi_init": xi_init,
        "z_target": z_target,
        "flat_params": ravel_pytree(params)[0],
    }
    traj = apply_tesseract(solver, inputs, vmap_method="broadcast_all")
    return tracking_error(traj["z_trajectory"], z_target) + penalties(traj)


def batch_loss(params, z_init, xi_init, z_target):
    losses = jax.vmap(loss, in_axes=(None, 0, 0, 0))(params, z_init, xi_init, z_target)
    return losses.mean()


with solver:
    grads = jax.jit(jax.grad(batch_loss))(params, z_init, xi_init, z_target)
```

`vmap_method="broadcast_all"` sends the whole batch of rollouts to the Tesseract as one call, which the solver supports by accepting an optional leading batch dimension. A containerized Tesseract works the same way through `Tesseract.from_image(..., gpus=["all"], gpu_transport="cuda_ipc")`, and Tesseract-Torch picks up the transport just like Tesseract-JAX. If the two processes can't share GPU memory, the first call raises with the reason instead of quietly falling back to the host. The [how-to guide](../content/how-to/gpu-transport.md) covers the details.

## What it buys

On an NVIDIA A100, here is one training step of Multi-Agent-DPC, a batch of 32 rollouts forward and backward, as it went from the team's setup to the GPU path ([benchmark script](https://github.com/dionhaefner/Multi-Agent-DPC/blob/dca8d074eacc9ea97a1cd43740f115340c437dff/examples/heat1d/decentralized/benchmark.py)):

| Setup                                                  | Time per training step |
| ------------------------------------------------------ | ---------------------: |
| The team's setup, on current Tesseract releases        |                  1.5 s |
| Batched, solver on the GPU, arrays through host memory |                 261 ms |
| Same, with `cuda_ipc` and the solver's VJP cache       |                 127 ms |
| Solver imported directly, one process                  |                 106 ms |

The last factor of two comes from two changes of similar size. The GPU transport removes the trip through host memory, and the {py:func}`VJP cache <tesseract_core.runtime.experimental.set_jax_vjp_cache_size>`, an experimental option in Tesseract's JAX recipe, keeps the residuals of the forward pass so the gradient call doesn't rerun all 300 steps. What remains, about 20 ms per step, is mostly the fixed cost of two HTTP requests and the handle exchange.

A Multi-Agent-DPC call moves about 4.8 MB, which is modest. How much the transport matters depends mostly on how many bytes cross the boundary, which the figure below isolates with a Tesseract that only doubles its input.

```{figure} ../static/blog/gpu-ipc-benchmark.png
:alt: "Log-log plot of median time per call against array size, from 1,000 to 100 million float32 elements. The host round trip rises from about 3 ms to about 3.2 s. A dashed line for the host path with json+binref files on /dev/shm starts at about 4 ms and rises to about 1 s. cuda_ipc stays near 4 ms up to a few million elements and rises to about 17 ms at 100 million, 190 times faster than the host round trip. The host round trip is faster below about 28,000 elements. A dotted line at 600,000 elements marks a call that moves as many bytes as one Multi-Agent-DPC call. The same computation in-process takes between 0.13 and 0.6 ms."

Median time of one jitted `apply_tesseract` call to a Tesseract that doubles a float32 vector, on an NVIDIA A100 80GB. The dotted line marks a call that moves as many bytes as one Multi-Agent-DPC call. The {download}`benchmark script </downloads/gpu-ipc-benchmark.py>` reproduces the figure.
```

Below about 100 kB per call, the trip through host memory is as fast or faster, because the GPU transport has a slightly higher fixed cost. Above that, the host path grows by about 8 ms per megabyte, spent on copies and text encoding on the CPU, while `cuda_ipc` stays nearly flat. At 100 million values, or 400 MB, the GPU transport is 190 times faster than Tesseract's default host path, and still almost 60 times faster than its fastest one, which exchanges raw binary files on shared memory. Calling the same Tesseract from PyTorch gives the same picture.

## How it works

For the curious, here is what happens when `jax.grad` meets a Tesseract call on the GPU.

```{figure} ../static/blog/gpu-call-stack.png
:alt: "Diagram of one gradient call. In the training process, jax.grad, the Tesseract-JAX primitive, an XLA FFI custom call, a C++ shim, and the Tesseract HTTP client are stacked top to bottom. The client exchanges a JSON request and response carrying 64-byte handles with the Tesseract runtime in the solver process, which calls the solver's VJP in tesseract_api.py. Underneath both processes, one GPU holds XLA's buffers and the solver's buffers, with device-to-device copies between them."

One gradient call, top to bottom. Only JSON with 64-byte handles crosses the process boundary, while the array data stays in the memory of the GPU that both processes share.
```

`jax.grad` asks Tesseract-JAX for a vector-Jacobian product (VJP). Tesseract-JAX answers every request, whether a forward evaluation, a VJP, or a Jacobian, by binding one JAX primitive that stands for a call to one of the Tesseract's endpoints, so nothing further down has code specific to gradients.

Under `jit`, XLA needs to know how to run that primitive. On the CPU it runs as a host callback into Python, and until now it ran that way on the GPU too. Host callbacks receive NumPy arrays, so XLA copied every input off the device before Python saw it, and no wire format downstream could undo that. JAX has no Python-level API for handing GPU buffers out of a compiled program, so the fix had to be native. On CUDA, Tesseract-JAX now lowers the primitive to an [XLA FFI](https://docs.jax.dev/en/latest/ffi.html) custom call into a small C++ shim. The shim receives raw device pointers from XLA, waits for XLA's stream so the inputs are ready, and calls back into Python with views that expose `__cuda_array_interface__`. It loads the CUDA runtime only when first used, so the same wheel installs and runs on machines without a GPU.

From there, the views go to the Tesseract's HTTP client. For each array, the client asks CUDA for an IPC handle, a 64-byte token that another process on the same machine can use to map the same allocation, and puts it in the JSON request where the encoded bytes would otherwise go. Here is one of the solver's 3.84 MB trajectory arrays both ways:

```json
{
  "object_type": "array",
  "shape": [32, 300, 100],
  "dtype": "float32",
  "data": {
    "buffer": "slPUPa5mfD5CBcs9OirIvgsQiz5TIAk+uPMkvv2EMj5I/t89C7e0PXWz...",
    "encoding": "base64"
  }
}
```

```json
{
  "object_type": "array",
  "shape": [32, 300, 100],
  "dtype": "float32",
  "data": {
    "buffer": "80c3660b-f7b0-48f9-8ad6-7f36c1e7e9bf:xAz7Rvl/rnXSpn869f++dNC9NWxplFo3qLs+qce5gyNA1eMWkRWGatgyGZfLq4e001iiEs3oW+hxjxR7Q+f2cg==:0:3840000",
    "encoding": "cuda_ipc"
  }
}
```

The base64 buffer goes on for another 5.1 million characters, while the `cuda_ipc` buffer is complete. It holds the GPU's UUID, the handle, and where the array sits within the allocation. Shapes and dtypes stay in the JSON, so the server still validates every request against its schema. The solver process maps the handles, copies the inputs into memory it owns, and computes the VJP. Its outputs travel back the same way, and the shim copies them into XLA's output buffers before returning control to XLA.

One wrinkle is that XLA allocates GPU memory in a way that IPC handles can't refer to, so the client first copies each array into a buffer it can export, and reuses those buffers from call to call. PyTorch's default allocator has no such problem, and its tensors already expose their device pointers to Python, so Tesseract-Torch reaches the same client without native code or the extra copy.

### What made it hard

None of this is a lot of code, but very little of it is obvious, and most mistakes don't crash:

- **XLA's memory can't be shared the obvious way.** The client first tries `cudaIpcGetMemHandle` on each array and falls back to the staging copy when that fails, which it always does on XLA's memory. CUDA records the failure as a process-wide error, and XLA's next kernel launch picks it up as its own. The first `jit`-compiled function ran fine, and every one compiled after it aborted, until we cleared the error right after the expected failure.
- **Two processes share one lifetime.** The server can't free a response's memory until the client has copied it out, and nobody may read a buffer before the GPU has finished writing it. Getting either wrong returns wrong numbers instead of errors. With eight client threads, 138 of 800 results came back holding another call's data, and a missing synchronization went unnoticed only because the HTTP round trip always outlasted the copy.
- **Slow is silent.** A fast path that quietly falls off the GPU still passes every test. Tesseract-JAX's GPU tests make the shim reject any pointer that isn't in device memory, so an accidental host round trip fails loudly. Profiling the training step found what no test could, such as a VJP cache that hashed its inputs by copying every GPU array back to the host on every call.

## Limits and what's next

The GPU transport is built on CUDA IPC, so the caller and the Tesseract must run on the same Linux machine and see the same GPU. It needs tesseract-core 1.15 on both sides, and tesseract-jax 0.5 or tesseract-torch 0.2 on the client. A Tesseract serving it runs a single worker, which can still serve concurrent requests, and the shim waits for the GPU before and after every call, which keeps it simple at the cost of overlap with other GPU work.

Next, we want to let other GPU work overlap with Tesseract calls, serve the transport from more than one worker, and skip the staging copy for JAX arrays, which [is in review](https://github.com/pasteurlabs/tesseract-core/pull/726). Beyond a single machine, the same idea extends to GPUs on other hosts, and that deserves a post of its own once we have measured it.

As a rule of thumb, turn the transport on when a GPU component and its caller share a machine and more than about 100 kB cross the boundary per call. Start with the [GPU transport how-to](../content/how-to/gpu-transport.md). If you have a GPU component that lives in another stack, or a training loop that bounces arrays through the CPU to reach one, we'd like to hear how this works for you on the [Forum](https://si-tesseract.discourse.group/). New to Tesseract? The [getting started guide](../content/tutorials/get-started.md) covers building and serving your first Tesseract.
