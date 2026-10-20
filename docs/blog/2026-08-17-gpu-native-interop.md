---
orphan: true
og:title: "Tesseract is now GPU-native"
og:description: "Tesseract-JAX and Tesseract-Torch can now call a component running in another process on the same GPU, forward and backward, without copying arrays through host memory. Here is why that matters for solver–ML hybrids, and the call stack that makes it work."
blog_date: "2026-10-01"
blog_author: "@dionhaefner"
blog_title: "Tesseract is now GPU-native"
blog_description: "Tesseract-JAX and Tesseract-Torch can now call a component running in another process on the same GPU, forward and backward, without copying arrays through host memory. Here is why that matters for solver–ML hybrids, and the call stack that makes it work."
---

# Tesseract is now GPU-native

The winning entry of our first hackathon, [Multi-Agent-DPC](https://github.com/SOLARIS-JHU/Multi-Agent-DPC) by Pietro Zanotta, Dibakar Roy, and Honghui Zheng, teaches a swarm of mobile heat sources to shape a temperature field. A small policy network decides how each agent moves, and it learns by backpropagating through 300 steps of a differentiable PDE solver. The team wrapped that solver as a [Tesseract](https://github.com/pasteurlabs/tesseract-core) and called it from their JAX training loop through [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), which makes a Tesseract behave like any other differentiable JAX function.

```{figure} ../static/blog/hackathon-multi-agent-dpc.png
:alt: "Diagram of Multi-Agent Differentiable Predictive Control. Each agent's initial condition, tracking reference, and constraints feed a DeepONet neural policy written in JAX. The policies drive an auto-differentiable PDE solver wrapped as a Tesseract, whose output feeds a differentiable loss for reference tracking and constraint penalties. Forward arrows run from the policies through the Tesseract to the loss, and backward arrows carry gradients from the loss through the Tesseract back to the policies."

A neural policy and a PDE solver trained as one differentiable program, with the solver behind a Tesseract. Figure from the [Multi-Agent-DPC repository](https://github.com/SOLARIS-JHU/Multi-Agent-DPC).
```

The training loss boils down to one Tesseract call. The policy's weights go in, the solver rolls out 300 steps with the policy choosing each agent's move at every step, and `jax.grad` carries the loss gradient back through the whole rollout into the weights:

```python
def loss(params, z_init, xi_init, z_target):
    traj = apply_tesseract(solver, {
        "z_init": z_init,
        "xi_init": xi_init,
        "z_target": z_target,
        "flat_params": ravel_pytree(params)[0],
    })
    return tracking_error(traj["z_trajectory"], z_target) + penalties(traj)

grads = jax.grad(loss)(params, z_init, xi_init, z_target)
```

It worked, but it was far too slow to train with, and their published training script calls the solver directly with the Tesseract path switched off. The blocker was the way each array crossed the boundary. It left the GPU for host memory, was base64-encoded into a JSON body, crossed HTTP, was decoded on the other side, and made the same trip back. A training step makes two such calls, the forward rollout and the vector-Jacobian product (VJP) on the way back. The rollout returns about 4.8 MB of trajectories, and the VJP sends as much back. With every array taking the host round trip anyway, there was little point in running the solver on the GPU, and its container installed the CPU build of JAX. The team published a [branch that reproduces the slowdown](https://github.com/SOLARIS-JHU/Multi-Agent-DPC/tree/repro-performance-issue), and that is where this work started.

Tesseract-JAX and [Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch) can now keep both calls on the GPU. The solver still runs in its own process, but only 64-byte memory handles cross the boundary. On an NVIDIA A100, the team's training step drops from 265 ms with arrays going through host memory to 126 ms, within 20% of importing the solver directly. Below, we look at why you would put a process boundary inside a GPU training loop at all, walk down the call stack that makes it work, and break down where the remaining time goes.

## Why put a process boundary in a training loop

If you can import a component, import it. Two JAX functions by the same authors can share an environment, and a process boundary between them only adds cost. The Multi-Agent-DPC team came to the same conclusion. [CINOC](https://github.com/SOLARIS-JHU/CINOC), the ICML 2026 paper that grew out of the project, imports its solvers directly. The boundary pays for itself when the component can't be rewritten, when you need gradients through it, when its dependencies, language runtime, global state, or autodiff framework clash with yours, and when each call does real work. Our guide on [when to use Tesseract](../content/concepts/when-to-use.md) goes through these conditions in detail.

Solver–ML hybrids are where those conditions most often meet a GPU. A neural network trained through a physics solver, a learned closure inside a simulation, and a surrogate standing in for part of a pipeline all pair two components that tend to come from different teams and different stacks. Both halves usually run on the GPU, their data (fields on grids, batches of trajectories) lives there already, and training crosses the boundary at every step, thousands of times. A round trip through host memory on every call is exactly the cost such a loop can least afford. Three entries in this year's hackathon show how real these boundaries are:

- [Prismo](https://github.com/benvial/prismo) optimizes the doping of a silicon photonic phase shifter. It couples a Julia drift-diffusion solver with a FEniCS eigenmode solver stuck on Python 3.10, all driven by `jax.grad` on Python 3.12.
- [δsnow17-sacsma](https://github.com/kasProg/dsnow17-sacsma) trains a neural network to predict the parameters of two Fortran hydrology models that NOAA uses for river forecasting, with gradients flowing through both.
- [Differentiable Silicon](https://github.com/hozaifa1/differentiable-silicon) optimizes device parameters through Synopsys Sentaurus, a closed-source semiconductor simulator on a separate machine, composed with a PyTorch model.

Even JAX within JAX sometimes needs two processes. A solver whose gradients need float64 has to switch on `jax_enable_x64`, and that flag applies to the whole process, including any float32 model sharing it. A solver pinned to a different JAX release can't share an interpreter with your training loop at all.

## One gradient, from top to bottom

```{figure} ../static/blog/gpu-call-stack.png
:alt: "Diagram of one gradient call. In the training process, jax.grad, the Tesseract-JAX primitive, an XLA FFI custom call, a C++ shim, and the Tesseract HTTP client are stacked top to bottom. The client exchanges a JSON request and response carrying 64-byte handles with the Tesseract runtime in the solver process, which calls the solver's VJP in tesseract_api.py. Underneath both processes, one GPU holds XLA's buffers and the solver's buffers, with device-to-device copies between them."

One VJP call, top to bottom. Only JSON with 64-byte handles crosses the process boundary, while the array data stays in the memory of the GPU that both processes share.
```

When `jax.grad` reaches the Tesseract call, it asks Tesseract-JAX for a VJP. Tesseract-JAX answers every derivative request the same way, by binding one JAX primitive that stands for a call to one of the Tesseract's endpoints. Forward evaluations, Jacobian-vector products, VJPs, and full Jacobians all go through it, so nothing below this point has code specific to gradients.

Under `jit`, XLA needs to know how to run that primitive. On the CPU it runs as a host callback into Python, and until now it ran that way on the GPU too. Host callbacks receive NumPy arrays, so XLA copies every input off the device before Python sees it, and no wire format downstream can undo that. JAX has no Python-level API for handing GPU buffers out of a compiled program, so the fix had to be native. On the `cuda` platform, Tesseract-JAX now lowers the primitive to an [XLA FFI](https://docs.jax.dev/en/latest/ffi.html) custom call into a small C++ shim.

The shim receives raw device pointers from XLA. It synchronizes XLA's stream so the inputs are ready, wraps each pointer in a minimal object that exposes `__cuda_array_interface__`, takes the GIL, and calls back into Python with those views. It loads the CUDA runtime with `dlopen` on first use and links against no CUDA library at build time, so the same wheel installs and runs on machines without a GPU.

In Python, the views go to the same dispatch function the CPU path uses, which passes them to the Tesseract's HTTP client. With the GPU transport enabled, the client doesn't encode any bytes. For each array it asks CUDA for an IPC handle, a 64-byte token that another process on the same machine can use to map the same allocation, and it puts that handle in the JSON request where the base64 would otherwise go. Here is one of the solver's 3.84 MB trajectory arrays both ways:

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

The base64 buffer goes on for another 5.1 million characters. The `cuda_ipc` buffer is complete: the GPU's UUID, the handle, and the array's offset and size within the allocation. Shapes and dtypes stay in the JSON, so the server still validates every request against its schema. One complication is that XLA allocates memory through CUDA's virtual memory management API, which legacy IPC handles can't refer to. The client therefore first copies each array into a plain `cudaMalloc` buffer it can export. Those staging buffers, and their handles, are reused from call to call.

The solver process maps the handles, copies the inputs into memory it owns, closes the mappings, and computes the VJP. Its outputs are XLA allocations too, and they travel back the same way. Back in the training process, the client maps the returned handles, and the shim copies the results into XLA's output buffers, synchronizes once more, and returns control to XLA.

Tesseract-Torch takes a shorter route to the same client. PyTorch hands its autograd functions CUDA tensors, which already expose their device pointers to Python, so no native code is involved on that side.

### What made it hard

None of this is a lot of code, but very little of it is obvious, and most mistakes don't crash:

- **XLA's memory can't be shared the obvious way.** The staging copy starts with a `cudaIpcGetMemHandle` call that is expected to fail on XLA's memory. CUDA records that failure as a process-wide error, and XLA's next kernel launch picks it up as its own. The first `jit`-compiled function ran fine, and every one compiled after it aborted, until we cleared the error right after the expected failure.
- **Two processes share one lifetime.** The server can't free a response's memory until the client has copied it out, and nobody may read a buffer before the GPU has finished writing it. Getting either wrong returns wrong numbers instead of errors. With eight client threads, 138 of 800 results came back holding another call's data, and a missing synchronization went unnoticed only because the HTTP round trip always outlasted the copy.
- **Slow is silent.** A fast path that quietly falls off the GPU still passes every test. Tesseract-JAX's GPU tests make the shim reject any pointer that isn't in device memory, so an accidental host round trip fails loudly. Profiling the training step found what no test could, such as a VJP cache that hashed its inputs by copying every GPU array back to the host on every call.

## The same training step, on the GPU

Running the solver in its own process doesn't require a container. `Tesseract.from_source` serves a `tesseract_api.py` from a subprocess, and pointing it at the training loop's interpreter gives the solver the same CUDA-enabled JAX. Asking for the GPU transport takes one keyword where the Tesseract is served, and Tesseract-JAX picks it up automatically (the [how-to guide](../content/how-to/gpu-transport.md) has the details):

```python
import os
import sys

# Two JAX processes share one GPU, so neither should claim 75% of it at startup.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax.flatten_util import ravel_pytree
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

solver = Tesseract.from_source(
    "tesseracts/solverHeat_decentralized/tesseract_api.py",
    python_executable=sys.executable,
    gpu_transport="cuda_ipc",
)

with solver:

    def rollout(params, z_init, xi_init, z_target):
        flat_params, _ = ravel_pytree(params)
        inputs = {
            "z_init": z_init,
            "xi_init": xi_init,
            "z_target": z_target,
            "flat_params": flat_params,
            "t_steps": 300,
        }
        return apply_tesseract(solver, inputs, vmap_method="broadcast_all")

    # The rest is the team's training loop, unchanged:
    # jax.jit(jax.value_and_grad(mean over jax.vmap(loss))), then an Adam update.
```

The only change the Tesseract itself needed is that its arrays accept an optional leading batch dimension, so `broadcast_all` can send all 32 rollouts in one call. We also took the chance to [port it to Tesseract's JAX recipe](https://github.com/dionhaefner/Multi-Agent-DPC/blob/dca8d074eacc9ea97a1cd43740f115340c437dff/tesseracts/solverHeat_decentralized/tesseract_api.py), which the last row of the table below relies on.

Asking for `gpu_transport="cuda_ipc"` also makes the setup fail loudly. Before its first call, Tesseract-JAX has the client exchange a small test array with the server over IPC, and if that fails, the call raises with the reason instead of falling back to the host path. That check can't help if JAX itself ends up on the CPU, for example because its CUDA plugin failed to load, so it is worth checking that `jax.default_backend()` returns `"gpu"` in both processes.

## Results

We timed one training step, a batch of 32 rollouts forward and backward plus the optimizer update, on an NVIDIA A100 80GB with CUDA 13 and JAX 0.11.2 ([benchmark script](https://github.com/dionhaefner/Multi-Agent-DPC/blob/dca8d074eacc9ea97a1cd43740f115340c437dff/examples/heat1d/decentralized/benchmark.py)). For comparison, the last column runs the same script with everything on the CPU of an Apple M1 Pro laptop. With JAX on both sides of the boundary, the rows differ only in how the solver is called, which is what makes this a clean test, even though it isn't a setup anyone should build on purpose.

| Setup                                                 | A100 GPU | M1 Pro CPU |
| ----------------------------------------------------- | -------: | ---------: |
| Solver imported directly, one process                 |   106 ms |     183 ms |
| Solver in its own process, arrays through host memory |   265 ms |     369 ms |
| Solver in its own process, `cuda_ipc`                 |   209 ms |          – |
| Same, with the solver's VJP cache                     |   126 ms |          – |

Removing the host round trip closes about a third of the gap to the single-process version on the GPU. Most of what the third row still pays for is compute rather than transport. To compute the VJP, the solver reruns the entire 300-step forward pass before it can go backward. The last row turns on an experimental option in Tesseract's JAX recipe, the {py:func}`VJP cache <tesseract_core.runtime.experimental.set_jax_vjp_cache_size>`, which keeps the residuals from the forward pass so the VJP call can reuse them, and together the two close almost 90% of the gap.

### Where the remaining time goes

What remains, about 20 ms per step, is almost all fixed cost. A Tesseract with the same inputs and outputs that computes nothing costs about 16 ms per step this way, against under a millisecond in-process. Roughly 5 ms of that goes to the two HTTP requests and their validation, 7 ms to exchanging handles and copying arrays on both sides, and 4 ms to the JAX recipe's endpoints, which adopt the inputs, key the VJP cache, and launch the forward and backward programs. Compute adds only a few milliseconds on balance. Running the forward pass and the VJP as two separate XLA programs costs about 10 ms more than the single program XLA builds when it sees both at once, but the Tesseract's version of the rollout compiles to a faster program than the team's direct code, which offsets most of that.

### How the cost scales

A useful way to think about any cross-process call is a fixed cost per call plus a cost per byte moved. The figure below measures both for a Tesseract that only doubles its input, so it returns as many bytes as it receives.

```{figure} ../static/blog/gpu-ipc-benchmark.png
:alt: "Log-log plot of median time per call against array size, from 1,000 to 100 million float32 elements. The host round trip rises from about 3 ms to about 3.4 s. cuda_ipc stays near 4 ms up to a few million elements and rises to about 18 ms at 100 million, 190 times faster than the host round trip. The host round trip is faster below about 30,000 elements. The same computation in-process takes between 0.13 and 0.6 ms."

Median time of one jitted `apply_tesseract` call to a Tesseract that doubles a float32 vector, on an NVIDIA A100 80GB, over five fresh processes per setup. The {download}`benchmark script </downloads/gpu-ipc-benchmark.py>` reproduces the figure.
```

Through the host, the fixed cost is about 3 ms, and each megabyte of input adds about 8.5 ms, spent on device-to-host copies, base64, JSON, and HTTP, all of it on the CPU. With `cuda_ipc`, the fixed cost rises to about 4 ms, but a megabyte adds only about 0.035 ms. The bytes now move as a handful of copies within GPU memory (into the staging buffer, into the server's own memory, and the same on the way back), plus the cost of mapping each allocation in the other process.

Two things follow from those numbers. The host path wins only while its per-byte cost is smaller than the extra millisecond of fixed cost, which holds up to about 120 kB, or 30,000 float32 values. For large arrays, the speedup approaches the ratio of the per-byte costs, a few hundred times, and reaches 190 times at 100 million values. In between, what matters for a given loop is whether the fixed cost is small next to the compute per call. That is why coarse calls matter. In Multi-Agent-DPC, each call runs an entire batch of 300-step rollouts, which is about as coarse as calls get.

## Limits and what's next

The GPU transport is built on CUDA IPC, so client and Tesseract must run on the same machine and see the same GPU. A Tesseract serving it runs a single worker, which can still serve concurrent requests. The shim synchronizes the stream before and after every call, which gives up overlap with other GPU work in exchange for simple correctness.

Most of the remaining fixed cost sits in the handle exchange and in HTTP handling. Replacing the full synchronization with CUDA events would let other GPU work overlap with Tesseract calls, and serving the GPU transport from more than one worker is the obvious usability item. A transport that exports XLA's memory by reference, skipping the staging copy and the cost of mapping the whole allocation, [is in review](https://github.com/pasteurlabs/tesseract-core/pull/726) and should help most with arrays in the hundreds of megabytes. Beyond a single machine, the same idea extends to GPUs on other hosts, and that deserves a post of its own once we have measured it.

To try it, start with the [GPU transport how-to](../content/how-to/gpu-transport.md). If you have a GPU component that lives in another stack, or a training loop that bounces arrays through the CPU to reach one, we'd like to hear how this works for you on the [Forum](https://si-tesseract.discourse.group/). New to Tesseract? The [getting started guide](../content/tutorials/get-started.md) covers building and serving your first Tesseract, and the [Tesseract-JAX docs](https://docs.pasteurlabs.ai/projects/tesseract-jax/latest/) explain how to make it differentiable from JAX.
