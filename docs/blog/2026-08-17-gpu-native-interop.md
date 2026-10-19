---
orphan: true
blog_draft: true
og:title: "jax.grad across a process boundary, without leaving the GPU"
og:description: "Tesseract-JAX can now call a solver running in another process on the same GPU, forward and backward, without copying a single array through host memory. Here is why you would want that, and the call stack that makes it work."
blog_date: "2026-10-01"
blog_author: "@dionhaefner"
blog_title: "jax.grad across a process boundary, without leaving the GPU"
blog_description: "Tesseract-JAX can now call a solver running in another process on the same GPU, forward and backward, without copying a single array through host memory. Here is why you would want that, and the call stack that makes it work."
---

<!--
DRAFT STATUS

Note that `blog_draft: true` is not honored by docs/conf.py (_collect_blog_posts lists
every post with a blog_date), so committing this file publishes it in the blog index.
Before publishing, rename the file, set blog_date to the publication date, update the
link in 2026-12-01-boring-envelope.md, and drop `blog_draft`.

MEASUREMENTS. All numbers were measured on 2026-10-19 on an NVIDIA A100 80GB PCIe
(driver 580.159.03, CUDA 13.0), JAX 0.11.2 (jax[cuda13]), flax 0.12.10,
tesseract-core 1.15.0 and tesseract-jax 0.5.0 from PyPI. Each table row is the median
over 3 fresh processes x 50 timed steps (5 warm-up steps); per-process medians agree
within 1.5 ms. A separate pass compared every step's gradient against the in-process
solver (cos >= 0.99999998, rel. L2 <= 2e-4, GPU nondeterminism) with
TESSERACT_FORBID_DEVICE_HOST_COPY and TESSERACT_JAX_DEBUG_CHECK_DEVICE_PTRS set.
Harness, raw outputs and the breakdown below: ~/localfiles/runs/blog-gpu-1.15.
Remaining-cost breakdown (same setup, medians of 3 processes): null Tesseract (same
schemas, no work) 22.3 ms/step over cuda_ipc vs 0.7 in-process; timers in both
processes split it into ~5.3 HTTP + server framework/validation, ~7.2 IPC encode/decode
(server input decode 4.0), ~9.2 recipe endpoints (cotangent zeros+set_at_path ~2.1,
adoption ~1.3, jax.vjp ~0.9, cache key ~0.2-0.75), ~1.2 shim/XLA, ~0.7 client JSON.
Compute: the Tesseract's batched rollout fused in-process 97.9 ms, as two programs
(forward with residuals, then VJP) 108.5 ms; the direct row (team's code) is 105.9.
Computed from the code: 300 steps; batch of 32; the rollout returns
32 x 300 x (100 + 3 x 8) = 1,190,400 float32 values, and the VJP sends as many
cotangents (plus 368,192 primal inputs, mostly the 11,298 policy weights broadcast 32
times).

The measured Tesseract is the team's solver.py and policy.py, unchanged, wrapped in the
JAX recipe with Array[..., Float32] shapes. The linked port on
github.com/dionhaefner/Multi-Agent-DPC (repro-performance-issue) is that Tesseract with
the VJP cache always on. The 1.15/0.5 version of the fork is local branch blog-1.15 in
~/localfiles/worktrees/dpc-1.15 (drops gpu_transport= from apply_tesseract and the
_adopt helper, pins 1.15.0/0.5.0/jax[cuda13]); with it, train.py runs 500 epochs in
73 s. Push it to repro-performance-issue before publishing.

OPEN ITEMS
- Gates: push the fork update above; docs how-to for the GPU transport
  (docs/content/how-to/gpu-transport.md, has its own TODOs and still says "one request
  at a time"); 2026 winners announced.
- Permission: the Multi-Agent-DPC/CINOC team (story, CINOC mention, appendix quote;
  offer review of the first two sections), and the authors of Prismo, δsnow17-sacsma,
  and Differentiable Silicon. Check each description against the project's README.
- The cuda_vmm figures in "What's next" are from #726's own benchmark (containerized,
  JAX in and out). The per-step comparison in this example was measured on #726's
  branch without #806, against cuda_ipc on the same branch. #726 is still an open
  draft and not in 1.15.0.
- Figure: docs/static/blog/gpu-call-stack.png is rendered from gpu-call-stack.svg
  (rsvg-convert -z 1.25 gpu-call-stack.svg -o gpu-call-stack.png). Leave `vmap` on
  the GPU path unclaimed unless it is exercised beyond broadcast_all.
-->

# `jax.grad` across a process boundary, without leaving the GPU

The winning entry of our first hackathon, [Multi-Agent-DPC](https://github.com/SOLARIS-JHU/Multi-Agent-DPC) by Pietro Zanotta, Dibakar Roy, and Honghui Zheng, teaches a swarm of mobile heat sources to shape a temperature field. A small policy network decides how each agent moves, and it learns by backpropagating through 300 steps of a differentiable PDE solver. The team wrapped that solver as a [Tesseract](https://github.com/pasteurlabs/tesseract-core) and called it from their JAX training loop through [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), which makes a Tesseract behave like any other differentiable JAX function.

It worked, but it was far too slow to train with, and their published training script calls the solver directly with the Tesseract path switched off. Two things stood in the way. The solver's container installed the CPU build of JAX, so while the training loop ran on a laptop GPU, the solver did not. And every call across the boundary took the long way round. Each array left the GPU for host memory, was base64-encoded into a JSON body, crossed HTTP, was decoded on the other side, and made the same trip back. A training step makes two such calls, the forward rollout and the vector-Jacobian product (VJP) on the way back. The rollout returns about 1.2 million float32 values, and the VJP sends as many back. Rather than giving up quietly, the team published a [branch that reproduces the slowdown](https://github.com/SOLARIS-JHU/Multi-Agent-DPC/tree/repro-performance-issue), and that is where this work started.

Tesseract-JAX can now keep both calls on the GPU. The solver runs in its own process, and only 64-byte memory handles cross the boundary. Below, we rerun the team's training step, take on the obvious question of why anyone would put a process boundary inside a GPU training loop, and walk down the call stack that makes it work.

## The same training step, on the GPU

Running the solver in its own process no longer requires a container. `Tesseract.from_source` serves a `tesseract_api.py` from a subprocess. By default it builds the Tesseract its own environment from its requirements, as `tesseract build` would, so we point it at the training loop's interpreter instead, which gives the solver the same CUDA-enabled JAX. Asking for the GPU transport takes one keyword where the Tesseract is served, and Tesseract-JAX picks it up from there (the [how-to guide](../content/how-to/gpu-transport.md) has the details):

```python
import os
import sys

# Two JAX processes share one GPU, so neither should claim 75% of it at startup.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax.flatten_util import ravel_pytree
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

with Tesseract.from_source(
    "tesseracts/solverHeat_decentralized/tesseract_api.py",
    python_executable=sys.executable,
    gpu_transport="cuda_ipc",
) as solver:

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

The Tesseract itself needed one change: its arrays accept an optional leading batch dimension, so `broadcast_all` can send all 32 rollouts in one call. We also took the chance to [port it to Tesseract's JAX recipe](https://github.com/dionhaefner/Multi-Agent-DPC/blob/repro-performance-issue/tesseracts/solverHeat_decentralized/tesseract_api.py), which the last row of the table below relies on. The recipe takes care of the arrays that arrive by handle, too. They arrive as framework-agnostic device arrays, which JAX won't accept as they are, so the recipe adopts them through DLPack, without a copy. A Tesseract that doesn't use the recipe does the same with `jnp.from_dlpack`.

We timed one training step, a batch of 32 rollouts forward and backward plus the optimizer update, on an NVIDIA A100 80GB with CUDA 13 and JAX 0.11.2. All four rows ran on that machine with the solver on the GPU, and each is the median over three fresh processes of 50 steps. The team's original numbers came from a setup with the solver on the CPU, so they aren't comparable.

| Setup                                                 | Time per training step |
| ----------------------------------------------------- | ---------------------: |
| Solver imported directly, one process                 |                 106 ms |
| Solver in its own process, arrays through host memory |                 265 ms |
| Solver in its own process, `cuda_ipc`                 |                 209 ms |
| Same, with the solver's VJP cache                     |                 132 ms |

Removing the host round trip closes about a third of the gap to the single-process version. Most of what the third row still pays for is compute rather than transport: to compute the VJP, the solver reruns the entire 300-step forward pass before it can go backward. The last row turns on an experimental option in Tesseract's JAX recipe that keeps the residuals from the forward pass, so the VJP call can reuse them instead, and together the two close more than 80% of the gap.

What remains, about 26 ms per step, is almost all fixed cost. A Tesseract with the same inputs and outputs that computes nothing costs about 22 ms per step this way, against under a millisecond in-process. Roughly 5 ms of that goes to the two HTTP requests and their validation, 7 ms to exchanging handles and copying arrays on both sides, and 9 ms to bookkeeping in the JAX recipe's endpoints, such as adopting the inputs, keying the VJP cache, and preparing the cotangents. Compute adds only a few milliseconds on balance. Running the forward pass and the VJP as two separate XLA programs costs about 10 ms more than the single program XLA builds when it sees both at once, but the Tesseract's version of the rollout compiles to a faster program than the team's direct code, which offsets most of that. The fixed cost is roughly constant per call, so it matters less the more work the solver does per call. Here each call runs an entire 300-step rollout, which is about as coarse as calls get.

## Why would you ever do that

The pipeline above has JAX on both sides of a process boundary, and nobody should build that on purpose. If both halves are JAX, you own both, and they can share an environment, import the solver. The Multi-Agent-DPC team came to the same conclusion. [CINOC](https://github.com/SOLARIS-JHU/CINOC), the ICML 2026 paper that grew out of the project, imports its solvers directly, which is the right call for two JAX components with the same authors.

JAX within JAX was still the right test for us because it isolates the one thing we changed. With the same framework, dtype, and device on both sides, the rows of the table differ only in what it costs to cross.

The real answer is that the training loop never knew it was talking to JAX. It calls `apply_tesseract` and gets differentiable arrays back, and the process on the other side can be anything that implements the same endpoints. CINOC's appendix points in that direction: "For large-scale or high-fidelity simulations, the Tesseract framework (Häfner & Lavin, 2025) provides scalable alternatives." Three entries in this year's hackathon show what those look like:

- [Prismo](https://github.com/benvial/prismo) optimizes the doping of a silicon photonic phase shifter. It couples a Julia drift-diffusion solver with a FEniCS eigenmode solver stuck on Python 3.10, all driven by `jax.grad` on Python 3.12. Its README spells out the alternative: "a hand-rolled subprocess protocol per solver plus a hand-written chain rule between them."
- [δsnow17-sacsma](https://github.com/kasProg/dsnow17-sacsma) trains a neural network to predict the parameters of two Fortran 77 hydrology models that NOAA uses for river forecasting, with gradients flowing through both.
- [Differentiable Silicon](https://github.com/hozaifa1/differentiable-silicon) optimizes device parameters through Synopsys Sentaurus, a closed-source semiconductor simulator on a separate machine reached over SSH, differentiated with finite differences and composed with a PyTorch model.

Sometimes the other side really is JAX, just not a JAX your process can host. A solver whose gradients need float64 has to switch on `jax_enable_x64`, and that flag applies to the whole process, including any float32 model sharing it. If a solver needs a different JAX release altogether, `from_source` can serve it from a separate Python environment.

In every one of these cases, the component decides where the boundary goes. A surrogate trained and served by another team fixes it just as firmly as a Fortran code does. What has changed is the price of crossing. When the component on the other side lives on the same GPU, crossing the boundary no longer means a trip through host memory.

## One gradient, from top to bottom

```{figure} ../static/blog/gpu-call-stack.png
:alt: "Diagram of one gradient call. In the training process, jax.grad, the Tesseract-JAX primitive, an XLA FFI custom call, a C++ shim, and the Tesseract HTTP client are stacked top to bottom. The client exchanges a JSON request and response carrying 64-byte handles with the Tesseract runtime in the solver process, which calls the solver's VJP in tesseract_api.py. Underneath both processes, one GPU holds XLA's buffers and the solver's buffers, with device-to-device copies between them."

One VJP call, top to bottom. Only JSON with 64-byte handles crosses the process boundary, while the array data stays in the memory of the GPU that both processes share.
```

When `jax.grad` reaches the Tesseract call, it asks Tesseract-JAX for a VJP. Tesseract-JAX answers every derivative request the same way, by binding one JAX primitive that stands for a call to one of the Tesseract's endpoints. Forward evaluations, Jacobian-vector products, VJPs, and full Jacobians all go through it, so nothing below this point has code specific to gradients.

Under `jit`, XLA needs to know how to run that primitive. On the CPU it runs as a host callback into Python, and until now it ran that way on the GPU too. Host callbacks receive NumPy arrays, so XLA copies every input off the device before Python sees it, and no wire format downstream can undo that. JAX has no Python-level API for handing GPU buffers out of a compiled program, so the fix had to be native. On the `cuda` platform, Tesseract-JAX now lowers the primitive to an [XLA FFI](https://docs.jax.dev/en/latest/ffi.html) custom call into a small C++ shim.

The shim receives raw device pointers from XLA. It synchronizes XLA's stream so the inputs are ready, wraps each pointer in a minimal object that exposes `__cuda_array_interface__`, takes the GIL, and calls back into Python with those views. It loads the CUDA runtime with `dlopen` on first use and links against no CUDA library at build time, so the same wheel installs and runs on machines without a GPU.

In Python, the views go to the same dispatch function the CPU path uses, which passes them to the Tesseract's HTTP client. With the GPU transport enabled, the client doesn't encode any bytes. For each array it asks CUDA for an IPC handle, a 64-byte token that another process on the same machine can use to map the same allocation, and it puts that handle in the JSON request where the base64 would otherwise go. Shapes and dtypes stay in the JSON, so the server still validates every request against its schema. One complication is that XLA allocates memory through CUDA's virtual memory management API, which legacy IPC handles can't refer to. The client therefore first copies each array into a plain `cudaMalloc` buffer it can export. Those staging buffers, and their handles, are reused from call to call, since allocating fresh GPU memory on every call can cost tens of milliseconds per gigabyte when two processes share a GPU.

The solver process maps the handles, copies the inputs into memory it owns, closes the mappings, and computes the VJP. Its outputs are XLA allocations too, and they travel back the same way. The server keeps a response's buffers alive until the client reports, with its next request, that it has copied the results out.

Back in the training process, the client maps the returned handles, and the shim copies the results into XLA's output buffers, synchronizes once more, and returns control to XLA.

We wrote this shim twice. The first version was in Rust and ran about a millisecond faster, but only because it reimplemented the entire Tesseract client, HTTP and JSON included, in thousands of extra lines. The C++ shim only moves pointers and leaves everything else to the existing Python client. Once the array bytes stopped passing through that client, there was very little left in it to optimize.

## Failing silently

When a GPU transport breaks, it rarely crashes. Usually it just gets slow, and nothing tells you. The team's original setup is an example: the solver's container installed the CPU build of JAX, every result was correct, and the only symptom was the time each step took. Asking for `gpu_transport="cuda_ipc"` guards against part of this. Before the first call, the client exchanges a small test array with the server over IPC, and if that fails, or if the shim isn't installed, the call raises with the reason instead of silently falling back to the host path. A Tesseract that doesn't ask for a transport, such as one from `Tesseract.from_url`, uses `cuda_ipc` when the server offers it and it works, and otherwise copies through the host. If the server offered `cuda_ipc` and the test failed, it warns once. Both checks only run when JAX compiles the call for a GPU, though. If JAX itself ends up on the CPU, for example because its CUDA plugin failed to load, the GPU transport has no effect and nothing raises, so it is worth checking that `jax.default_backend()` returns `"gpu"` in both processes.

Once both processes are on the GPU, the fast path has to check itself. Tesseract-JAX's GPU tests run with a residency check switched on (`TESSERACT_JAX_DEBUG_CHECK_DEVICE_PTRS`), which makes the shim reject any pointer that isn't in device memory, so an accidental host round trip on any dispatch path fails loudly. That check covers the boundary, but not what happens behind it. Timing each piece of the training step above turned up two costs that no test could catch, because every result was correct. The VJP cache found its entries by hashing the raw bytes of its inputs, which copied every GPU input back to the host on every call. And the gradient endpoints rebuilt a schema validator for every array on every request, about a millisecond each. Together the two cost several milliseconds per training step.

Two other bugs are worth recording. The staging copy for XLA's memory begins with a `cudaIpcGetMemHandle` call that is expected to fail, and does. CUDA also records that failure as a process-wide error, which XLA's next kernel launch picks up as its own. The first `jit`-compiled function ran fine and every newly compiled one after it aborted, until we cleared the error right after the expected failure. The second bug appeared only at exit, when a Python object held in a C++ static variable was destroyed after the interpreter had shut down and took the process with it. The fix is to allocate the object on the heap and never free it, one of the few cases where a memory leak is the correct answer.

A third one we found by reading the code rather than by watching it fail. The runtime handed out an IPC handle without waiting for the GPU to finish writing the memory behind it, so nothing guaranteed that the other process would read finished data. In practice the HTTP round trip always took longer than the copy, which is exactly why the race never showed. The runtime now synchronizes before a handle leaves the process.

The last one needed more than one client. The server used to release a response's buffers when the next request arrived, which assumed one client making one request at a time. With eight client threads, 138 of 800 results silently came back holding another call's data. The server now keeps each response's buffers until the client that received them reports that it has copied them out.

## Limits

The GPU transport is experimental and opt-in, and it is built on CUDA IPC, which sets its boundaries. Client and Tesseract must run on the same machine and see the same GPU, though they may number it differently. A Tesseract in a container runs with `--ipc=host`, which `Tesseract.from_image` adds when you ask for `cuda_ipc`. A Tesseract serving the GPU transport runs a single worker, because the client's report that it has copied a response out could otherwise reach a worker that doesn't hold the buffers. That worker can serve concurrent requests. Input dtypes must match the schema exactly, since the transport doesn't cast on the device. The shim synchronizes the stream before and after every call, which gives up overlap with other GPU work in exchange for simple correctness. To avoid allocating buffers on every call, each process keeps idle ones for reuse, up to a quarter of the GPU's memory by default (`TESSERACT_CUDA_IPC_POOL_FRACTION`). When two JAX processes share a GPU, as in the example above, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` or lower `XLA_PYTHON_CLIENT_MEM_FRACTION`, because otherwise each process tries to claim 75% of device memory at startup.

## What's next

The staging copy is the last copy a transport could remove. [`cuda_vmm`](https://github.com/pasteurlabs/tesseract-core/pull/726) exports XLA's memory by reference instead, which also avoids the cost of mapping a legacy handle, a cost that grows with the size of the allocation. In its own benchmark it is about seven times faster than `cuda_ipc` on 2 GB arrays. It pays a fixed cost per array for passing a file descriptor between the processes, though, which makes it a couple of milliseconds slower below roughly 150 MB. In the training step above, whose arrays are all under 4 MB, it made each step between 13 and 24 ms slower, and Tesseract-JAX doesn't use it yet. Most of the remaining fixed cost here sits in the recipe's bookkeeping, the handle exchange, and HTTP handling instead. Replacing the full synchronization with CUDA events would let other GPU work overlap with Tesseract calls, and serving the GPU transport from more than one worker is the obvious usability item. Beyond a single machine, the same idea extends to GPUs on other hosts, and that deserves a post of its own once we have measured it.

To try it, start with the [GPU transport how-to](../content/how-to/gpu-transport.md). If you have a GPU component that lives in another stack, or a training loop that bounces arrays through the CPU to reach one, we'd like to hear how this works for you on the [Forum](https://si-tesseract.discourse.group/). New to Tesseract? The [getting started guide](../content/tutorials/get-started.md) covers building and serving your first Tesseract, and the [Tesseract-JAX docs](https://docs.pasteurlabs.ai/projects/tesseract-jax/latest/) explain how to make it differentiable from JAX.
