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

Plan: repo-root gpu-jax-post-plan.md. Note that `blog_draft: true` is not honored by
docs/conf.py (_collect_blog_posts lists every post with a blog_date), so committing
this file publishes it in the blog index. Before publishing, rename the file, set
blog_date to the publication date, update the link in 2026-12-01-boring-envelope.md,
and drop `blog_draft`.

PLACEHOLDERS. Every number marked **[placeholder: ...]** is invented. Replace all of
them with measurements from workstream B of the plan (Multi-Agent-DPC on one GPU box):
- GPU model, CUDA version, driver
- the four rows of the timing table
- the share of the gap closed by the GPU transport
- the remaining per-step overhead and its breakdown

Computed from the code, not placeholders: 300 steps; batch of 32; about 1.2 million
float32 values per call (32 x 300 x (100 + 3 x 8)).

OPEN ITEMS
- Gates: tesseract-core release with #781 and #669 (from_source), tesseract-jax
  v0.5.0, docs how-to for the GPU transport (drafted at
  docs/content/how-to/gpu-transport.md, has its own TODOs), 2026 winners
  announced.
- Permission: the Multi-Agent-DPC/CINOC team (story, CINOC mention, appendix
  quote; offer review of the first two sections), and the authors of Prismo,
  δsnow17-sacsma, and Differentiable Silicon. Check each description against the
  project's README.
- The VJP-cache row assumes the solver uses the recipe's experimental cache
  (set_jax_vjp_cache_size, core #577). The team's Tesseract has a hand-rolled one on
  the repro branch. Make the measured setup match the sentence.
- If tesseract-jax #274 (traceable=True) has merged, add one sentence to "Why would
  you ever do that" (marked below).
- Figure: docs/static/blog/gpu-call-stack.png is rendered from gpu-call-stack.svg
  (rsvg-convert -z 1.25 gpu-call-stack.svg -o gpu-call-stack.png). Leave `vmap` on
  the GPU path unclaimed unless workstream B exercises it beyond broadcast_all.
- The solver's apply runs eqx.filter_jit on its raw inputs. Handle-backed inputs
  arrive as device wrappers, so it probably needs to adopt them with
  jnp.from_dlpack first (untested: Tesseract-JAX's GPU tests serve CuPy).
-->

# `jax.grad` across a process boundary, without leaving the GPU

The winning entry of our first hackathon, [Multi-Agent-DPC](https://github.com/SOLARIS-JHU/Multi-Agent-DPC) by Pietro Zanotta, Dibakar Roy, and Honghui Zheng, teaches a swarm of mobile heat sources to shape a temperature field. A small policy network decides how each agent moves, and it learns by backpropagating through 300 steps of a differentiable PDE solver. The team wrapped that solver as a [Tesseract](https://github.com/pasteurlabs/tesseract-core) and called it from their JAX training loop through [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), which makes a Tesseract behave like any other differentiable JAX function.

It worked, but it was far too slow to train with, and their published training script calls the solver directly with the Tesseract path switched off. Two things stood in the way. The solver's container installed the CPU build of JAX, so while the training loop ran on a laptop GPU, the solver did not. And every call across the boundary took the long way round. Each array left the GPU for host memory, was base64-encoded into a JSON body, crossed HTTP, was decoded on the other side, and made the same trip back. A training step makes two such calls, the forward rollout and the vector-Jacobian product (VJP) on the way back, and each carries about 1.2 million float32 values. Rather than giving up quietly, the team published a [branch that reproduces the slowdown](https://github.com/SOLARIS-JHU/Multi-Agent-DPC/tree/repro-performance-issue), and that is where this work started.

Tesseract-JAX can now keep both calls on the GPU. The solver runs in its own process, and only 64-byte memory handles cross the boundary. Below, we rerun the team's training step, take on the obvious question of why anyone would put a process boundary inside a GPU training loop, and walk down the call stack that makes it work.

## The same training step, on the GPU

Running the solver in its own process no longer requires a container. `Tesseract.from_source` serves a `tesseract_api.py` from a subprocess, so the solver picks up the same CUDA-enabled JAX as the training loop. Asking for the GPU transport takes one keyword on each end (the [how-to guide](../content/how-to/gpu-transport.md) has the details):

```python
import os

# Two JAX processes share one GPU, so neither should claim 75% of it at startup.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax.flatten_util import ravel_pytree
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

with Tesseract.from_source(
    "tesseracts/solverHeat_decentralized/tesseract_api.py",
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
        return apply_tesseract(
            solver, inputs, vmap_method="broadcast_all", gpu_transport="cuda_ipc"
        )

    # The rest is the team's training loop, unchanged:
    # jax.jit(jax.value_and_grad(mean over jax.vmap(loss))), then an Adam update.
```

We timed one training step, a batch of 32 rollouts forward and backward plus the optimizer update, on an NVIDIA **[placeholder: GPU model]** with CUDA **[placeholder: version]**. All four rows ran on that machine with the solver on the GPU. The team's original numbers came from a setup with the solver on the CPU, so they aren't comparable.

| Setup                                                 |    Time per training step |
| ----------------------------------------------------- | ------------------------: |
| Solver imported directly, one process                 | **[placeholder: 0.30 s]** |
| Solver in its own process, arrays through host memory | **[placeholder: 1.12 s]** |
| Solver in its own process, `cuda_ipc`                 | **[placeholder: 0.49 s]** |
| Same, with the solver's VJP cache                     | **[placeholder: 0.34 s]** |

Removing the host round trip closes **[placeholder: about three quarters]** of the gap to the single-process version. The last row turns on an experimental option in Tesseract's JAX recipe that keeps the residuals from the forward pass, so the VJP call can reuse them instead of rerunning all 300 steps. What remains, about **[placeholder: 40 ms]** per step, is mostly fixed cost: **[placeholder: two HTTP requests and their validation (~20 ms), full stream synchronization around each call (~15 ms), and a few device-to-device copies (~5 ms)]**. That overhead is roughly constant per call, so it matters less the more work the solver does per call. Here each call runs an entire 300-step rollout, which is about as coarse as calls get.

## Why would you ever do that

The pipeline above has JAX on both sides of a process boundary, and nobody should build that on purpose. If both halves are JAX, you own both, and they can share an environment, import the solver. The Multi-Agent-DPC team came to the same conclusion. [CINOC](https://github.com/SOLARIS-JHU/CINOC), the ICML 2026 paper that grew out of the project, imports its solvers directly, which is the right call for two JAX components with the same authors.

<!-- If tesseract-jax #274 (traceable=True) has merged, add one sentence here:
when both sides are JAX, Tesseract-JAX can inline the call instead. -->

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

In Python, the views go to the same dispatch function the CPU path uses, which passes them to the Tesseract's HTTP client. With the GPU transport enabled, the client doesn't encode any bytes. For each array it asks CUDA for an IPC handle, a 64-byte token that another process on the same machine can use to map the same allocation, and it puts that handle in the JSON request where the base64 would otherwise go. Shapes and dtypes stay in the JSON, so the server still validates every request against its schema.

The solver process opens the handles, copies the inputs into memory it owns, and computes the VJP. Its outputs travel back the same way, with two complications. JAX exposes its arrays through DLPack rather than `__cuda_array_interface__`, so the runtime reads the device pointer from a DLPack capsule. And XLA allocates memory through CUDA's virtual memory management API, which legacy IPC handles can't refer to, so the runtime first copies each output into a plain `cudaMalloc` buffer it can export. The response carrying those handles is sent only after the endpoint returns, so the server keeps the buffers alive until the next request arrives. This is also why a Tesseract serving the GPU transport handles one request at a time for now.

Back in the training process, the client opens the returned handles, and the shim copies the results into XLA's output buffers, synchronizes once more, and returns control to XLA.

We wrote this shim twice. The first version was in Rust and ran about a millisecond faster, but only because it reimplemented the entire Tesseract client, HTTP and JSON included, in thousands of extra lines. The C++ shim only moves pointers and leaves everything else to the existing Python client. Once the array bytes stopped passing through that client, there was very little left in it to optimize.

## Failing silently

When a GPU transport breaks, it rarely crashes. Usually it just gets slow, and nothing tells you. The most instructive bug of this project was one of those. The runtime decided whether an output lived on the GPU by looking for `__cuda_array_interface__`, the protocol CuPy and PyTorch implement. JAX arrays expose their device memory only through DLPack. A Tesseract written in JAX, like the Multi-Agent-DPC solver, therefore returned arrays the runtime didn't recognize as GPU arrays, and they quietly took the host path. Every test passed, because every result was correct. [#781](https://github.com/pasteurlabs/tesseract-core/pull/781) fixed it by reading device metadata from either protocol.

The more general lesson is that the fast path has to check itself. Tesseract-JAX's GPU tests now run with a residency check switched on (`TESSERACT_JAX_DEBUG_CHECK_DEVICE_PTRS`), which makes the shim reject any pointer that isn't in device memory, so an accidental host round trip on any dispatch path fails loudly. For the same reason, asking for `gpu_transport="cuda_ipc"` on a machine where the shim isn't available raises an error instead of silently falling back to the host path.

Two other bugs are worth recording. The staging copy for XLA's memory begins with a `cudaIpcGetMemHandle` call that is expected to fail, and does. CUDA also records that failure as a process-wide error, which XLA's next kernel launch picks up as its own. The first `jit`-compiled function ran fine and every newly compiled one after it aborted, until we cleared the error right after the expected failure. The second bug appeared only at exit, when a Python object held in a C++ static variable was destroyed after the interpreter had shut down and took the process with it. The fix is to allocate the object on the heap and never free it, one of the few cases where a memory leak is the correct answer.

## Limits

The GPU transport is experimental and opt-in, and it is built on CUDA IPC, which sets its boundaries. Client and Tesseract must run on the same machine and see the same GPU. They must also share an IPC namespace, which means `--ipc=host` when the Tesseract runs in a container. A Tesseract serving the GPU transport handles one request at a time. Arrays must be C-contiguous, and their dtypes must match the schema exactly. The shim synchronizes the stream before and after every call, which gives up overlap with other GPU work in exchange for simple correctness. When two JAX processes share a GPU, as in the example above, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` or lower `XLA_PYTHON_CLIENT_MEM_FRACTION`, because otherwise each process tries to claim 75% of device memory at startup.

## What's next

The staging copy goes away with a transport designed for the memory XLA actually allocates. `cuda_vmm`, currently [in review](https://github.com/pasteurlabs/tesseract-core/pull/726), exports that memory directly. It also avoids a cost that grows with allocation size whenever the receiving process opens a legacy handle. In our tests it is about seven times faster than `cuda_ipc` on 2 GB arrays and a couple of milliseconds slower below roughly 150 MB. Replacing the full synchronization with CUDA events would let other GPU work overlap with Tesseract calls, and serving more than one request at a time is the obvious usability item. Beyond a single machine, the same idea extends to GPUs on other hosts, and that deserves a post of its own once we have measured it.

To try it, start with the [GPU transport how-to](../content/how-to/gpu-transport.md). If you have a GPU component that lives in another stack, or a training loop that bounces arrays through the CPU to reach one, we'd like to hear how this works for you on the [Forum](https://si-tesseract.discourse.group/). New to Tesseract? The [getting started guide](../content/tutorials/get-started.md) covers building and serving your first Tesseract, and the [Tesseract-JAX docs](https://docs.pasteurlabs.ai/projects/tesseract-jax/latest/) explain how to make it differentiable from JAX.
