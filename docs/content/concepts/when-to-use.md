---
og:title: "When to use Tesseract"
og:description: "When wrapping a component as a Tesseract pays off, and when a custom_vjp or an in-framework rewrite is the better answer."
---

# When to use Tesseract

Tesseract runs a component in its own process and lets JAX and PyTorch differentiate through it as if it were native code. In exchange, calls usually cross a process boundary (see {doc}`/content/concepts/performance`), and the component has to be written as a Tesseract: a folder with a `tesseract_api.py`, a `tesseract_config.yaml`, and a requirements file, with its inputs and outputs declared as schemas (see {doc}`/content/tutorials/create`). That structure suits a self-contained component, since it declares the component's environment and interface in one place. It gets in the way when the component is one function inside a larger codebase that you are still changing, or when its natural interface has several entrypoints, which become several Tesseracts.

This page helps you decide when that trade pays off, and points to simpler answers where it doesn't.

## When it fits

Tesseract pays off most when all four of these hold:

| Condition                         | Typical reason                                                                                       | If it doesn't hold                                                                                                                                                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A rewrite is off the table**    | The code is validated, proprietary, decades in the making, or a model trained on data you don't have | Rewrite it in JAX or PyTorch. Textbook functions and small ODE systems are better as native code.                                                                                                                           |
| **You need gradients through it** | It sits inside an optimizer, a sampler, or a training loop                                           | A Tesseract without derivative endpoints still works, but a plain service or script may be all you need.                                                                                                                    |
| **It needs its own process**      | Its dependencies, language runtime, global state, or autodiff framework clash with yours             | Wrap it in a `jax.custom_vjp` or `torch.autograd.Function` and call it in-process, or use a bridge the solver ships, such as Firedrake's `firedrake.ml.pytorch` and `firedrake.ml.jax`.                                     |
| **Each call does real work**      | A call costs a few milliseconds plus the time to move its arrays                                     | Batch calls, or load the Tesseract in-process with `Tesseract.from_tesseract_api`. {doc}`/content/concepts/performance` has measurements and other options. Long chains of cheap calls, like MCMC steps, are the hard case. |

```{admonition} Beyond the obvious
:class: tip

Tesseract also helps with:

- **Developing, testing, and shipping components in isolation.** Each Tesseract declares its environment and interface in one place, so it can be built and tested on its own, handed to someone who never reads its source, and deployed unchanged. Schemas and generated API docs tell its users what it expects and returns.
- **Testing a hand-written gradient.** A schema plus {doc}`check-gradients </content/how-to/check-gradients>` verifies a VJP or adjoint against finite differences, even if the component later runs in-process.
- **Mixing differentiation strategies.** One gradient can combine finite differences through a cheap CAD geometry with a handful of parameters and an exact adjoint through the expensive solver behind it. It can also combine forward-mode derivatives through a physics model with reverse-mode autodiff through a neural network, as in [δSnowSac](https://github.com/kasProg/dSnowSac).

In general, don't let the absence of one of the four main conditions discourage you from using Tesseract if you believe it's the right tool for your problem.
```

## What you get

The simplest way to call a component in another environment is to write its inputs to disk, run it with `subprocess.run`, and read its outputs back. Tesseract improves on this in many ways:

- **A persistent process.** The component starts once and stays alive between calls, so imports, meshes, and loaded models are reused instead of rebuilt.
- **Typed interfaces.** Pydantic schemas validate every input and output, including array shapes and dtypes, and generate an OpenAPI schema and API docs.
- **Native differentiation.** [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) and [Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch) turn a Tesseract into a JAX primitive or a PyTorch operator, so `jax.grad`, `jax.jit`, and `torch.autograd` treat it like any other operation. Its gradient can come from an adjoint solver, autodiff in another framework, or finite differences.
- **Gradient checks.** `check-gradients` compares every derivative endpoint against finite differences of `apply` (see {doc}`/content/how-to/check-gradients`).
- **Managed environments.** `Tesseract.from_source` builds an isolated uv or conda environment from the Tesseract's requirements and reuses it on later runs. `tesseract build` packages the same folder as a container image.
- **Efficient array transport.** Arrays travel as base64 or as binary files, optionally compressed or on shared memory (see {doc}`/content/how-to/fast-local-runs`). Arrays already on a GPU can skip the copy to the host with the experimental GPU transport (see {ref}`choosing-the-encoding-per-request`).
- **Logs and diagnostics.** Output from the component can stream back to the caller, metrics and artifacts go to files or MLflow, and built-in profiling, tracing, and debug mode help find slow or failing calls (see {doc}`/content/how-to/debugging`).
- **Composition across stacks.** A Fortran solver, a Julia package, and a PyTorch model can share one gradient without sharing an interpreter (see {doc}`/content/how-to/pipelines`).
- **One component, many deployments.** The same `tesseract_api.py` runs in-process, as a local process, in a container, on an HPC cluster, or as a remote service (see {ref}`ways-to-run`).

Ready to try it? The {doc}`/content/tutorials/get-started` tutorial writes and calls a Tesseract from scratch.
