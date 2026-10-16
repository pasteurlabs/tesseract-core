---
og:title: "When to use Tesseract"
og:description: "When wrapping a component as a Tesseract pays off, and when a custom_vjp or an in-framework rewrite is the better answer."
---

# When to use Tesseract

Tesseract runs a component in its own process and lets JAX and PyTorch differentiate through it as if it were native code. In exchange, every call crosses a process boundary, and the component has to be written as a Tesseract. This page helps you decide when that trade pays off, and points to simpler answers where it doesn't.

## When it fits

Tesseract pays off most when all four of these hold:

| Condition                         | Typical reason                                                                                       | If it doesn't hold                                                                                                                                                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A rewrite is off the table**    | The code is validated, proprietary, decades in the making, or a model trained on data you don't have | Rewrite it in JAX or PyTorch. Textbook functions and small ODE systems are better as native code.                                                                                                                           |
| **You need gradients through it** | It sits inside an optimizer, a sampler, or a training loop                                           | A Tesseract without derivative endpoints still works, but a plain service or script may be all you need.                                                                                                                    |
| **It needs its own process**      | Its dependencies, language runtime, global state, or autodiff framework clash with yours             | Wrap it in a `jax.custom_vjp` or `torch.autograd.Function` and call it in-process, or use a bridge the solver ships, such as Firedrake's `firedrake.ml.pytorch` and `firedrake.ml.jax`.                                     |
| **Each call does real work**      | A call costs a few milliseconds plus the time to move its arrays                                     | Batch calls, or load the Tesseract in-process with `Tesseract.from_tesseract_api`. {doc}`/content/concepts/performance` has measurements and other options. Long chains of cheap calls, like MCMC steps, are the hard case. |

A rewrite is rarely impossible, given enough time. It is off the table when the person who owns the problem wouldn't accept it as the same thing. That happens when the value lies in years of validation against experiment or regulatory acceptance, in capability like turbulence modeling, contact mechanics, or a CAD kernel that no project re-implements on the side, in a surrogate's training data, or in source code you can run but not read.

```{admonition} Beyond the obvious
:class: tip

The above four conditions aren't the whole story. There are other ways in which Tesseract provides value:

- **Testing a hand-written gradient.** A schema plus {doc}`check-gradients </content/how-to/check-gradients>` verifies a VJP or adjoint against finite differences, even if the component later runs in-process.
- **Mixing differentiation strategies.** One gradient can combine finite differences through a cheap CAD geometry with a handful of parameters and an exact adjoint through the expensive solver behind it.
- **Handing a component to someone else.** Schemas and generated API docs describe what a Tesseract expects and returns, without anyone reading its source.

In general, don't let the absence of one of the four main conditions discourage you from using Tesseract if you believe it's the right tool for your problem.
```

## What it costs

- **Time per call.** A call to a Tesseract on the same machine costs a few milliseconds, plus the time to move its arrays. Shared-memory transport keeps the second part small for large arrays (see {doc}`/content/how-to/fast-local-runs`).
- **A fixed structure.** A Tesseract is a folder with a `tesseract_api.py`, a `tesseract_config.yaml`, and a requirements file, and its inputs and outputs are declared as schemas. Much of this is boilerplate that a coding agent can draft from your existing code.

The structure helps when the component is self-contained. Its environment and interface are then declared in one place, and anyone can run it without asking you how. It gets in the way when the component is one function inside a larger codebase that you are still changing, since the Tesseract has to depend on that codebase as a package, or when the natural interface has several entrypoints, which become several Tesseracts.

## Which way to run it

The same `tesseract_api.py` runs in four ways, without changes:

| How                                                                   | What it does                                                                   | Use it when                                                                                                                       |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| [`Tesseract.from_source`](#Tesseract.from_source)                     | Serves it from a local process, in an environment built from its configuration | Developing and calling a Tesseract on your machine. This is the default.                                                          |
| [`Tesseract.from_tesseract_api`](#Tesseract.from_tesseract_api)       | Imports it into your Python process                                            | Its dependencies fit into your environment, and you want the cheapest calls or a debugger (see {doc}`/content/how-to/debugging`). |
| `tesseract build` and [`Tesseract.from_image`](#Tesseract.from_image) | Builds a container image and serves it from Docker                             | You share it with people who don't have your environment, need system packages or compiled toolchains, or deploy to the cloud.    |
| [`Tesseract.from_url`](#Tesseract.from_url)                           | Connects to a Tesseract served elsewhere                                       | It runs on another machine or a cluster, or several programs share one instance (see {doc}`/content/how-to/deploy`).              |

## What you get

The usual first attempt at calling a component in another environment is to write arrays to disk, run the other environment with `subprocess.run`, and read the results back. That works for a few dozen calls. In a training loop with thousands of gradient calls, each one pays for a fresh process and a round trip through the filesystem, the solver's state is lost between calls, and every shape and error is handled by hand.

With Tesseract, the component stays alive in its own process, its inputs and outputs are typed, and [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) and [Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch) make it a differentiable operation in your program. Its gradient can come from an adjoint solver, autodiff in another framework, or finite differences, and `jax.grad` or `torch.autograd` combine it with the rest of your program. A Fortran solver, a Julia package, and a PyTorch model can share one gradient without sharing an interpreter, and the component you wrote on your laptop runs unchanged on a cluster or in the cloud.

Ready to try it? The {doc}`/content/tutorials/get-started` tutorial writes and calls a Tesseract from scratch.
