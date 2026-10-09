---
og:title: "Do you need Tesseract?"
og:description: "When a process boundary with gradients is worth it, when a custom_vjp or a rewrite is the better answer, and where containers fit in."
---

# Do you need Tesseract?

Tesseract puts a process boundary between your code and a component, with a typed interface and derivative endpoints on the far side. A boundary is not free. Each call costs milliseconds, and someone has to write the schema. This page is meant to help you decide whether your problem needs one.

## The short version

Tesseract is most useful when four things hold at once:

1. **The component can't be replaced.** A rewrite would not be accepted as the same thing.
2. **You need gradients through it**, typically because it sits inside an optimization, inference, or training loop.
3. **It needs its own process**, because it wants a different interpreter, dependency set, language runtime, global state, or autodiff framework than the code calling it.
4. **Each call is coarse enough** that a few milliseconds of overhead don't matter.

If only some of these hold, there is often a simpler answer, and the sections below say what it is.

## Would a rewrite be accepted?

Almost anything can be rewritten in JAX or PyTorch, given time. The useful question is whether the person who owns the problem would accept the rewrite as the same thing. Four kinds of value don't survive a rewrite:

- **Validation.** Years of comparison against experiment, regulatory acceptance, or institutional trust belong to a specific code, and a faithful port is a different code to whoever certified the original.
- **Encoded capability.** Person-decades of turbulence modeling, contact mechanics, meshing, unstructured discretizations, or a CAD kernel are not something a project re-implements on the side.
- **Data and weights.** A surrogate trained on data you don't have can be called but not retrained.
- **Source access.** Proprietary or partner-owned code can be run but not read.

If the component's value is its mathematics, as with a textbook test function or a small ODE system, rewriting it in your framework is usually the better path. Several of our own tutorials wrap components like that, but only to teach the mechanics.

## Does it need its own process?

If the component imports cleanly into your model's environment, write a `jax.custom_vjp` or a `torch.autograd.Function` around it and call it in-process. Some solvers ship a bridge of their own for this. Firedrake, for example, provides `firedrake.ml.pytorch` and `firedrake.ml.jax`. Where such a bridge exists, prefer it, unless the process boundary is itself the requirement. That is the case when the two software stacks won't coexist in one environment, when the solver runs on another machine, or when several consumers need to share one component.

When the component does need its own process, the usual first attempt is to write arrays to disk, call the other environment with `subprocess.run`, and read the results back. That is a reasonable afternoon's work and holds up for a few dozen calls. It stops holding up when the prototype becomes a training loop with thousands of VJP calls, each paying for a fresh process and a round trip through the filesystem, with the solver's state lost between calls and every shape and error handled by hand.

The next step is usually a persistent server with a typed interface and a binary array protocol, which is what Tesseract provides. [`Tesseract.from_source`](#Tesseract.from_source) is the drop-in replacement for the subprocess approach. It serves a `tesseract_api.py` from its own environment, keeps it alive across calls, and plugs into JAX and PyTorch through [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) and [Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch).

## Is each call coarse enough?

A call to a Tesseract on the same machine costs a few milliseconds plus the time to move its arrays, and shared-memory transport keeps the second part small for large arrays. That is negligible for a solver that runs for seconds and dominant for a function that runs for microseconds. {doc}`/content/concepts/performance` has measurements, and {doc}`/content/how-to/fast-local-runs` shows how to cut array transfer costs on one machine.

## Smaller reasons

Some benefits don't depend on all four conditions:

- **Testing a differentiable component in isolation.** A schema plus {doc}`check-gradients </content/how-to/check-gradients>` is a quick way to verify a hand-written VJP or adjoint, even if the component later runs in-process via `Tesseract.from_tesseract_api`.
- **Mixing differentiation strategies.** One gradient can combine finite differences on a cheap, low-dimensional link, such as a CAD geometry with a handful of parameters, with an exact adjoint on the expensive solver behind it.
- **Handing a component to someone else.** Schemas and generated API docs describe what a Tesseract expects and returns without anyone reading its source.

## Where containers fit in

You don't need Docker to develop or use a Tesseract. A container is a deployment choice, and a good one when you want to share a component with someone who doesn't have your environment, serve it from cloud infrastructure, or pin system-level dependencies. The same `tesseract_api.py` serves all three paths, as a subprocess with `Tesseract.from_source`, as a container built with `tesseract build`, and as a remote service reached with `Tesseract.from_url`. See {doc}`/content/how-to/deploy` for the container path.
