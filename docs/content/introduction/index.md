---
og:title: "Tesseract Core Documentation"
og:description: "Tesseract wraps any solver behind a typed interface with derivative endpoints, runs it in its own process, and hands it to jax.grad or torch.autograd as a single differentiable operation."
---

# Tesseract Core Documentation

Tesseract wraps any solver behind a typed interface with derivative endpoints, runs it in its own process, and hands it to `jax.grad` or `torch.autograd` as a single differentiable operation.
New here? Start with the [Get Started](../tutorials/get-started.md) tutorial, or read [](../concepts/when-to-use.md) first.

## What a Tesseract is

```{figure} ../../img/tesseract-interfaces.svg
:alt: Tesseract interfaces
:width: 300px
:align: right
:figclass: only-light

<small>Internal and external interfaces of a Tesseract.</small>
```

```{figure} ../../img/tesseract-interfaces-dark.svg
:alt: Tesseract interfaces
:width: 300px
:align: right
:figclass: only-dark

<small>Internal and external interfaces of a Tesseract.</small>
```

A Tesseract is a component defined in one file, `tesseract_api.py`:

- **Schemas** declare its inputs and outputs as Pydantic models and mark which fields are differentiable.
- **`apply`** is the primary entrypoint and wraps whatever software you choose.
- **Derivative endpoints** (`jacobian`, `jacobian_vector_product`, `vector_jacobian_product`) are optional and differentiate `apply`.
- **`abstract_eval`** returns the output structure without running `apply`. See [endpoints](../reference/endpoints.md) for the full list.
- **A process boundary** separates it from its callers, who reach it over HTTP or through the Python SDK instead of importing it, so it keeps its own interpreter and dependencies.

## Design decisions

A few choices shape what working with a Tesseract is like. Each one helps in some settings and limits you in others.

- **One `apply` per Tesseract.** A Tesseract exposes a single function, which keeps its interface and gradients easy to reason about. To expose several functions, build several Tesseracts and compose them (see [](../concepts/design-patterns.md)).
- **Python is always the glue.** A Tesseract can wrap software in any language, but the code between the runtime and that software is Python. Every Tesseract gets the same schemas and tooling as a result, and Python projects are the easiest to wrap.
- **No knowledge of its context.** A Tesseract doesn't know which loop calls it or where it runs. The same component works inside an optimizer, a training loop, or a sampler, and orchestration stays in the calling code.

## The workflow

1. **Define** the schemas and endpoints in `tesseract_api.py`. See [](../tutorials/create.md) and [](../how-to/defining-apis.md).
2. **Check** every derivative endpoint against finite differences with [`check-gradients`](../how-to/check-gradients.md).
3. **Run** it in its own environment with `Tesseract.from_source`, which needs no Docker, and call it from Python.
4. **Compose** it into JAX or PyTorch programs with [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) or [Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch), or chain several Tesseracts into a [pipeline](../how-to/pipelines.md). [](../concepts/differentiable-programming.md) explains how gradients cross the boundary.
5. **Ship** it once it needs to run somewhere else, as a container image built with `tesseract build` or as a remote service reached with `Tesseract.from_url`. See [](../how-to/deploy.md).

For tests and debugging, `Tesseract.from_tesseract_api` imports a Tesseract directly into the calling process (see [](../how-to/debugging.md)).

## Where to go next

- **Using a Tesseract someone else built:** [](../tutorials/interact.md)
- **Finding ready-made differentiable solvers:** [Mosaic](https://github.com/pasteurlabs/mosaic), a benchmark suite of physics solvers packaged as Tesseracts
- **Making calls faster:** [](../concepts/performance.md)
- **Seeing complete examples:** [](../demo/demo.md)

## Citing Tesseract

If you use Tesseract in your research, please cite:

```bibtex
@article{TesseractCore,
  doi = {10.21105/joss.08385},
  url = {https://doi.org/10.21105/joss.08385},
  year = {2025},
  publisher = {The Open Journal},
  volume = {10},
  number = {111},
  pages = {8385},
  author = {Häfner, Dion and Lavin, Alexander},
  title = {Tesseract Core: Universal, autodiff-native software components for Simulation Intelligence},
  journal = {Journal of Open Source Software}
}
```

```{toctree}
:caption: Introduction
:maxdepth: 2
:hidden:

installation.md
Changelog <https://github.com/pasteurlabs/tesseract-core/releases>
```

```{toctree}
:caption: Tutorials & Examples
:maxdepth: 2
:hidden:

../tutorials/get-started.md
../tutorials/create.md
../tutorials/interact.md
../demo/demo.md
../examples/example_gallery.md
../examples/ansys_gallery.md
```

```{toctree}
:caption: How-to Guides
:maxdepth: 2
:hidden:

../how-to/defining-apis.md
../how-to/pipelines.md
../how-to/advanced-usage.md
../how-to/deploy.md
../how-to/check-gradients.md
../how-to/fast-local-runs.md
../how-to/debugging.md
```

```{toctree}
:caption: Concepts
:maxdepth: 2
:hidden:

../concepts/when-to-use.md
../concepts/design-patterns.md
../concepts/differentiable-programming.md
../concepts/performance.md
```

```{toctree}
:caption: Reference — SDK
:maxdepth: 2
:hidden:

../reference/tesseract-cli.md
../reference/tesseract-api.md
../reference/config.md
```

```{toctree}
:caption: Reference — Runtime
:maxdepth: 2
:hidden:

../reference/endpoints.md
../reference/array-encodings.md
../reference/tesseract-runtime-cli.md
../reference/tesseract-runtime-api.md
```
