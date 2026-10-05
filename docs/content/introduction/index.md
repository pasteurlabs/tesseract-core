---
og:title: "Tesseract Core Documentation"
og:description: "Tesseract wraps scientific software behind a typed, differentiable interface and runs it in its own process, so JAX and PyTorch programs can train and optimize through solvers they can't replace."
---

# Tesseract Core Documentation

Tesseract wraps scientific software behind a typed, differentiable interface and runs it in its own process, so that JAX and PyTorch programs can train and optimize through solvers they can't replace.
New here? Start with the [Get Started](../tutorials/get-started.md) tutorial, or read [](../concepts/when-to-use.md) first.

## How it works

```{figure} ../../img/tesseract-interfaces.png
:alt: Tesseract interfaces
:width: 250px
:align: right

<small>Internal and external interfaces of a Tesseract.</small>
```

Every Tesseract has a primary entrypoint, `apply`, which wraps a software functionality of your choice. All other [endpoints](../reference/endpoints.md) relate to this entrypoint: `abstract_eval` returns output structure, `jacobian` computes derivatives, and so on.

A Tesseract is an inter-process contract. The same `tesseract_api.py` can run in three ways, exposing the same endpoints over HTTP and through the Python SDK in each:

1. **As a subprocess**, in its own environment, via `Tesseract.from_source`. This is the quickest way to start and needs no Docker.
2. **As a container image**, built with `tesseract build`, for sharing and deployment.
3. **As a remote service**, reached via `Tesseract.from_url`.

For tests and debugging, `Tesseract.from_tesseract_api` also imports a Tesseract directly into the calling process.

The typical workflow is to **define** endpoints in `tesseract_api.py`, **check** its gradients with [`check-gradients`](../how-to/check-gradients.md), **compose** it into JAX or PyTorch programs with [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) or [Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch), and **build** an image once it needs to run somewhere else.

## Features and limitations

::::{tab-set}
:::{tab-item} Features

- **Self-documenting** — Tesseracts announce their interfaces, so users can inspect them without reading source code and perform static validation without running the code.
- **Auto-validating** — Input data is automatically validated against the schema, so internal logic can assume the data is in the expected format.
- **Autodiff-native** — Tesseracts support [differentiable programming](../concepts/differentiable-programming.md) and integrate as native operations in PyTorch and JAX — but exposing derivatives is _strictly optional_.
- **Batteries included** — Every Tesseract comes with a CLI, a REST API, and a Python SDK, and runs as a subprocess, a container, or a remote service.

:::
:::{tab-item} Limitations

- **Python as glue** — Tesseracts may use any software under the hood, but they always use Python as glue between the runtime and the wrapped functionality. Support for Python projects is more mature than other languages.
- **Single entrypoint** — Each Tesseract has a single `apply` entrypoint. To expose N functions, create N Tesseracts.
- **Context-free** — Tesseracts are not aware of outer-loop orchestration or runtime details.
- **Runtime overhead** — Calls usually cross a process boundary, which costs milliseconds, so Tesseracts suit components whose calls take much longer than that (see [performance](../concepts/performance.md)).

:::
::::

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
Tesseract User Forums <https://si-tesseract.discourse.group/>
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
../how-to/llm-assistance.md
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
