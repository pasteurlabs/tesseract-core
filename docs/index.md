# Tesseract

Universal, autodiff-native software components for Simulation Intelligence

```{seealso}
Already convinced? 👉 See how to [Get started](content/introduction/get-started.md) instead.
```

## The problem

Real-world scientific workflows span multiple tools, languages, and computing environments. You might have a mesh generator in C++, a solver in Julia, and post-processing in Python. Getting these to work together is painful. Getting [gradients to flow through them](content/introduction/differentiable-programming) for optimization? Nearly impossible.

Existing autodiff frameworks work great within a single codebase, but fall short when your pipeline crosses framework boundaries or includes legacy or commercial tools.

## The solution

Tesseract packages scientific software into **self-contained, portable components** that:

- **Run anywhere** — Local machines, cloud, HPC clusters. Same container, same results.
- **Expose clean interfaces** — CLI, REST API, and Python SDK. No more deciphering undocumented scripts.
- **Propagate gradients** — Each component can expose derivatives, enabling end-to-end optimization across heterogeneous pipelines.
- **Self-document** — Schemas, types, and API docs are generated automatically.

## Who is this for?

- **Researchers** building differentiable physics simulations or probabilistic models who need to combine tools from different ecosystems
- **R&D engineers** packaging research code for use by others, without spending weeks on DevOps
- **Platform engineers** deploying scientific workloads at scale with consistent interfaces and dependency isolation

## How it works

```{figure} img/tesseract-interfaces.png
:alt: Tesseract interfaces
:width: 250px
:align: right

<small>Internal and external interfaces of a Tesseract.</small>
```

Every Tesseract has a primary entrypoint, `apply`, which wraps a software functionality of your choice. All other [endpoints](content/api/endpoints.md) relate to this entrypoint: `abstract_eval` returns output structure, `jacobian` computes derivatives, and so on.

There are several ways to interact with Tesseracts:

1. **Define** entrypoints in `tesseract_api.py`
2. **Build** a container with `tesseract build`
3. **Serve** via HTTP with `tesseract serve`
4. **Invoke** via CLI, HTTP, or the [Python API](content/api/tesseract-api.md)

## Features and restrictions

::::{tab-set}
:::{tab-item} Features

- **Self-documenting** – Tesseracts announce their interfaces, so that users can inspect them without needing to read the source code, and perform static validation without running the code.
- **Auto-validating** – When data reaches a Tesseract, it is automatically validated against the schema, so that internal logic can be sure that the data is in the expected format.
- **Autodiff-native** – Tesseracts support [Differentiable Programming](content/introduction/differentiable-programming), meaning that they can be used in gradient-based optimization algorithms – or not, since exposing derivatives is _strictly optional_.
- **Batteries included** – Tesseracts ship with a containerized runtime, which can be run on a variety of platforms, and exposes the Tesseract's functionality via a command line interface (CLI) and a REST API.
  :::
  :::{tab-item} Restrictions
- **Python first** – Although Tesseracts may use any software under the hood, Tesseracts always use Python as glue between the Tesseract runtime and the wrapped functionality. This also means that support for working with Python projects is more mature than other languages.
- **Single entrypoint** – Tesseracts have a single entrypoint, `apply`, which wraps a software functionality of the Tesseract creator's choice. When exposing N entrypoints of a software, users need to create N distinct Tesseracts.
- **Context-free** – Tesseracts are not aware of outer-loop orchestration or runtime details.
- **Runtime overhead** – Tesseracts are primarily designed for compute kernels and data transformations that run at least several seconds, so they may not be the best choice for workloads with very low latency requirements.
  :::
  ::::

## Why Tesseracts?

Tesseracts help you manage **diversity** in scientific computing:

- **Diversity of roles** — The software creator's job ends when code is packaged as a Tesseract. Pipeline builders focus on high-level logic. Team members can inspect interfaces and schemas without diving into implementations.

- **Diversity of software** — Components can use any framework or language: PyTorch, JAX, C++, Julia, or shell scripts. A thin Python wrapper (`tesseract_api.py`) connects everything.

- **Diversity of hardware** — Components don't need to run on the same machine. Distribute work across GPUs, CPUs, and clusters while maintaining end-to-end differentiability.

- **Diversity of differentiation strategies** — Mix automatic differentiation, analytic adjoints, and finite differences in the same pipeline. Each component chooses its own approach.

If you're a single developer working with a single software stack in a single environment, you might not need Tesseracts. Everyone else, read on!

## Ecosystem

Tesseract Core is the foundation of the Tesseract ecosystem. Additional packages extend its capabilities:

- **[tesseract-jax](https://github.com/pasteurlabs/tesseract-jax)** — Utilities for creating JAX-based Tesseracts with automatic differentiation support. If you're using JAX, this package makes it easy to expose JAX functions as Tesseracts with autodiff endpoints generated automatically.
- **[tesseract-streamlit](https://github.com/pasteurlabs/tesseract-streamlit)** — Auto-generate interactive Streamlit web apps from running Tesseracts. Instantly create UIs for your components without writing frontend code.

```{toctree}
:caption: Introduction
:maxdepth: 2
:hidden:

content/introduction/installation.md
content/introduction/get-started.md
content/introduction/differentiable-programming.md
Tesseract User Forums <https://si-tesseract.discourse.group/>
```

```{toctree}
:caption: Creating Tesseracts
:maxdepth: 2
:hidden:

content/creating-tesseracts/create.md
content/creating-tesseracts/design-patterns.md
content/creating-tesseracts/llm-assistance.md
content/creating-tesseracts/advanced.md
content/creating-tesseracts/deploy.md
```

```{toctree}
:caption: Using Tesseracts
:maxdepth: 2
:hidden:

content/using-tesseracts/use.md
content/using-tesseracts/array-encodings.md
content/using-tesseracts/advanced.md
```

```{toctree}
:caption: Examples
:maxdepth: 2
:hidden:

content/examples/example_gallery.md
content/examples/ansys_gallery.md
content/demo/demo.md
Tesseract Showcase <https://si-tesseract.discourse.group/c/showcase/11>
```

```{toctree}
:caption: Misc
:maxdepth: 2
:hidden:

content/misc/faq.md
```

```{toctree}
:caption: API Reference
:maxdepth: 2
:hidden:

content/api/config.md
content/api/endpoints.md
content/api/tesseract-cli.md
content/api/tesseract-api.md
content/api/tesseract-runtime-cli.md
content/api/tesseract-runtime-api.md
```
