# Tesseract

Autodiff-native, self-documenting software components for [Simulation Intelligence](#what-is-si). 📦

```{seealso}
Already convinced? 👉 See how to [Get started](content/introduction/get-started.md) instead.
```

## What is a Tesseract?

Tesseracts are components that expose experimental, research-grade software to the world. They are self-contained, self-documenting, and self-executing, via command line and HTTP. They are designed to be easy to create, easy to use, and easy to share, including in a production environment. This repository contains all you need to define your own and execute them.

Tesseracts provide built-in support for propagating [gradient information](content/introduction/differentiable-programming) at the level of individual components, making it easy to build complex, diverse software pipelines that can be optimized end-to-end.

## Core concepts

```{figure} img/tesseract-interfaces.png
:alt: Tesseract interfaces
:width: 220px
:align: right

<small>Internal and external user interfaces.</small>
```

Every Tesseract has a single entrypoint, `apply`, which wraps a software functionality of the tesseractor's choice. All other [endpoints](content/api/endpoints.md) that a Tesseract exposes are in relation to this entrypoint. For example, `input_schema` returns its input types, `jacobian` its derivative, ...

There are several ways users might interact with Tesseracts:

1. When defining entrypoints - `tesseract_api.py`.
2. When building a container – `tesseract build`.
3. Invocation via HTTP - `tesseract serve`.
4. Invocation via the command line - `tesseract run`.

## Features and restrictions

::::{tab-set}
:::{tab-item} Features
- **Self-documenting** – Tesseracts announce their interfaces, so that users can inspect them without needing to read the source code, and perform static validation without running the code.
- **Auto-validating** – When data reaches a Tesseract, it is automatically validated against the schema, so that internal logic can be sure that the data is in the expected format.
- **Autodiff-native** – Tesseracts support [Differentiable Programming](content/introduction/differentiable-programming), meaning that they can be used in gradient-based optimization algorithms – or not, since exposing derivatives is *strictly optional*.
- **Batteries included** – Tesseracts ship with a containerized runtime, which can be run on a variety of platforms, and exposes the Tesseract's functionality via a command line interface (CLI) and a REST API.
:::
:::{tab-item} Restrictions
- **Python first** – Although Tesseracts may use any software under the hood, Tesseracts always use Python as glue between the Tesseract runtime and the wrapped functionality. This also means that support for working with Python projects is more mature than other languages.
- **Single entrypoint** – Tesseracts have a single entrypoint, `apply`, which wraps a software functionality of the tesseractor's choice. When exposing N entrypoints of a software, users need to create N distinct Tesseracts.
- **Context-free** – Tesseracts are not aware of outer-loop orchestration or runtime details.
- **Runtime overhead** – Tesseracts are primarily designed for compute kernels and data transformations that run at least several seconds, so they may not be the best choice for workloads with very low latency requirements.
:::
::::

## Why Tesseracts?

Tesseracts are primarily useful for managing **diversity**.

- **Diversity of roles** *(within a team)* – The job of the software creator ends when it is packaged as a Tesseract. Users creating pipelines can focus on the high-level logic of their application, and not worry about the low-level details of how the components are implemented. Team members requesting data can inspect interfaces, docs, and schemas but don't need to dive into implementations.

- **Diversity of workloads** – By enforcing a standardized way to define interfaces, software components will work together as long as they are passed inputs that adhere to the expected schema. The comprehensive auto-generated schemas make it possible to build composable data pipelines with minimal friction.

- **Diversity of software** – Components can be implemented in any framework / language, from PyTorch or JAX, to a wrapped C++ or Julia backend, to a glorified shell script gluing together different components. All that's needed is a thin Python wrapper (`tesseract_api.py`) to hook up external entry points to internal logic.

- **Diversity of hardware** – Components do not need to be executed on the same hardware to work together in a shared pipeline, including end-to-end autodiff.

If you're a single developer working with a single software stack in a single environment – you might not need Tesseracts. Everyone else, read on!


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
:caption: Tutorials
:maxdepth: 2
:hidden:

Expert showcase <https://si-tesseract.discourse.group/c/showcase/expert-showcase/12>
How-to Guides <https://si-tesseract.discourse.group/c/support/how-to-guides/10>
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
