<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/pasteurlabs/tesseract-core/blob/main/docs/static/logo-dark.png" width="128" align="right">
  <img alt="" src="https://github.com/pasteurlabs/tesseract-core/blob/main/docs/static/logo-light.png" width="128" align="right">
</picture>

### Tesseract Core

Universal, autodiff-native software components for [Simulation Intelligence](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/misc/faq.html#what-is-simulation-intelligence) 📦

[Read the docs](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/) |
[Report an issue](https://github.com/pasteurlabs/tesseract-core/issues) |
[Community forum](https://si-tesseract.discourse.group/) |
[Contribute](https://github.com/pasteurlabs/tesseract-core/blob/main/CONTRIBUTING.md)

---

[![DOI](https://joss.theoj.org/papers/10.21105/joss.08385/status.svg)](https://doi.org/10.21105/joss.08385)
[![SciPy](https://img.shields.io/badge/SciPy-2025-blue)](https://proceedings.scipy.org/articles/kvfm5762)

## The problem

Real-world scientific workflows span multiple tools, languages, and computing environments. You might have a mesh generator in C++, a solver in Julia, and post-processing in Python. Getting these to work together is painful. Getting gradients to flow through them for optimization is nearly impossible.

Existing autodiff frameworks work great within a single codebase, but fall short when your pipeline crosses framework boundaries or includes legacy/commercial tools.

## The solution

Tesseract packages scientific software into **self-contained, portable components** that:

- **Run anywhere** — Local machines, cloud, HPC clusters. Same container, same results.
- **Expose clean interfaces** — CLI, REST API, and Python SDK. No more deciphering undocumented scripts.
- **Propagate gradients** — Each component can expose derivatives, enabling end-to-end optimization across heterogeneous pipelines.
- **Self-document** — Schemas, types, and API docs are generated automatically.

## Who is this for?

- **Researchers** interfacing with (differentiable) simulators or probabilistic models, or who need to combine tools from different ecosystems
- **R&D engineers** packaging research code for use by others, without spending weeks on DevOps
- **Platform engineers** deploying scientific workloads at scale with consistent interfaces and dependency isolation

## Example: Shape optimization across tools

The [rocket fin optimization case study](https://si-tesseract.discourse.group/t/parametric-shape-optimization-of-rocket-fins-with-ansys-spaceclaim-pyansys-and-tesseract/109) combines three Tesseracts:

```
[SpaceClaim geometry] → [Mesh + SDF] → [PyMAPDL FEA solver]
         ↑                                      |
         └──────── gradients flow back ─────────┘
```

Each component uses a different differentiation strategy (analytic adjoints, finite differences, JAX autodiff), yet they compose into a single optimizable pipeline.

## Quick start

> [!NOTE]
> Requires [Docker](https://docs.docker.com/engine/install/) and Python 3.10+.

```bash
$ pip install tesseract-core

# Clone and build an example
$ git clone https://github.com/pasteurlabs/tesseract-core
$ tesseract build tesseract-core/examples/vectoradd

# Run it
$ tesseract run vectoradd apply '{"inputs": {"a": [1, 2], "b": [3, 4]}}'
# → {"result": [4.0, 6.0], ...}

# Compute the Jacobian
$ tesseract run vectoradd jacobian '{"inputs": {"a": [1, 2], "b": [3, 4]}, "jac_inputs": ["a"], "jac_outputs": ["result"]}'

# See auto-generated API docs
$ tesseract apidoc vectoradd
```

<p align="center">
<img src="https://github.com/pasteurlabs/tesseract-core/blob/main/docs/img/apidoc-screenshot.png" width="600">
</p>

## Core features

- **Containerized** — Docker-based packaging ensures reproducibility and dependency isolation
- **Multi-interface** — CLI, REST API, and Python SDK for the same component
- **Differentiable** — First-class support for Jacobians, JVPs, and VJPs across component and network boundaries
- **Schema-validated** — Pydantic models define explicit input/output contracts
- **Language-agnostic** — Wrap Python, Julia, C++, [Fortran](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/examples/building-blocks/fortran.html), or any executable behind a thin Python API

## Ecosystem

- **[tesseract-core](https://github.com/pasteurlabs/tesseract-core)** — CLI, Python API, and runtime (this repo)
- **[Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax)** — Embed Tesseracts as JAX primitives into end-to-end differentiable JAX programs
- **[Tesseract-Streamlit](https://github.com/pasteurlabs/tesseract-streamlit)** — Auto-generate interactive web apps from Tesseracts

## Learn more

- [Documentation](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/)
- [Creating your first Tesseract](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/create.html)
- [Differentiable programming guide](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/differentiable-programming.html)
- [Design patterns](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/design-patterns.html)
- [Example gallery](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/examples/example_gallery.html)

## License

Tesseract Core is licensed under the [Apache License 2.0](https://github.com/pasteurlabs/tesseract-core/blob/main/LICENSE) and is free to use, modify, and distribute (under the terms of the license).

Tesseract is a registered trademark of Pasteur Labs, Inc. and may not be used without permission.
