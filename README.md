<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/pasteurlabs/tesseract-core/blob/main/docs/static/logo-dark.png" width="128" align="right">
  <img alt="" src="https://github.com/pasteurlabs/tesseract-core/blob/main/docs/static/logo-light.png" width="128" align="right">
</picture>

### Tesseract Core

Universal components for differentiable scientific computing 📦

[Read the docs](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/) |
[Demos & tutorials](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/demo/demo/) |
[Blog](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/blog/) |
[Report an issue](https://github.com/pasteurlabs/tesseract-core/issues) |
[Contribute](https://github.com/pasteurlabs/tesseract-core/blob/main/CONTRIBUTING.md)

---

[![DOI](https://joss.theoj.org/papers/10.21105/joss.08385/status.svg)](https://doi.org/10.21105/joss.08385)
[![SciPy](https://img.shields.io/badge/SciPy-2025-blue)](https://proceedings.scipy.org/articles/kvfm5762)

## The problem

**Real-world scientific workflows span multiple tools, languages, and computing environments.** You might have a mesh generator in C++, a solver in Julia, and post-processing in Python. Getting these to work together is painful. Getting gradients to flow through them for optimization is nearly impossible.

Existing autodiff frameworks work great within a single codebase, but fall short when your pipeline crosses framework boundaries or includes legacy tools.

## The solution

Tesseract packages scientific software into **self-contained, portable components** that:

- **Run anywhere** — Local machines, cloud, HPC clusters. Same container, same results.
- **Expose clean interfaces** — CLI, REST API, and Python SDK. No more deciphering undocumented scripts.
- **Propagate gradients** — Each component can expose derivatives, enabling end-to-end optimization across heterogeneous pipelines.
- **Self-document** — Schemas, types, and API docs are generated automatically.

## Who is this for?

- **Researchers** interfacing with (differentiable) simulators or probabilistic models, or who need to combine tools from different ecosystems.
- **R&D engineers** packaging research code for use by others, without spending weeks on DevOps.

## Example: Shape optimization across tools

<a href="https://docs.pasteurlabs.ai/projects/tesseract-core/latest/blog/2025-11-28-rocket-fin-optimization/">
<img src="https://github.com/pasteurlabs/tesseract-core/blob/main/docs/img/grid_fin_stl.png" width="200" align="right" alt="Rocket grid fin geometry optimized by a differentiable Tesseract pipeline" title="Rocket grid fin optimized end-to-end across SpaceClaim, a mesher, and PyMAPDL.">
</a>

The [rocket fin optimization case study](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/blog/2025-11-28-rocket-fin-optimization/) combines three Tesseracts:

```
[SpaceClaim geometry] → [Mesh + SDF] → [PyMAPDL FEA solver]
         ↑                                      |
         └──────── gradients flow back ─────────┘
```

Each component uses a different differentiation strategy (analytic adjoints, finite differences, JAX autodiff), yet they compose into a single optimizable pipeline that [is one `jax.grad` call away](https://github.com/pasteurlabs/tesseract-jax) from end-to-end gradients.

> [!TIP]
> More examples in the [demos](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/demo/demo/) and the [example gallery](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/examples/example_gallery.html).

## Quick start

<p align="center">
<img src="https://github.com/pasteurlabs/tesseract-core/blob/main/docs/img/demo.gif" width="720" alt="Demo: install, build, and run a Tesseract in under a minute">
<br>
<em>Getting started: install, build an example, and run it.</em>
</p>

> [!NOTE]
> Requires Python 3.10+. Building container images, as in this example, also requires [Docker](https://docs.docker.com/engine/install/). To use Tesseracts without Docker, see [running without containers](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/how-to/advanced-usage/#running-without-containers).

**CLI:**

```bash
# Install Tesseract Core
$ pip install tesseract-core

# Create a new project in the current directory
$ tesseract init --name my-tesseract

# Edit `tesseract_api.py`, or download an example
$ curl -so ./tesseract_api.py https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/examples/vectoradd/tesseract_api.py

# Build it into a container
$ tesseract build .

# Run it
$ tesseract run my-tesseract apply '{"inputs": {"a": [1, 2, 3], "b": [10, 20, 30]}}'
# → {"result": [11, 22, 33]}

# Compute the Jacobian
$ tesseract run my-tesseract jacobian '{"inputs": {"a": [1, 2, 3], "b": [10, 20, 30]}, "jac_inputs": ["a"], "jac_outputs": ["result"]}'
# → {"result": {"a": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}
```

**Python SDK:**

```python
from tesseract_core import Tesseract

with Tesseract.from_image("my-tesseract") as t:
    result = t.apply({"a": [1, 2, 3], "b": [10, 20, 30]})
    jac = t.jacobian({"a": [1, 2, 3], "b": [10, 20, 30]}, jac_inputs=["a"], jac_outputs=["result"])
```

## Core features

- **Containerized** — Docker-based packaging ensures reproducibility and dependency isolation.
- **Multi-interface** — Use the same components via CLI, REST API, and Python SDK.
- **Differentiable** — First-class support for Jacobians, JVPs, and VJPs across component and network boundaries.
- **Schema-validated** — Pydantic models define explicit input/output contracts.
- **Language-agnostic** — Wrap Python, Julia, C++, [Fortran](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/examples/building-blocks/fortran.html), or any executable behind a thin Python API.
- **Self-documenting** — Auto-generated API docs and schemas for every Tesseract (`tesseract apidoc <name>`).

<p align="center">
<img src="https://github.com/pasteurlabs/tesseract-core/blob/main/docs/img/apidoc-screenshot.png" width="600" alt="Auto-generated API documentation for a Tesseract">
<br>
<em>Auto-generated API documentation (<code>tesseract apidoc</code>).</em>
</p>

## The Ecosystem

- **[Tesseract Core](https://github.com/pasteurlabs/tesseract-core)** — CLI, Python SDK, and runtime (this repo).
- **[Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax)** — Embed Tesseracts as JAX primitives into end-to-end differentiable JAX programs.
- **[Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch)** — Embed Tesseracts as PyTorch operators into end-to-end differentiable PyTorch programs.
- **[Tesseract-Streamlit](https://github.com/pasteurlabs/tesseract-streamlit)** — Auto-generate interactive web apps from Tesseracts.

## Learn more

- [Documentation](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/)
- [Creating your first Tesseract](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/tutorials/create.html)
- [Differentiable programming guide](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/concepts/differentiable-programming.html)
- [Design patterns](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/concepts/design-patterns.html)
- [Example gallery](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/examples/example_gallery.html)

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

## License

Tesseract Core is licensed under the [Apache License 2.0](https://github.com/pasteurlabs/tesseract-core/blob/main/LICENSE) and is free to use, modify, and distribute (under the terms of the license).

Tesseract is a registered trademark of Pasteur Labs, Inc. and may not be used without permission.
