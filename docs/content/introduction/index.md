# Tesseract Core Documentation

Tesseract packages scientific software into portable, self-documenting, differentiable components.
New here? Start with the [Get Started](get-started.md) tutorial or the [Installation](installation.md) guide.

## How it works

```{figure} ../../img/tesseract-interfaces.png
:alt: Tesseract interfaces
:width: 250px
:align: right

<small>Internal and external interfaces of a Tesseract.</small>
```

Every Tesseract has a primary entrypoint, `apply`, which wraps a software functionality of your choice. All other [endpoints](../api/endpoints.md) relate to this entrypoint: `abstract_eval` returns output structure, `jacobian` computes derivatives, and so on.

There are several ways to interact with Tesseracts:

1. **Define** entrypoints in `tesseract_api.py`
2. **Build** a container with `tesseract build`
3. **Serve** via HTTP with `tesseract serve`
4. **Invoke** via CLI, HTTP, or the [Python SDK](../api/tesseract-api.md)

## Features and limitations

::::{tab-set}
:::{tab-item} Features

- **Self-documenting** — Tesseracts announce their interfaces, so users can inspect them without reading source code and perform static validation without running the code.
- **Auto-validating** — Input data is automatically validated against the schema, so internal logic can assume the data is in the expected format.
- **Autodiff-native** — Tesseracts support [differentiable programming](../misc/differentiable-programming.md) and integrate as native operations in PyTorch and JAX — but exposing derivatives is _strictly optional_.
- **Batteries included** — Every Tesseract ships with a containerized runtime, a CLI, a REST API, and a Python SDK.

:::
:::{tab-item} Limitations

- **Python as glue** — Tesseracts may use any software under the hood, but they always use Python as glue between the runtime and the wrapped functionality. Support for Python projects is more mature than other languages.
- **Single entrypoint** — Each Tesseract has a single `apply` entrypoint. To expose N functions, create N Tesseracts.
- **Context-free** — Tesseracts are not aware of outer-loop orchestration or runtime details.
- **Runtime overhead** — Tesseracts are designed for compute kernels that run at least several seconds, so they may not suit very low-latency workloads.

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
get-started.md
Tesseract User Forums <https://si-tesseract.discourse.group/>
Changelog <https://github.com/pasteurlabs/tesseract-core/releases>
```

```{toctree}
:caption: Creating Tesseracts
:maxdepth: 2
:hidden:

../creating-tesseracts/create.md
../creating-tesseracts/design-patterns.md
../creating-tesseracts/advanced.md
../creating-tesseracts/llm-assistance.md
../creating-tesseracts/deploy.md
```

```{toctree}
:caption: Using Tesseracts
:maxdepth: 2
:hidden:

../using-tesseracts/use.md
../using-tesseracts/array-encodings.md
../using-tesseracts/advanced.md
```

```{toctree}
:caption: Demos & Tutorials
:maxdepth: 2
:hidden:

../demo/demo.md
../examples/example_gallery.md
../examples/ansys_gallery.md
```

```{toctree}
:caption: Guides
:maxdepth: 2
:hidden:

../misc/differentiable-programming.md
../misc/debugging.md
../misc/performance.md
```

```{toctree}
:caption: API Reference — SDK
:maxdepth: 2
:hidden:

../api/tesseract-cli.md
../api/tesseract-api.md
../api/config.md
```

```{toctree}
:caption: API Reference — Runtime
:maxdepth: 2
:hidden:

../api/endpoints.md
../api/tesseract-runtime-cli.md
../api/tesseract-runtime-api.md
```
