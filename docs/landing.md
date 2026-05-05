---
orphan: true
html_theme.sidebar_secondary.remove: true
sd_hide_title: true
html_class: landing-page
---

# Tesseract

::::{div} landing-hero

:::{div} landing-hero-logo

```{image} static/logo-light.png
:alt: Tesseract
:width: 180px
:class: landing-logo only-light
```

```{image} static/logo-dark.png
:alt: Tesseract
:width: 180px
:class: landing-logo only-dark
```

:::

:::{div} landing-hero-text

**End-to-end differentiable pipelines for scientific computing**

Tesseract packages any computational routine &mdash; in any language, with your choice of gradient implementation &mdash; into a differentiable containerized component.
Compose solvers, geometry ops, and ML models into pipelines with gradients flowing across every boundary.
Open source, published in [JOSS](https://doi.org/10.21105/joss.08385).

:::{div} landing-cta
{bdg-ref-primary-line}`Install <content/introduction/installation>`
{bdg-ref-primary-line}`Get Started <content/introduction/get-started>`
{bdg-ref-primary-line}`Demos <content/demo/demo>`
:::

:::

::::

:::{div} landing-divider
:::

## Why Tesseract

:::{div} section-intro
Tesseract is built for differentiable systems: solver-in-the-loop training, simulation-based inference, shape & topology optimization, learned closures, surrogate modeling, optimal control, ...
:::

::::{grid} 1 2 3 3
:gutter: 4

:::{grid-item-card} End-to-end gradients
:class-card: feature-card

Differentiate through your entire pipeline, no matter how gradients are computed.
Mix analytic adjoints, autodiff, and finite differences freely.
:::

:::{grid-item-card} Any language
:class-card: feature-card

Fortran, C++, Julia, JAX, PyTorch, or shell scripts.
Your code stays in its native language, Python is the glue.
:::

:::{grid-item-card} JAX native
:class-card: feature-card

Every Tesseract becomes a JAX primitive,
with full support for `grad`, `jit`, and `vmap`.
:::

:::{grid-item-card} Run anywhere
:class-card: feature-card

Share a Tesseract, get identical results — on any laptop, cloud, or HPC cluster.
No dependency conflicts, no version mismatches.
:::

:::{grid-item-card} Self-documenting
:class-card: feature-card

Schemas, types, and API docs are generated from your code.
Know exactly what any Tesseract expects and returns without reading its source.
:::

:::{grid-item-card} Community-driven
:class-card: feature-card

Created at [Pasteur Labs](https://pasteurlabs.ai), developed with and for the community. Open source under Apache License 2.0.
:::

::::

:::{div} landing-divider
:::

## How it works

:::{div} section-intro
Define a differentiable component in `tesseract_api.py`, build it into a
container, and call it — including its gradients — from the CLI, REST API,
or Python SDK. Compose multiple Tesseracts into end-to-end differentiable pipelines.
:::

:::::::{grid} 1 1 2 2
:gutter: 3

:::::{grid-item}
:class: howto-define

**Define a Tesseract**

```python
# tesseract_api.py
import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array
from tesseract_core.runtime import Differentiable
from tesseract_core.runtime import Float64

class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float64]]

class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float64]]

def apply(inputs: InputSchema) -> OutputSchema:
    # Replace with your FEM solver, mesh generator,
    # or neural surrogate
    return OutputSchema(y=inputs.x ** 2)

def jacobian(inputs: InputSchema, jac_inputs, jac_outputs):
    # Use autodiff, analytic gradients,
    # finite differences, ...
    return {"y": {"x": np.diag(2 * inputs.x)}}
```

:::::

:::::{grid-item}
**Use it**

::::{tab-set}
:::{tab-item} CLI

```bash
$ tesseract build .
 [i] Built image my-tesseract:latest

$ tesseract run my-tesseract apply \
    '{"inputs": {"x": [3.0]}}'
# => {"y": {"object_type": "array", "shape": [1], ..., "data": {"buffer": [9.0], ...}}}

$ tesseract run my-tesseract jacobian \
    '{"inputs": {"x": [3.0]}, "jac_inputs": ["x"], "jac_outputs": ["y"]}'
# => {"y": {"x": {"object_type": "array", "shape": [1, 1], ..., "data": {"buffer": [6.0], ...}}}}
```

:::
:::{tab-item} Python SDK

```python
from tesseract_core import Tesseract

with Tesseract.from_image("my-tesseract") as t:
    result = t.apply(inputs={"x": [3.0]})
    # result["y"] => array([9.0])

    jac = t.jacobian(
        inputs={"x": [3.0]},
        jac_inputs=["x"], jac_outputs=["y"],
    )
    # jac["y"]["x"] => array([[6.0]])
```

:::
:::{tab-item} JAX

```python
import jax
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

t = Tesseract.from_image("my-tesseract")
t.serve()

apply_jit = jax.jit(apply_tesseract, static_argnums=(0,))

out = apply_jit(t, {"x": x_array})
# out["y"] => Array([9.0])

jac_fn = jax.jacobian(
    lambda x: apply_jit(t, {"x": x})["y"]
)
# jac_fn(x_array) => Array([[6.0]])

t.teardown()
```

:::
::::

:::::

:::::::

:::{div} section-intro
The example above defines a differentiable Tesseract and calls it from the CLI, Python, and JAX.
Ready to build your own? The {doc}`Get Started <content/introduction/get-started>` tutorial walks you through a complete example from scratch.
:::

:::{div} landing-divider
:::

## Demos

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} 4D-Var Data Assimilation
:link: content/demo/data-assimilation
:link-type: doc
:class-card: demo-card
:img-top: static/demo-data-assimilation.svg
:class-img-top: demo-schematic

A complete 4D-Variational data assimilation scheme for a chaotic dynamical
system (Lorenz-96), built with a differentiable JAX Tesseract.
:::

:::{grid-item-card} CFD Flow Optimization
:link: content/demo/cfd-optimization
:link-type: doc
:class-card: demo-card
:img-top: static/demo-cfd.svg
:class-img-top: demo-schematic

Optimize initial conditions of a 2D Navier-Stokes simulation so the
vorticity evolves into a target image, via a JAX-CFD Tesseract.
:::

:::{grid-item-card} FEM Shape Optimization
:link: content/demo/fem-shape-optimization
:link-type: doc
:class-card: demo-card
:img-top: static/demo-fem-shapeopt.svg
:class-img-top: demo-schematic

Compose a geometry Tesseract with a FEM solver Tesseract for end-to-end
parametric structural optimization.
:::

::::

:::{div} landing-cta
{bdg-ref-primary-line}`All demos & tutorials <content/demo/demo>`
:::

:::{div} landing-divider
:::

## The Tesseract Ecosystem

:::{div} section-intro
Tesseract Core is the foundation. Additional packages extend its capabilities.
:::

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Tesseract Core
:link: content/introduction/index
:link-type: doc
:class-card: ecosystem-card

CLI, Python SDK, and container runtime for wrapping and running
differentiable components.
:::

:::{grid-item-card} Tesseract-JAX
:link: https://github.com/pasteurlabs/tesseract-jax
:class-card: ecosystem-card

Embed Tesseracts as JAX primitives. Fully compatible with `jit`, `vmap`,
and `grad`.
:::

:::{grid-item-card} Tesseract-Streamlit
:link: https://github.com/pasteurlabs/tesseract-streamlit
:class-card: ecosystem-card

Auto-generate interactive web apps from running Tesseracts. No frontend
code required.
:::

::::

:::{div} landing-divider
:::

## Get Involved

:::{div} section-intro
Tesseract is an open-source project. Build something cool with Tesseract and
[share it on the forum](https://si-tesseract.discourse.group/c/showcase/11) or contribute
bug reports, docs improvements, and feature proposals.
:::

:::{div} landing-cta
{bdg-link-primary-line}`Community Forum <https://si-tesseract.discourse.group/>`
{bdg-link-primary-line}`Contributing Guide <https://github.com/pasteurlabs/tesseract-core/blob/main/CONTRIBUTING.md>`
{bdg-link-primary-line}`Report an Issue <https://github.com/pasteurlabs/tesseract-core/issues>`
:::

::::{div} landing-footer

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item}
**Project**

- {doc}`Get Started <content/introduction/get-started>`
- {doc}`Installation <content/introduction/installation>`
- {doc}`API Reference <content/api/tesseract-api>`
- [JOSS Paper ↗](https://doi.org/10.21105/joss.08385)
- [Changelog ↗](https://github.com/pasteurlabs/tesseract-core/releases)
  :::

:::{grid-item}
**Community**

- [Forums ↗](https://si-tesseract.discourse.group/)
- [GitHub ↗](https://github.com/pasteurlabs/tesseract-core)
- [Contributing ↗](https://github.com/pasteurlabs/tesseract-core/blob/main/CONTRIBUTING.md)
- [Code of Conduct ↗](https://github.com/pasteurlabs/tesseract-core/blob/main/CODE_OF_CONDUCT.md)
  :::

:::{grid-item}
**About**

- Created at [Pasteur Labs ↗](https://pasteurlabs.ai)
- Open source — [Apache License ↗](https://github.com/pasteurlabs/tesseract-core/blob/main/LICENSE)
  :::

::::

::::
