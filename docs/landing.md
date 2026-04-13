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

**Universal components for differentiable scientific computing**

Scientific simulators are hard to share, hard to compose, and hard to differentiate through.
Tesseract packages any operation &mdash; like simulators, preprocessors, and other computational routines &mdash; into a portable, self-documenting component with gradients built in.
Open source under the Apache License, published in [JOSS](https://doi.org/10.21105/joss.08385).

:::{div} landing-cta
{bdg-ref-primary-line}`Get Started <content/introduction/get-started>`
{bdg-ref-primary-line}`Install <content/introduction/installation>`
{bdg-link-primary-line}`GitHub <https://github.com/pasteurlabs/tesseract-core>`
:::

:::

::::

:::{div} landing-divider
:::

## Why Tesseract

:::{div} section-intro
Tesseract grew out of the need to compose (differentiable) research software into end-to-end pipelines, where each has its own language, framework, and
differentiation strategy.
:::

::::{grid} 1 2 3 3
:gutter: 4

:::{grid-item-card} Run anywhere
:class-card: feature-card

Same container, same results: laptop, cloud, or HPC cluster.
No dependency conflicts, no version mismatches.
:::

:::{grid-item-card} End-to-end gradients
:class-card: feature-card

Differentiate across heterogeneous pipelines, even through
black-box solvers. Mix autodiff, analytic adjoints, and finite
differences freely.
:::

:::{grid-item-card} Any language
:class-card: feature-card

Fortran, C++, Julia, JAX, PyTorch, or shell scripts.
Python is the interface layer; your solver stays in its native language.
:::

:::{grid-item-card} JAX native
:class-card: feature-card

Every Tesseract becomes a JAX primitive,
with full support for `grad`, `jit`, and `vmap`.
:::

:::{grid-item-card} Self-documenting
:class-card: feature-card

Schemas, types, and API docs are generated from your code.
Inspect any Tesseract without reading its source.
:::

:::{grid-item-card} Community-driven
:class-card: feature-card

Created at [Pasteur Labs](https://pasteurlabs.ai), developed with and for the community.
Apache licensed. No vendor lock-in, no proprietary dependencies.
:::

::::

:::{div} landing-divider
:::

## How it works

:::{div} section-intro
Define a differentiable component in `tesseract_api.py`, build it into a
container, and call it, including its gradients, from the CLI, REST API,
or Python SDK.
:::

:::::::{grid} 1 1 2 2
:gutter: 3

:::::{grid-item}
:class: howto-define

**Define a Tesseract**

```python
# tesseract_api.py
from pydantic import BaseModel
from tesseract_core.runtime import (
    Array, Differentiable, Float64,
)

class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float64]]

class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float64]]

def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(y=inputs.x ** 2)
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
# => {"y": {"shape": [1], "data": [9.0], ...}}

$ tesseract run my-tesseract jacobian \
    '{"inputs": {"x": [3.0]}}'
# => {"y": {"x": {"shape": [1, 1], "data": [[6.0]], ...}}}
```

:::
:::{tab-item} Python SDK

```python
from tesseract_core import Tesseract

t = Tesseract.from_image("my-tesseract")
result = t.apply(inputs={"x": [3.0]})
# result["y"] => array([9.0])

jac = t.jacobian(inputs={"x": [3.0]})
# jac["y"]["x"] => array([[6.0]])
```

:::
:::{tab-item} JAX

```python
from tesseract_jax import apply_tesseract

out = apply_tesseract(t, {"x": x_array})
# out["y"] => Array([9.0])

grad_fn = jax.grad(
    lambda x: apply_tesseract(t, {"x": x})["y"]
)
# grad_fn({"x": x_array}) => Array([6.0])
```

:::
::::

:::::

:::::::

:::{div} landing-divider
:::

## Demos

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Bayesian Inference
:link: content/demo/bayesian-inference
:link-type: doc
:class-card: demo-card
:img-top: \_static/demo-bayesian.svg
:class-img-top: demo-schematic

Use a Tesseract as the forward model inside a NumPyro probabilistic
programming workflow, with full posterior inference over simulator parameters.
:::

:::{grid-item-card} Learned Closures
:link: content/demo/learned-closure
:link-type: doc
:class-card: demo-card
:img-top: \_static/demo-learned-closure.svg
:class-img-top: demo-schematic

Train a neural network end-to-end _through_ a PDE solver, with gradients
flowing across two independent Tesseracts.
:::

:::{grid-item-card} Multiphysics Optimization
:link: content/demo/multiphysics-optimization
:link-type: doc
:class-card: demo-card
:img-top: \_static/demo-multiphysics.svg
:class-img-top: demo-schematic

Couple a thermal solver and a structural solver, then optimize design
parameters with gradients across both.
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

CLI, Python SDK, and container runtime. The building blocks for creating
and running Tesseracts.
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
Tesseract is an open-source project and we welcome contributions of all kinds:
bug reports, new Tesseract implementations, documentation improvements, or
feature proposals.
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
- [JOSS Paper](https://doi.org/10.21105/joss.08385)
- [Changelog](https://github.com/pasteurlabs/tesseract-core/releases)
  :::

:::{grid-item}
**Community**

- [Forums](https://si-tesseract.discourse.group/)
- [GitHub](https://github.com/pasteurlabs/tesseract-core)
- [Contributing](https://github.com/pasteurlabs/tesseract-core/blob/main/CONTRIBUTING.md)
- [Code of Conduct](https://github.com/pasteurlabs/tesseract-core/blob/main/CODE_OF_CONDUCT.md)
  :::

:::{grid-item}
**About**

- Created at [Pasteur Labs](https://pasteurlabs.ai)
- Open source — [Apache License](https://github.com/pasteurlabs/tesseract-core/blob/main/LICENSE)
  :::

::::

::::
