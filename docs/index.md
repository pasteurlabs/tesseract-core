---
orphan: true
html_theme.sidebar_secondary.remove: true
sd_hide_title: true
html_class: landing-page
og:title: "Tesseract — Universal components for differentiable scientific computing"
og:description: "Train and optimize JAX and PyTorch programs through Fortran, Julia, C++, and commercial solvers. Each component runs in its own process behind a typed, differentiable interface. Open source."
---

# Tesseract

::::::{div} landing-hero

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

:::::{div} landing-hero-text

:::{div} landing-hero-tagline
Universal components for differentiable scientific computing
:::

:::{div} landing-hero-subtitle
Train and optimize through a Fortran code, a Julia package, a FEniCS model, or a commercial tool from JAX and PyTorch.
Tesseract wraps each component behind a typed interface with derivative endpoints and runs it in its own process, no Docker required.
Open source, published in [JOSS](https://doi.org/10.21105/joss.08385).
:::

:::{div} landing-cta
{bdg-ref-primary-line}`Get Started <content/tutorials/get-started>`
{bdg-ref-primary-line}`Demos <content/demo/demo>`
{bdg-link-primary-line}`GitHub <https://github.com/pasteurlabs/tesseract-core>`
:::

:::::

::::::

:::{div} landing-divider
:::

## What you can do with it

:::{div} section-intro
Tesseract is for differentiable programs that need to reach outside their own framework: solver-in-the-loop training, calibration and inverse problems, simulation-based inference, and shape optimization through real engineering tools.
:::

::::{grid} 1 2 3 3
:gutter: 4

:::{grid-item-card} Train through a solver in another stack
:class-card: feature-card

Put a Fortran, Julia, C++, or FEniCS solver inside a PyTorch or JAX training loop.
Gradients reach the network's weights through the solver's own VJP.
:::

:::{grid-item-card} Optimize through tools you can't replace
:class-card: feature-card

Validated legacy codes, commercial CAD and FEA, closed-source binaries.
Adjoints, autodiff, and finite differences combine in one gradient.
:::

:::{grid-item-card} Check your gradients
:class-card: feature-card

`check-gradients` compares every derivative endpoint against finite differences,
so a wrong VJP is caught before an optimizer depends on it.
:::

:::{grid-item-card} Keep environments apart
:class-card: feature-card

Each component runs in its own process with its own dependencies,
so conflicting stacks never have to share an interpreter.
:::

:::{grid-item-card} Call it from JAX and PyTorch
:class-card: feature-card

Every Tesseract becomes a JAX primitive or a PyTorch operator,
with gradients flowing through `jax.grad` and `torch.autograd`.
:::

:::{grid-item-card} Hand it to someone else
:class-card: feature-card

Schemas and API docs are generated from your code. Build a container image
when the component needs to run on another machine or in the cloud.
:::

::::

:::{div} landing-divider
:::

## How it works

:::{div} section-intro
Define a differentiable component in `tesseract_api.py`, run it in its own process,
and call it, including its gradients, from Python, JAX, or PyTorch.
When it needs to run elsewhere, the same file builds into a container image.
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
    Array, Differentiable, Float32, ShapeDType
)

class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float32]]

class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float32]]

def apply(inputs: InputSchema) -> OutputSchema:
    # Call your solver, mesh generator,
    # or surrogate here
    return OutputSchema(y=inputs.x**2)

def vector_jacobian_product(
    inputs, vjp_inputs, vjp_outputs, cotangent_vector
):
    # Hand-written, autodiff, adjoint solve,
    # finite differences, ...
    return {"x": 2 * inputs.x * cotangent_vector["y"]}

def abstract_eval(abstract_inputs):
    # Output shapes from input shapes
    return {"y": ShapeDType(
        shape=abstract_inputs.x.shape, dtype="float32"
    )}
```

:::::

:::::{grid-item}
**Use it**

::::{tab-set}
:::{tab-item} Python SDK

```python
from tesseract_core import Tesseract

with Tesseract.from_source("tesseract_api.py") as t:
    result = t.apply({"x": [3.0]})
    # result["y"] => array([9.])

    vjp = t.vector_jacobian_product(
        {"x": [3.0]},
        vjp_inputs=["x"], vjp_outputs=["y"],
        cotangent_vector={"y": [1.0]},
    )
    # vjp["x"] => array([6.])
```

:::
:::{tab-item} JAX

```python
import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

with Tesseract.from_source("tesseract_api.py") as t:
    f = lambda x: apply_tesseract(t, {"x": x})["y"].sum()

    jax.jit(f)(jnp.array([3.0]))
    # => Array(9.)
    jax.grad(f)(jnp.array([3.0]))
    # => [6.]
```

:::
:::{tab-item} PyTorch

```python
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

with Tesseract.from_source("tesseract_api.py") as t:
    x = torch.tensor([3.0], requires_grad=True)

    out = apply_tesseract(t, {"x": x})
    # out["y"] => tensor([9.])

    out["y"].sum().backward()
    # x.grad => tensor([6.])  (via the VJP endpoint)
```

:::
:::{tab-item} Check & ship

```bash
# Compare the VJP against finite differences
$ TESSERACT_API_PATH=tesseract_api.py \
    tesseract-runtime check-gradients \
    '{"inputs": {"x": [1.0, 2.0, 3.0]}}'
✅ Gradient check for vector_jacobian_product passed ✅

# Build a container image (requires Docker)
$ tesseract build .
$ tesseract run my-tesseract apply \
    '{"inputs": {"x": [3.0]}}'
```

:::
::::

:::::

:::::::

:::{div} section-intro
Ready to build your own? The {doc}`Get Started <content/tutorials/get-started>` tutorial walks you through a complete example from scratch,
and {doc}`content/concepts/when-to-use` helps you decide whether your problem needs a Tesseract at all.
:::

:::{div} landing-divider
:::

## Built with Tesseract

:::{div} section-intro
Each caption names the boundary the gradient crosses. Most of these projects are entries to the
[2026 Tesseract Hackathon](blog/2026-09-30-tesseract-hackathon-winners).
:::

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} PRISMO
:link: https://github.com/benvial/prismo
:class-card: demo-card
:img-top: img/gallery/prismo.jpg

A Julia semiconductor solver and a FEniCS optics solver in one JAX gradient,
reaching more than five times the optical phase shift of the starting design.
:::

:::{grid-item-card} δSnowSac
:link: https://github.com/kasProg/dSnowSac
:class-card: demo-card
:img-top: img/gallery/dsnowsac.jpg

A PyTorch network learns to calibrate NOAA's operational Fortran models,
Snow-17 and SAC-SMA, with gradients flowing through the untouched Fortran.
:::

:::{grid-item-card} Rocket grid fins
:link: blog/2025-11-28-rocket-fin-optimization
:link-type: doc
:class-card: demo-card
:img-top: img/gallery/rocket-fin.jpg

Ansys SpaceClaim geometry and PyMAPDL structural analysis in one optimization loop,
giving a design 24% stiffer than the baseline at the same mass.
:::

:::{grid-item-card} Differentiable Silicon
:link: https://github.com/hozaifa1/differentiable-silicon
:class-card: demo-card
:img-top: img/gallery/differentiable-silicon.jpg

Sentaurus TCAD, a closed-source binary driven over SSH, differentiated with
finite differences and Broyden updates to tune a transistor for a spiking network.
:::

:::{grid-item-card} Coldplate
:link: https://github.com/TAUIL-Abd-Elilah/coldplate
:class-card: demo-card
:img-top: img/gallery/coldplate.jpg

A C++ fluid solver coupled both ways to a thermal solver in Fortran (via Enzyme) or JAX.
Leaving the coupling out of the gradient flips a third of its signs.
:::

:::{grid-item-card} NeuroLocate
:link: https://github.com/BrainCapture/NeuroLocate
:class-card: demo-card
:img-top: img/gallery/neurolocate.jpg

A PyTorch network proposes EEG sources, and JAX refines them with gradients
through OpenMEEG, a C++ boundary element solver with no autodiff of its own.
:::

::::

:::{div} landing-divider
:::

## Demos

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Differentiable Fortran (Enzyme)
:link: content/demo/enzyme-lfortran
:link-type: doc
:class-card: demo-card
:img-top: static/demo-enzyme-lfortran.svg
:class-img-top: demo-schematic invert-on-dark

Differentiate a Fortran heat-conduction solver end-to-end with Enzyme at the
LLVM IR level, and solve an inverse problem through Tesseract-JAX.
:::

:::{grid-item-card} Multi-Physics Optimization
:link: content/demo/multiphysics-optimization
:link-type: doc
:class-card: demo-card
:img-top: static/demo-multiphysics.svg
:class-img-top: demo-schematic invert-on-dark

Couple independently built thermal and structural Tesseracts with two-way
thermoelastic feedback, and differentiate through the resulting equilibrium
to solve an inverse-design problem.
:::

:::{grid-item-card} Learned Closure (PyTorch)
:link: content/demo/learned-closure
:link-type: doc
:class-card: demo-card
:img-top: static/demo-learned-closure.svg
:class-img-top: demo-schematic invert-on-dark

Train a neural viscosity closure end-to-end through a Burgers' equation
solver, with PyTorch gradients flowing through both via Tesseract-Torch.
:::

:::{grid-item-card} FEM Shape Optimization
:link: content/demo/fem-shape-optimization
:link-type: doc
:class-card: demo-card
:img-top: static/demo-fem-shapeopt.svg
:class-img-top: demo-schematic invert-on-dark

Compose a geometry Tesseract with a FEM solver Tesseract for end-to-end
parametric structural optimization.
:::

:::{grid-item-card} 4D-Var Data Assimilation
:link: content/demo/data-assimilation
:link-type: doc
:class-card: demo-card
:img-top: static/demo-data-assimilation.svg
:class-img-top: demo-schematic invert-on-dark

A complete 4D-Variational data assimilation scheme for a chaotic dynamical
system (Lorenz-96), built with a differentiable JAX Tesseract.
:::

:::{grid-item-card} Bayesian Inference
:link: content/demo/bayesian-inference
:link-type: doc
:class-card: demo-card
:img-top: static/demo-bayesian-inference.svg
:class-img-top: demo-schematic invert-on-dark

Use a Lorenz-96 Tesseract as the forward model in a NumPyro workflow and
recover the posterior over an unknown forcing parameter with gradient-based MCMC.
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

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Tesseract Core
:link: content/introduction/index
:link-type: doc
:class-card: ecosystem-card

CLI, Python SDK, and runtime for wrapping differentiable components
and running them as subprocesses, containers, or remote services.
:::

:::{grid-item-card} Tesseract-JAX
:link: https://github.com/pasteurlabs/tesseract-jax
:class-card: ecosystem-card

Embed Tesseracts as JAX primitives. Fully compatible with `jit`, `vmap`,
and `grad`.
:::

:::{grid-item-card} Tesseract-Torch
:link: https://github.com/pasteurlabs/tesseract-torch
:class-card: ecosystem-card

Embed Tesseracts as PyTorch operators. Gradients flow through with
`torch.autograd`.
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
Wrap your solver or model as a Tesseract, or compose existing ones into a new pipeline.
[Show us what you built](https://si-tesseract.discourse.group/c/showcase/11), or help improve the project.
:::

:::{div} landing-cta
{bdg-link-primary-line}`GitHub <https://github.com/pasteurlabs/tesseract-core>`
{bdg-ref-primary-line}`Blog <blog/index>`
{bdg-ref-primary-line}`Example Gallery <content/examples/example_gallery>`
:::

::::{div} landing-footer

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item}
**Project**

- {doc}`Get Started <content/tutorials/get-started>`
- {doc}`Installation <content/introduction/installation>`
- {doc}`API Reference <content/reference/tesseract-api>`
- [JOSS Paper](https://doi.org/10.21105/joss.08385)
- [Changelog](https://github.com/pasteurlabs/tesseract-core/releases)
  :::

:::{grid-item}
**Community**

- [Forums](https://si-tesseract.discourse.group/)
- {doc}`Blog <blog/index>`
- [GitHub](https://github.com/pasteurlabs/tesseract-core)
- [Contributing](https://github.com/pasteurlabs/tesseract-core/blob/main/CONTRIBUTING.md)
- [Code of Conduct](https://github.com/pasteurlabs/tesseract-core/blob/main/CODE_OF_CONDUCT.md)
- [Report an Issue](https://github.com/pasteurlabs/tesseract-core/issues)
  :::

:::{grid-item}
**About**

- Created at [Pasteur Labs](https://pasteurlabs.ai)
- Open source — [Apache License](https://github.com/pasteurlabs/tesseract-core/blob/main/LICENSE)
  :::

::::

::::
