# Demos & Tutorials

End-to-end examples that show Tesseracts in action — from optimization workflows to data assimilation.

```{toctree}
:maxdepth: 1
:hidden:

data-assimilation.ipynb
lorenz_tesseract.md
cfd-optimization.ipynb
fem-shape-optimization.ipynb
JAX Rosenbrock Minimization <https://si-tesseract.discourse.group/t/jax-based-rosenbrock-function-minimization/48>
PyTorch Rosenbrock Minimization <https://si-tesseract.discourse.group/t/pytorch-based-rosenbrock-function-minimization/44>
JAX RBF Fitting <https://si-tesseract.discourse.group/t/jax-auto-diff-templates-gaussian-radial-basis-function-fitting/51>
```

```{tip}
For more community-contributed examples, check out the [Tesseract Showcase](https://si-tesseract.discourse.group/c/showcase/11) on the forum.
```

## Data assimilation demo

A complete 4D-Variational data assimilation scheme for a chaotic dynamical system, built with differentiable Tesseracts.

(cards-clickable)=

::::{grid} 2
:gutter: 2

:::{grid-item-card} 4D-Var Data Assimilation
:link: data-assimilation.html

Full walkthrough of a 4D-Var scheme using a differentiable Lorenz-96 Tesseract — from building the Tesseract to running the optimization loop.
:::
:::{grid-item-card} Lorenz Tesseract
:link: lorenz_tesseract.html

Detailed implementation of the JAX-based Lorenz-96 solver Tesseract used in the 4D-Var demo.
:::

::::

## Simulation & design optimization demos

End-to-end differentiable optimization through physics simulators, using Tesseract-JAX to compose Tesseracts with JAX code.

::::{grid} 2
:gutter: 2

:::{grid-item-card} CFD Flow Optimization
:link: cfd-optimization.html

Optimize the initial velocity field of a 2D Navier-Stokes simulation so its vorticity evolves into a target image — gradient-based optimization through a JAX-CFD Tesseract.
:::
:::{grid-item-card} FEM Shape Optimization
:link: fem-shape-optimization.html

Compose a geometry Tesseract (PyVista, finite-difference gradients) with a FEM Tesseract (jax-fem) to optimize structural bar configurations for minimum compliance.
:::

::::

## Optimization tutorials

These tutorials walk through complete optimization workflows using Tesseracts with different autodiff frameworks:

::::{grid} 2
:gutter: 2

:::{grid-item-card} JAX Rosenbrock Minimization
:link: https://si-tesseract.discourse.group/t/jax-based-rosenbrock-function-minimization/48

End-to-end function minimization using JAX autodiff with Tesseract-JAX.
:::
:::{grid-item-card} PyTorch Rosenbrock Minimization
:link: https://si-tesseract.discourse.group/t/pytorch-based-rosenbrock-function-minimization/44

End-to-end function minimization using PyTorch autodiff.
:::
:::{grid-item-card} JAX RBF Fitting
:link: https://si-tesseract.discourse.group/t/jax-auto-diff-templates-gaussian-radial-basis-function-fitting/51

Gaussian radial basis function fitting with JAX automatic differentiation.
:::

::::
