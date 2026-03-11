# Demos & Tutorials

End-to-end examples that show Tesseracts in action — from optimization workflows to data assimilation.

```{toctree}
:maxdepth: 1
:hidden:

data-assimilation-4dvar.ipynb
lorenz_tesseract.md
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
:link: data-assimilation-4dvar.html

Full walkthrough of a 4D-Var scheme using a differentiable Lorenz-96 Tesseract — from building the Tesseract to running the optimization loop.
:::
:::{grid-item-card} Lorenz Tesseract
:link: lorenz_tesseract.html

Detailed implementation of the JAX-based Lorenz-96 solver Tesseract used in the 4D-Var demo.
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
