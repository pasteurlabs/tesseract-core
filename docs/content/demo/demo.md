# Demos & Tutorials

End-to-end examples that send gradients through solvers written in other languages and frameworks, followed by showcases built with real engineering tools and smaller tutorials that teach the mechanics.

```{toctree}
:maxdepth: 1
:hidden:

enzyme-lfortran.ipynb
multiphysics-optimization.ipynb
learned-closure.ipynb
fem-shape-optimization.ipynb
data-assimilation.ipynb
lorenz_tesseract.md
bayesian-inference.ipynb
cfd-optimization.ipynb
JAX Rosenbrock Minimization <https://si-tesseract.discourse.group/t/jax-based-rosenbrock-function-minimization/48>
PyTorch Rosenbrock Minimization <https://si-tesseract.discourse.group/t/pytorch-based-rosenbrock-function-minimization/44>
JAX RBF Fitting <https://si-tesseract.discourse.group/t/jax-auto-diff-templates-gaussian-radial-basis-function-fitting/51>
```

## Gradients across language and framework boundaries

Each of these demos differentiates through a component that lives outside the program optimizing it, whether in another language, another framework, or a separately built solver.

(cards-clickable)=

::::{grid} 2
:gutter: 2

:::{grid-item-card} Differentiable Fortran (Enzyme)
:link: enzyme-lfortran
:link-type: doc

Solve two inverse heat-transfer problems by differentiating a Fortran solver end-to-end. Enzyme generates exact derivatives at the LLVM IR level, and `jax.value_and_grad` drives the optimization through Tesseract-JAX.
:::
:::{grid-item-card} Multi-Physics Optimization
:link: multiphysics-optimization
:link-type: doc

Couple two independently built thermal and structural Tesseracts with two-way thermoelastic feedback, and differentiate through the resulting equilibrium to solve an inverse-design problem, with constant-memory gradients via implicit differentiation.
:::
:::{grid-item-card} Learned Closure (PyTorch)
:link: learned-closure
:link-type: doc

Train a native PyTorch neural viscosity closure end-to-end through a Burgers' equation solver Tesseract, served in its own process and used as a differentiable layer. Gradients flow from the loss through the solver's VJP, over HTTP, into the network using Tesseract-Torch.
:::
:::{grid-item-card} FEM Shape Optimization
:link: fem-shape-optimization
:link-type: doc

Compose a geometry Tesseract (PyVista, finite-difference gradients) with a FEM Tesseract (jax-fem) to optimize structural bar configurations for minimum compliance.
:::

::::

## Showcases

Larger case studies built on commercial tools and other people's solvers.

::::{grid} 2
:gutter: 2

:::{grid-item-card} Rocket Grid Fin Optimization
:link: ../../blog/2025-11-28-rocket-fin-optimization
:link-type: doc

Optimize a rocket grid fin across Ansys SpaceClaim geometry, a mesher, and PyMAPDL structural analysis, with three differentiation strategies in one gradient.
:::
:::{grid-item-card} Tesseract Hackathon 2026
:link: ../../blog/2026-09-30-tesseract-hackathon-winners
:link-type: doc

Winning entries that differentiate through NOAA's Fortran hydrology models, Sentaurus TCAD, OpenSees, OpenMEEG, and coupled Julia and FEniCS solvers.
:::
:::{grid-item-card} Ansys Integration Gallery
:link: ../examples/ansys_gallery
:link-type: doc

Wrap Ansys SpaceClaim as a geometry engine served without a container, and the MAPDL solver as a differentiable Tesseract with an analytic adjoint.
:::

::::

## Learn the mechanics

Smaller tutorials on self-contained problems. The components here are easy to rewrite in JAX or PyTorch, so they are best read as walkthroughs of how Tesseracts, schemas, and gradient endpoints fit together, rather than as examples of when to use one (see {doc}`/content/concepts/when-to-use`).

::::{grid} 2
:gutter: 2

:::{grid-item-card} 4D-Var Data Assimilation
:link: data-assimilation
:link-type: doc

Full walkthrough of a 4D-Var scheme using a differentiable Lorenz-96 Tesseract, from building the Tesseract to running the optimization loop.
:::
:::{grid-item-card} Lorenz Tesseract
:link: lorenz_tesseract
:link-type: doc

Detailed implementation of the JAX-based Lorenz-96 solver Tesseract used in the 4D-Var demo.
:::
:::{grid-item-card} Bayesian Inference
:link: bayesian-inference
:link-type: doc

Use the same Lorenz-96 Tesseract as the forward model inside a NumPyro probabilistic workflow, and recover the posterior over an unknown forcing parameter with gradient-based MCMC.
:::
:::{grid-item-card} CFD Flow Optimization
:link: cfd-optimization
:link-type: doc

Optimize the initial velocity field of a 2D Navier-Stokes simulation so its vorticity evolves into a target image, with gradient-based optimization through a JAX-CFD Tesseract. JAX-CFD is no longer maintained upstream.
:::
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
