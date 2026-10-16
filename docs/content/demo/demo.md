# Demos & Tutorials

End-to-end demos, case studies with real engineering tools, and smaller tutorials that teach the mechanics.

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

## Realistic use cases

(cards-clickable)=

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Differentiable Fortran (Enzyme)
:link: enzyme-lfortran
:link-type: doc
:class-card: demo-card
:img-top: /static/demo-enzyme-lfortran.svg
:class-img-top: demo-schematic invert-on-dark

Solve inverse heat-transfer problems through a Fortran solver differentiated by Enzyme.
:::

:::{grid-item-card} Multi-Physics Optimization
:link: multiphysics-optimization
:link-type: doc
:class-card: demo-card
:img-top: /static/demo-multiphysics.svg
:class-img-top: demo-schematic invert-on-dark

Differentiate through the equilibrium of coupled thermal and structural Tesseracts.
:::

:::{grid-item-card} Learned Closure (PyTorch)
:link: learned-closure
:link-type: doc
:class-card: demo-card
:img-top: /static/demo-learned-closure.svg
:class-img-top: demo-schematic invert-on-dark

Train a PyTorch closure model through a Burgers' solver running in its own process.
:::

:::{grid-item-card} FEM Shape Optimization
:link: fem-shape-optimization
:link-type: doc
:class-card: demo-card
:img-top: /static/demo-fem-shapeopt.svg
:class-img-top: demo-schematic invert-on-dark

Combine finite-difference geometry gradients with a jax-fem solver to minimize compliance.
:::

::::

## Case studies

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Rocket Grid Fin Optimization
:link: ../../blog/2025-11-28-rocket-fin-optimization
:link-type: doc
:class-card: demo-card
:img-top: /img/gallery/rocket-fin.jpg

Ansys SpaceClaim, a mesher, and PyMAPDL in one gradient, with three differentiation strategies.
:::

:::{grid-item-card} Tesseract Hackathon 2026
:link: ../../blog/2026-09-30-tesseract-hackathon-winners
:link-type: doc
:class-card: demo-card
:img-top: /img/gallery/dsnowsac.jpg

Winning entries, from NOAA's Fortran hydrology models to Sentaurus TCAD.
:::

:::{grid-item-card} Ansys Integration Gallery
:link: ../examples/ansys_gallery
:link-type: doc
:class-card: demo-card
:img-top: /img/pymapdl_simp_compliance.png

SpaceClaim served without a container, and MAPDL with an analytic adjoint.
:::

::::

## Learn the mechanics

These components are easy to rewrite in JAX or PyTorch, so read them as walkthroughs of the mechanics, not as examples of when to use Tesseract (see {doc}`/content/concepts/when-to-use`).

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} 4D-Var Data Assimilation
:link: data-assimilation
:link-type: doc
:class-card: demo-card
:img-top: /static/demo-data-assimilation.svg
:class-img-top: demo-schematic invert-on-dark

Build a differentiable Lorenz-96 Tesseract and run 4D-Var through it.
:::

:::{grid-item-card} Bayesian Inference
:link: bayesian-inference
:link-type: doc
:class-card: demo-card
:img-top: /static/demo-bayesian-inference.svg
:class-img-top: demo-schematic invert-on-dark

Recover a posterior with gradient-based MCMC in NumPyro, using the same Lorenz-96 Tesseract.
:::

:::{grid-item-card} CFD Flow Optimization
:link: cfd-optimization
:link-type: doc
:class-card: demo-card
:img-top: /static/demo-cfd.svg
:class-img-top: demo-schematic invert-on-dark

Optimize a 2D Navier-Stokes initial condition through JAX-CFD, which is no longer maintained upstream.
:::

::::

More walkthroughs:

- {doc}`lorenz_tesseract`, the implementation behind the 4D-Var and Bayesian inference demos
- [JAX Rosenbrock minimization](https://si-tesseract.discourse.group/t/jax-based-rosenbrock-function-minimization/48)
- [PyTorch Rosenbrock minimization](https://si-tesseract.discourse.group/t/pytorch-based-rosenbrock-function-minimization/44)
- [JAX RBF fitting](https://si-tesseract.discourse.group/t/jax-auto-diff-templates-gaussian-radial-basis-function-fitting/51)
