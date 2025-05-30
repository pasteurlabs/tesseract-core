# Interoperable Autodiff Engines, aka Tesseracted Surrogates

This tutorial illustrates the process of packing surrogate models and their functionalities via <a href="../../../../tesseract-docs/index.html"><span class="product">Tesseracts</span></a>. Here, two surrogate models implemented and trained in two different environments, namey `Julia` and `JAX`, are Tesseracted. Their use within downstream frameworks, such as design optimization, is demostrated in a [separate demo](../optimization_with_surrogates/linear_ellipse_optimization.md).

```{figure} tesseracting_surrogates.png
:alt: linear-elasticity-ellipse
:width: 1200px
```

## Contents
```{toctree}
:maxdepth: 2
overview.md
problem_setting.md
surrogates_overview/surrogates_overview.md
tesseracting_surrogates/tesseracting_surrogates.md
```
