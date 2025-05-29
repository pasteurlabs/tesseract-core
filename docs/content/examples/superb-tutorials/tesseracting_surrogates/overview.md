# The need for interoperable autodiff engines in SciML industrial pipelines
At Pasteur, software artifacts are distributed via containers called <span class="product">Tesseracts</span>. These are docker-based containers purposely built for Scientific ML, as they embed autodiff capabilities. This fact, together with other ML-specific features, makes <span class="product">Tesseracts</span> the ideal means for distribution of SciML components.

Fundamental <span class="product">Tesseract</span> features include,

- Freedom of implementation: components can be implemented in any framework / language;
- Separation of concerns: the job of the R&D staff ends when the component respects the <span class="product">Tesseract</span>'s specs;
- Heterogeneous computing: the components do not need to be executed on the same hardware to work together in a downstream shared pipeline;
- Universal deployment: deployment of components consists of applying a fixed containerization recipe and parsing the exposed API endpoint.

ML surrogates are ubiquitous SciML components as they enable fast, robust, and differentiable emulation of physics processes. As such, in this demo, ML surrogates implemented and trained in different environments are containerized and deployed.

More specifically, this demo showcases how, starting from surrogates trained in different environments, namely `Julia` and `JAX`, <span class="product">Tesseracts</span> enable their deployment in a universal and standardized manner. For simplicity, the surrogates emulate the map from parametrized loading conditions to von-Mises stresses (in `Julia`) and displacements (in `JAX`) in a two-dimensional linear-elasticity problem.

```{figure} tesseracting_surrogates.png
:alt: linear-elasticity-ellipse
:width: 1200px
```
