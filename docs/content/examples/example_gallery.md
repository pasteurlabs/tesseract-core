# Example Gallery: Building Blocks

```{toctree}
:name: example-gallery
:caption: Example Gallery
:maxdepth: 2
:hidden:
:glob:
building-blocks/helloworld.md
building-blocks/vectoradd.md
building-blocks/univariate.md
building-blocks/fortran.md
building-blocks/enzyme.md
building-blocks/matlab.md
building-blocks/packagedata.md
building-blocks/arm64.md
building-blocks/localpackage.md
building-blocks/dataloader.md
building-blocks/file_io.md
building-blocks/finitediff.md
building-blocks/gradient-fallbacks.md
```

This is a gallery of Tesseract examples that end at the `build` stage of the Tesseract lifecycle, and that can act as starting points to define and build your own Tesseracts.

You can also find these Tesseracts in the `examples` directory of the [code repository](https://github.com/pasteurlabs/tesseract-core).

```{important}
**Beyond the Build**: The real magic happens long after building a Tesseract. For some example applications that *use* Tesseracts in workflows, check out the [Demos & Tutorials](../demo/demo.md) and [Community Showcase](https://si-tesseract.discourse.group/c/showcase/11).
```

::::{grid} 2
:gutter: 2

:::{grid-item-card} HelloWorld
:link: building-blocks/helloworld
:link-type: doc

A simple "hello world" Tesseract.
:::
:::{grid-item-card} VectorAdd
:link: building-blocks/vectoradd
:link-type: doc

Tesseract performing vector addition. Highlighting simple array operations and how to use the Tesseract Python SDK.
:::
:::{grid-item-card} Univariate
:link: building-blocks/univariate
:link-type: doc

A Tesseract that wraps the univariate Rosenbrock function, which is a common test problem for optimization algorithms.
:::
:::{grid-item-card} Fortran Integration
:link: building-blocks/fortran
:link-type: doc

Wrapping a Fortran heat equation solver. Demonstrates subprocess-based integration for legacy compiled code.
:::
:::{grid-item-card} Differentiating Compiled Code
:link: building-blocks/enzyme
:link-type: doc

Differentiating a Fortran solver with Enzyme. Demonstrates exact, machine-precision JVP and VJP for compiled code with no hand-written adjoints.
:::
:::{grid-item-card} MATLAB Integration
:link: building-blocks/matlab
:link-type: doc

Wrapping a MATLAB spring-mass-damper ODE solver. Demonstrates using the official MathWorks Docker image to run MATLAB code via `matlab -batch`.
:::
:::{grid-item-card} Package Data
:link: building-blocks/packagedata
:link-type: doc

A guide on including local files into a built Tesseract.
:::
:::{grid-item-card} Pyvista on ARM64
:link: building-blocks/arm64
:link-type: doc

A guide showcasing how to use custom build steps to install pyvista within an ARM64 Tesseract.
:::
:::{grid-item-card} Local Dependencies
:link: building-blocks/localpackage
:link-type: doc

A guide on installing local Python packages into a Tesseract.
:::
:::{grid-item-card} Data Loader
:link: building-blocks/dataloader
:link-type: doc

Tesseract that loads in data samples from a folder without loading them into memory.
:::

:::{grid-item-card} Input/Output Path References
:link: building-blocks/file_io
:link-type: doc

Tesseract that passes files and directories through the input/output mounts
instead of serializing their contents into the request payload.
Useful when inputs or outputs are large on disk, or consist of
many (or a variable number of) files.
:::

:::{grid-item-card} Finite Difference Gradients
:link: building-blocks/finitediff
:link-type: doc

Make any Tesseract differentiable without implementing analytical gradients.
Useful for prototyping and complex nested schemas. _(Experimental)_
:::

:::{grid-item-card} Gradient Endpoint Derivation Fallbacks
:link: building-blocks/gradient-fallbacks
:link-type: doc

Derive missing gradient endpoints (JVP, VJP, Jacobian) from ones you have
already implemented. _(Experimental)_
:::

::::
