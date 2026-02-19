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
building-blocks/packagedata.md
building-blocks/arm64.md
building-blocks/localpackage.md
building-blocks/dataloader.md
building-blocks/filereference.md
```

This is a gallery of Tesseract examples that end at the `build` stage of the Tesseract lifecycle, and that can act as starting points to define and build your own Tesseracts.

You can also find these Tesseracts in the `examples` directory of the [code repository](https://github.com/pasteurlabs/tesseract-core).

```{important}
**Beyond the Build**: The real magic happens long after building a Tesseract. For some example applications that *use* Tesseracts in workflows, check out the [Demo](../demo/demo.md) and [Community Showcase](https://si-tesseract.discourse.group/c/showcase/11).
```

::::{grid} 2
:gutter: 2

:::{grid-item-card} HelloWorld
:link: building-blocks/helloworld.html

      A simple "hello world" Tesseract.

:::
:::{grid-item-card} VectorAdd
:link: building-blocks/vectoradd.html

     Tesseract performing vector addition. Highlighting simple array operations and how to use the Tesseract Python API.

:::
:::{grid-item-card} Univariate
:link: building-blocks/univariate.html

      A Tesseract that wraps the univariate Rosenbrock function, which is a common test problem for optimization algorithms.

:::
:::{grid-item-card} Fortran Integration
:link: building-blocks/fortran.html

      Wrapping a Fortran heat equation solver. Demonstrates subprocess-based integration for legacy compiled code.

:::
:::{grid-item-card} Package Data
:link: building-blocks/packagedata.html

      A guide on including local files into a built Tesseract.

:::
:::{grid-item-card} Pyvista on ARM64
:link: building-blocks/arm64.html

      A guide showcasing how to use custom build steps to install pyvista within an ARM64 Tesseract.

:::
:::{grid-item-card} Local Dependencies
:link: building-blocks/localdependency.html

      A guide on installing local Python packages into a Tesseract.

:::
:::{grid-item-card} Data Loader
:link: building-blocks/dataloader.html

     Tesseract that loads in data samples from a folder without loading them into memory.

:::

:::{grid-item-card} Input/Output File References
:link: building-blocks/filereference.html

     Tesseract that mounts input and output directories as datasets.
     To be used for Tesseracts with large inputs and/or outputs.

:::

::::
