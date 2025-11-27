# Ansys Integration

```{toctree}
:name: ansys-gallery
:caption: Ansys Gallery
:maxdepth: 2
:hidden:
:glob:
ansys_integration/spaceclaim_tess.md
ansys_integration/pymapdl_tess.md
```

This is a gallery of Tesseract examples integrating with Ansys products in different ways, demonstrating the versatility that you can then use to build your own Tesseracts.

You can find these Ansys Tesseracts in the `demo/showcase` directory of the [code repository](https://github.com/pasteurlabs/tesseract-core).


::::{grid} 2
   :gutter: 2

   :::{grid-item-card} Spaceclaim
      :link: ansys_integration/spaceclaim_tess.html

      A Tesseract that wraps SpaceClaim geometry creation.
   :::
   :::{grid-item-card} PyMAPDL
      :link: ansys_integration/pymapdl_tess.html

      A differentiable Tesseract that wraps PyMAPDL performing SIMP topology optimization.
   :::

::::
