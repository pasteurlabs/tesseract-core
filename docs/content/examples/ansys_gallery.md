# Example Gallery: Integrating with Ansys Products

```{toctree}
:name: ansys-gallery
:caption: Ansys Gallery
:maxdepth: 2
:hidden:
:glob:
ansys_integration/spaceclaim_tess.md
ansys_integration/pymapdl_tess.md
```

This is a collection of examples where Tesseract is integrated with [Ansys products](https://www.ansys.com/) in various ways.

The Ansys software suite is heavily used throughout the engineering simulation industry. Wrapping Ansys products as Tesseracts allows you to embed them into Tesseract-driven compute pipelines --- acting as poweful data generators, differentiable solvers, geometry engines, and more.

```{seealso}
You can find the code for all Ansys Tesseracts in the `demo/_showcase` directory of the [Tesseract Core repository](https://github.com/pasteurlabs/tesseract-core/tree/main/demo/_showcase).
```

::::{grid} 2
   :gutter: 2

   :::{grid-item-card} SpaceClaim
      :link: ansys_integration/spaceclaim_tess.html

      A Tesseract that wraps SpaceClaim for CAD geometry creation.
   :::
   :::{grid-item-card} MAPDL
      :link: ansys_integration/pymapdl_tess.html

      A differentiable Tesseract that wraps the MAPDL solver via PyMAPDL with an analytic adjoint, for use in SIMP topology optimization.
   :::

::::
