---
orphan: true
og:title: "Gradient-based rocket fin design with Ansys, JAX, and Tesseract"
og:description: "We used Tesseract to connect Ansys tools into a differentiable pipeline, optimize rocket grid fins, and turn the optimizer's insights into a practical design that's 24% stiffer."
blog_date: "2026-05-12"
blog_author: "@andrinr"
blog_title: "Gradient-based rocket fin design with Ansys, JAX, and Tesseract"
blog_description: "We used Tesseract to connect Ansys tools into a differentiable pipeline and let gradients reshape a rocket grid fin."
---

# Gradient-based rocket fin design with Ansys, JAX, and Tesseract

_For the full methodology, see the [original forum post](https://si-tesseract.discourse.group/t/parametric-shape-optimization-of-rocket-fins-with-ansys-spaceclaim-pyansys-and-tesseract/109). All code to reproduce the results is available [here](https://github.com/pasteurlabs/tesseract-core/tree/main/demo/_showcase/ansys-shapeopt)._

## Introduction

If you work in simulation-driven design, you've probably hit this wall: you have a parametric CAD model in one tool, a mesher in another, and a finite element solver in a third. Each tool is good at what it does. Getting them to work together, let alone pass gradients between them, is where things fall apart.

We recently built a case study around exactly this kind of pipeline: optimizing rocket grid fin geometry using Ansys SpaceClaim for CAD, a custom mesh converter, and PyMAPDL for structural analysis. The physics is interesting, but what we really want to talk about is the engineering challenge underneath, and how Tesseract made it tractable.

<figure>
<img src="../_static/blog/rocket-fins-grid-fins.jpg" alt="Titanium grid fins on a Falcon 9 booster">
<figcaption>Second-generation titanium grid fins on a Falcon 9 booster. <a href="https://commons.wikimedia.org/wiki/File:Second-generation_titanium_grid_fins,_Iridium-2_Mission_(35533873795).jpg">SpaceX, Public Domain</a>.</figcaption>
</figure>

## Three tools, two operating systems, zero shared dependencies

Here's what the pipeline looks like: Ansys SpaceClaim generates parametric geometry from 16 design variables (angular positions of 8 bars on a grid fin). That geometry gets converted to a signed distance field on a regular grid. PyMAPDL then solves the linear elasticity problem and returns compliance --- a measure of how much the structure deforms under load.

<figure>
<img src="../_static/blog/rocket-fins-workflow.png" alt="Optimization workflow" class="blog-img-full">
<figcaption>End-to-end optimization workflow connecting Ansys SpaceClaim, SDF conversion, and PyMAPDL via Tesseract.</figcaption>
</figure>

Each of these tools lives in a different world. SpaceClaim runs on Windows with a commercial license. PyMAPDL has its own Python environment and dependency tree. JAX handles the glue code and autodiff on Linux. Without some way to bridge these environments, much of the effort goes into dependency management and data plumbing rather than the actual optimization.

With Tesseract, each tool becomes a self-contained component with a clean interface. We packaged SpaceClaim, the SDF converter, and PyMAPDL as three separate Tesseracts, each with isolated dependencies. The pipeline composition happens through [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), which handles orchestration and gradient flow.

## Gradients across boundaries

This is where things get interesting from a Tesseract perspective. Each component in this pipeline uses a _different_ differentiation strategy:

- PyMAPDL computes gradients via an **analytical adjoint**
- The SDF converter uses **finite differences**, wrapping another Tesseract to do so (a higher-order Tesseract)
- The JAX glue code between components uses **automatic differentiation**

Tesseract's differentiation interface unifies all of these behind the same contract: every component exposes Jacobian, JVP, or VJP endpoints, regardless of how it computes them internally. From the optimizer's perspective, it's just one differentiable function. Pipeline-level autodiff doesn't require everything to be written in one AD framework. It means gradients flow end-to-end even when the components couldn't be more different.

## The result: optimization as a design partner

With gradients in hand, we ran Adam from two different starting configurations. Both converged to similar topologies, which told us we were finding something real rather than getting stuck in local minima.

<div class="double-figure">
<figure>
<img src="../_static/blog/rocket-fins-opt-grid.gif" alt="Optimization from grid initial conditions">
<figcaption>Optimization from grid initial conditions.</figcaption>
</figure>
<figure>
<img src="../_static/blog/rocket-fins-opt-rnd.gif" alt="Optimization from random initial conditions">
<figcaption>Optimization from random initial conditions.</figcaption>
</figure>
</div>

The raw optimized designs were asymmetric and impractical to manufacture. But they taught us something useful: where to put the material. The optimizer consistently reinforced bar roots near attachment points, organized bars into orthogonal patterns, and created diagonal load paths to the fixed boundaries.

We took those insights and hand-designed a symmetric, manufacturable geometry. Running it back through the pipeline: **24% stiffer than the baseline grid, at the same mass.**

<div class="double-figure">
<figure>
<img src="../_static/blog/rocket-fins-manual-design.jpeg" alt="Manufacturable symmetric design">
<figcaption>Symmetric, manufacturable design informed by optimization insights.</figcaption>
</figure>
<figure>
<img src="../_static/blog/rocket-fins-conv-manual.png" alt="Convergence comparison with manual design">
<figcaption>Compliance comparison: manual design vs. optimized and baseline configurations.</figcaption>
</figure>
</div>

## The bigger picture

The rocket fin problem is a good example, but the pattern is general. Any workflow coupling simulation tools across different languages, platforms, or differentiation strategies faces the same integration challenge. Tesseract lets these tools compose without requiring them to share environments, dependencies, or differentiation frameworks.

We've since validated this by swapping in [PyVista](https://docs.pyvista.org/) for geometry and [JAX-FEM](https://github.com/deepmodeling/jax-fem) for the solver with minimal changes, which gives us some confidence that the modularity holds up in practice.

If this is the kind of workflow you're building, the [full technical writeup](https://si-tesseract.discourse.group/t/parametric-shape-optimization-of-rocket-fins-with-ansys-spaceclaim-pyansys-and-tesseract/109) has all the details, and the [demo code](https://github.com/pasteurlabs/tesseract-core/tree/main/demo/_showcase/ansys-shapeopt) is ready to run.

## Learn more

- Check out the [Tesseract Core documentation](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/)
- Join the [Tesseract Community Forum](https://si-tesseract.discourse.group/)
- Visit [Tesseract Core on GitHub](https://github.com/pasteurlabs/tesseract-core)
