---
orphan: true
og:title: "Tesseract Hackathon winners push the boundaries of differentiable scientific pipelines"
og:description: "Announcing the winners and highlights from the 2026 Tesseract Hackathon."
blog_date: "2026-09-30"
blog_author: "@samalipio"
blog_title: "Tesseract Hackathon winners push the boundaries of differentiable scientific pipelines"
blog_description: "Announcing the winners and highlights from the 2026 Tesseract Hackathon."
---

# Tesseract Hackathon winners push the boundaries of differentiable scientific pipelines

In our latest hackathon, we challenged researchers and engineers around the world to compose a differentiable scientific workflow from multiple Tesseracts, using end-to-end gradients to solve a real design, inference, or training problem. Over the past month, nearly 125 participants from 33 different countries took on this challenge, culminating in 30 completed projects. Read on to learn more about the winning teams that each took home a portion of our $20,000 prize pool.

## Grand Prize: PRISMO

Author: Benjamin Vial | [Explore PRISMO](https://github.com/benvial/prismo)

**What it is:** Free-form optimization of the dopant layout in a silicon optical phase shifter.

**Why we love it:** A state-of-the-art Julia semiconductor solver and a FEniCS optics solver combine into one JAX gradient, each supplying its own kind of adjoint.

<figure>
<video autoplay loop muted playsinline aria-label="Animation of the dopant distribution in a silicon waveguide evolving during optimization.">
  <source src="../../_static/blog/2026-09-30-tesseract-hackathon-winners/prismo.mp4" type="video/mp4">
</video>
<figcaption>PRISMO reshaping the dopant distribution around a silicon waveguide core, iteration by iteration.</figcaption>
</figure>

The top winner of this year’s hackathon is PRISMO, which explores a different way of designing components for optical chips. It optimizes the distribution of electrical dopants in a silicon device that controls light, allowing the structure itself to evolve rather than restricting the design to a predefined geometry. Starting from a conventional design, the optimization discovers a ring-like structure that wraps around the region where light travels and produces more than five times the change in optical phase.

> Tesseract made this possible by letting PRISMO treat separate physics solvers as pieces of one differentiable model. The semiconductor and optical simulations are written in different languages, but Tesseract gives them a common interface for both their calculations and derivatives. JAX can then connect them and propagate gradients through the whole multiphysics simulation, without having to rewrite or tightly couple the underlying solvers.
>
> — Benjamin Vial

## Second Prize: δSnowSac

Author: Kamlesh Sawadekar | [Explore δSnowSac](https://github.com/kasProg/dSnowSac)

**What it is:** A neural network that calibrates NOAA’s operational streamflow models, Snow-17 and SAC-SMA, so they no longer have to be tuned basin by basin.

**Why we love it:** The learning signal flows back through both original Fortran models into the network, combining forward-mode derivatives across the physics with reverse-mode autograd across the network.

<figure>
<video autoplay loop muted playsinline aria-label="Animation of simulated streamflow converging toward observed streamflow over training epochs.">
  <source src="../../_static/blog/2026-09-30-tesseract-hackathon-winners/dsnowsac.mp4" type="video/mp4">
</video>
<figcaption>Simulated vs. observed streamflow in a basin held out from training, as the network learns to calibrate Snow-17 and SAC-SMA.</figcaption>
</figure>

Our second prize winner, δSnowSac, is a perfect illustration of how to bridge legacy solvers (implemented in Fortran in the 1970s, no less!) with modern PyTorch models in an efficient gradient-based parameter learning pipeline.

Prediction of hydrological variables is important for water resource management, and NOAA has long relied on models like Snow-17 and SAC-SMA for streamflow prediction. However, these models traditionally require laborious tuning basin by basin. δSnowSac is a differentiable modeling framework that teaches a neural network to do that tuning automatically. One network learns all 27 model parameters across many basins at once, including basins it has never seen. Unlike a pure machine-learning model, it doesn’t replace the physics. It works through NOAA’s original Fortran code, left untouched, so every prediction still comes with the snowpack, soil moisture and physical settings forecasters already know and trust.

> A differentiable model usually requires rewriting the physics model in PyTorch or JAX. This bottleneck was alleviated using Tesseract, which was wrapped around each of NOAA’s models. The Tesseract’s PyTorch integration helps in chaining the two containers, allowing the learning signal to flow from the final streamflow loss back through both Fortran models into the network. Overall, the Tesseract kept the models separate, just as NOAA maintains them, and still let them be trained as one.
>
> — Kamlesh Sawadekar

## Best in Track Awards

### Track 1: Normax

_Inverse design & shape optimization_ | Author: Rafael Pastrana | [Explore Normax](https://github.com/arpastrana/normax)

**What it is:** Shape and member-size optimization of lightweight structures, with a Eurocode 3 building code check inside the optimization loop.

**Why we love it:** Making the building code differentiable lets form finding, structural analysis, and compliance share one gradient, which uses 29–67% less material than sizing members on a fixed geometry.

<figure>
<video autoplay loop muted playsinline aria-label="Animation comparing three gridshell optimization runs, with the end-to-end run reaching the lightest design.">
  <source src="../../_static/blog/2026-09-30-tesseract-hackathon-winners/normax.mp4" type="video/mp4">
</video>
<figcaption>Three ways to optimize a gridshell: sections only, heights and sections, and the full pipeline end to end. The end-to-end run finds the lightest design.</figcaption>
</figure>

### Track 2: Coldplate

_Multi-physics & coupled systems_ | Author: Abd Elilah Tauil | [Explore Coldplate](https://github.com/TAUIL-Abd-Elilah/coldplate)

**What it is:** Topology optimization of a natural-convection cold plate, with a fluid solver and a thermal solver coupled in both directions.

**Why we love it:** It differentiates the coupled equilibrium properly with the implicit function theorem, and shows that leaving the two-way coupling out of the gradient makes it catastrophically wrong, with a third of the signs flipped.

<figure>
<video autoplay loop muted playsinline aria-label="Animation of a cold plate topology optimization showing material layout, coolant flow, and chip temperature.">
  <source src="../../_static/blog/2026-09-30-tesseract-hackathon-winners/coldplate.mp4" type="video/mp4">
</video>
<figcaption>Cold plate topology optimization: material layout, temperature and coolant flow, and chip temperature per design iteration.</figcaption>
</figure>

### Track 3: Differentiable Silicon

_Hybrid ML + mechanistic models_ | Author: S M Hozaifa Hossain | [Explore Differentiable Silicon](https://github.com/hozaifa1/differentiable-silicon)

**What it is:** Tuning how a ferroelectric transistor is manufactured so that a spiking neural network built from it classifies heartbeats better.

**Why we love it:** The gradient reaches into the thorniest piece of software in the entire hackathon, Sentaurus TCAD (a closed-source binary with no derivatives, driven over SSH), via finite differences refined with Broyden’s method.

<figure>
<video autoplay loop muted playsinline aria-label="Animation of transistor transfer curves and four fabrication parameters changing over Broyden steps.">
  <source src="../../_static/blog/2026-09-30-tesseract-hackathon-winners/differentiable-silicon.mp4" type="video/mp4">
</video>
<figcaption>Each accepted Broyden step moves four fabrication parameters and reshapes the device’s simulated transfer curves.</figcaption>
</figure>

### Track 4: OpenSees-SHM

_Differentiable inference & UQ_ | Authors: Weipeng Xu, Ziyuan Xie, Dazhi Zhao, Tianju Xue | [Explore OpenSees-SHM](https://github.com/xwpken/opensees-shm-tesseract)

**What it is:** Inferring where and how badly a steel structure has corroded from simulated sensor data, with uncertainty estimates.

**Why we love it:** An entire OpenSees finite element program becomes a differentiable input, with gradients from OpenSees’ own sensitivity analysis, composed with a finite-difference Tesseract for the corroded cross-sections.

<figure>
<video autoplay loop muted playsinline aria-label="Animation of a truss bridge deforming under transient excitation, with candidate damage segments highlighted.">
  <source src="../../_static/blog/2026-09-30-tesseract-hackathon-winners/opensees-shm.mp4" type="video/mp4">
</video>
<figcaption>Bridge response under a designed transient excitation, with candidate damage segments in red.</figcaption>
</figure>

### Track 5: Tesseract Inverse Thermography

_Differentiable graphics & rendering_ | Author: Usi Adia-Nimuwa | [Explore Tesseract Inverse Thermography](https://github.com/il-miscusi/tesseract-inverse-thermography)

**What it is:** Recovering hidden heat sources from a single thermal camera image.

**Why we love it:** The camera itself is a differentiable renderer, so gradients run from the pixels back through a coupled Fortran/JAX/PyTorch flow–heat equilibrium, learned closure model included.

```{figure} ../static/blog/2026-09-30-tesseract-hackathon-winners/inverse-thermography.png
:alt: Grid of heat source recoveries and image residuals for a calibrated and a mis-calibrated renderer.

Heat sources recovered from the same noisy thermal image through a calibrated and a mis-calibrated renderer.
```

## Best Engineering / Tesseract Hack: Impact-Adjoint

Author: Harsh Singh ([@singhharsh1708](https://github.com/singhharsh1708)) | [Explore Impact-Adjoint](https://github.com/singhharsh1708/impact-adjoint)

**What it is:** Exact gradients through collisions: naive autodiff through a bouncing-ball simulation can return a gradient of exactly zero when the true one isn’t, so a Julia solver supplies the collision sensitivities instead.

**Why we love it:** Since the hackathon began, Harsh has landed nearly 30 merged pull requests across Tesseract Core, Tesseract-JAX, Tesseract-Torch, and Tesseract-Streamlit, the deepest engagement with the stack of any entry.

## Best Visual: Normax

Track 1 winner Rafael Pastrana also took home the award for best visual in the hackathon for the [animations](https://github.com/arpastrana/normax#what-is-special-about-normax) in his project, Normax.

We also want to give a shout to the [interactive walkthrough](https://julian-8897.github.io/tesseract-hybrid-closure/) of Julian Chan’s [Differentiable Hybrid Closure for 2D Turbulence](https://github.com/julian-8897/tesseract-hybrid-closure), where a PyTorch closure model corrects a JAX spectral solver.

## Honorable Mentions

This year’s hackathon was particularly competitive, with many more excellent projects than prizes available. The following honorable mentions were particularly compelling:

- ```{image} ../static/blog/2026-09-30-tesseract-hackathon-winners/neurolocate-head.png
  :alt: NeuroLocate recovering a brain source location inside a head model.
  :class: blog-img-thumb
  ```

  [NeuroLocate](https://github.com/BrainCapture/NeuroLocate) by Magnus Guldberg Pedersen localizes brain activity from EEG recordings, chaining a PyTorch proposal network, the OpenMEEG boundary element solver, and a JAX refinement loop. The network makes a first guess, and the refinement loop improves it with gradients through a C++ solver that has no autodiff of its own.

- ```{image} ../static/blog/2026-09-30-tesseract-hackathon-winners/cadjoint.webp
  :alt: The Cadjoint editor with Python code next to a parametric bracket in a 3D viewport.
  :class: blog-img-thumb
  ```

  [Cadjoint](https://github.com/andrinr/cadjoint) by Andrin Rehmann is code-first CAD in the browser. Sketches, constraints, meshing and FEM simulation form one function that JAX can differentiate end to end, and a compiler turns JAX programs into WebGPU shaders so models render live.

One of the most exciting threads across the hackathon was teams writing custom differentiation rules for components that are normally non-differentiable. Beyond the winners above, [Aerostealth](https://github.com/esemsc-ss2524/aerostealth) turns OpenFOAM’s adjoint solver into a differentiable component for co-designing an airfoil’s drag and radar signature, [Tesseract Physics-Guided Diffusion Design](https://github.com/xiezy964/tes-phy-guide) differentiates through Gmsh meshing inside a diffusion-based design loop, and [Vitrify](https://github.com/Marc-Dvci/Vitrify) derives an exact adjoint for a 3D FEniCSx thermomechanics solver.

Finally, the most unexpected application area goes to [Harmonicut](https://github.com/zkasuran/harmonicut), which reshapes the undercut of a marimba bar to pull its overtones toward the ideal 1:4:10 tuning.
