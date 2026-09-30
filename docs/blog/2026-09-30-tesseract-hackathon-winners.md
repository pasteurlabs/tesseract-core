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

<figure>
<video autoplay loop muted playsinline aria-label="Animation of the dopant distribution in a silicon waveguide evolving during optimization.">
  <source src="../../_static/blog/2026-09-30-tesseract-hackathon-winners/prismo.mp4" type="video/mp4">
</video>
<figcaption>PRISMO reshaping the dopant distribution around a silicon waveguide core, iteration by iteration.</figcaption>
</figure>

The top winner of this year’s hackathon is PRISMO, which explores a different way of designing components for optical chips. It optimizes the distribution of electrical dopants in a silicon device that controls light, allowing the structure itself to evolve rather than restricting the design to a predefined geometry. Starting from a conventional design, the optimization discovers a ring-like structure that wraps around the region where light travels and produces more than five times the change in optical phase. The setup is highly non-trivial, where a state-of-the-art Julia semiconductor solver and a FEniCS optics solver combine into one JAX gradient, each supplying its own kind of adjoint.

> Tesseract made this possible by letting PRISMO treat separate physics solvers as pieces of one differentiable model. The semiconductor and optical simulations are written in different languages, but Tesseract gives them a common interface for both their calculations and derivatives. JAX can then connect them and propagate gradients through the whole multiphysics simulation, without having to rewrite or tightly couple the underlying solvers.
>
> — Benjamin Vial

## Second Prize: δSnowSac

Author: Kamlesh Sawadekar | [Explore δSnowSac](https://github.com/kasProg/dSnowSac)

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

<figure>
<video autoplay loop muted playsinline aria-label="Animation comparing three gridshell optimization runs, with the end-to-end run reaching the lightest design.">
  <source src="../../_static/blog/2026-09-30-tesseract-hackathon-winners/normax.mp4" type="video/mp4">
</video>
<figcaption>Three ways to optimize a gridshell: sections only, heights and sections, and the full pipeline end to end. The end-to-end run finds the lightest design.</figcaption>
</figure>

What if typically “hidden” requirements like building codes could be part of the design process from the start? Normax makes a Eurocode 3 compliance check differentiable, so form finding, structural analysis, and the building code share one optimization loop. Optimizing shape and member sizes together this way uses 29–67% less material than sizing members on a fixed geometry. We’re excited to see where it goes, and how projects like it are going to empower engineers to find better solutions faster.

### Track 2: Coldplate

_Multi-physics & coupled systems_ | Author: Abd Elilah Tauil | [Explore Coldplate](https://github.com/TAUIL-Abd-Elilah/coldplate)

<figure>
<video autoplay loop muted playsinline aria-label="Animation of a cold plate topology optimization showing material layout, coolant flow, and chip temperature.">
  <source src="../../_static/blog/2026-09-30-tesseract-hackathon-winners/coldplate.mp4" type="video/mp4">
</video>
<figcaption>Cold plate topology optimization: material layout, temperature and coolant flow, and chip temperature per design iteration.</figcaption>
</figure>

A brilliant example of how to find an optimal coupled equilibrium through Newton-Krylov iteration when Picard iteration is unstable, Coldplate leverages both forward- and reverse-mode AD and explains why ignoring the implicit function theorem can provide misleading results. **Coldplate shows how leaving the two-way coupling between the thermal and fluid solvers out of the gradient makes it catastrophically wrong, with a third of the signs flipped**.

### Track 3: Differentiable Silicon

_Hybrid ML + mechanistic models_ | Author: S M Hozaifa Hossain | [Explore Differentiable Silicon](https://github.com/hozaifa1/differentiable-silicon)

<figure>
<video autoplay loop muted playsinline aria-label="Animation of transistor transfer curves and four fabrication parameters changing over Broyden steps.">
  <source src="../../_static/blog/2026-09-30-tesseract-hackathon-winners/differentiable-silicon.mp4" type="video/mp4">
</video>
<figcaption>Each accepted Broyden step moves four fabrication parameters and reshapes the device’s simulated transfer curves.</figcaption>
</figure>

Differentiable Silicon pushes a heartbeat-classification loss back through a spiking neural network and into a chip manufacturing simulator, tuning how a ferroelectric transistor is made so the hardware network classifies ECGs better. It combines a Tesseract wrapped around the thorniest piece of software in the entire hackathon (Sentaurus TCAD, a closed-source binary with no derivatives, driven over SSH on a separate machine) and an impressive gradient update mechanism with Broyden’s method.

### Track 4: OpenSees-SHM

_Differentiable inference & UQ_ | Authors: Weipeng Xu, Ziyuan Xie, Dazhi Zhao, Tianju Xue | [Explore OpenSees-SHM](https://github.com/xwpken/opensees-shm-tesseract)

<figure>
<video autoplay loop muted playsinline aria-label="Animation of a truss bridge deforming under transient excitation, with candidate damage segments highlighted.">
  <source src="../../_static/blog/2026-09-30-tesseract-hackathon-winners/opensees-shm.mp4" type="video/mp4">
</video>
<figcaption>Bridge response under a designed transient excitation, with candidate damage segments in red.</figcaption>
</figure>

OpenSees-SHM wraps the OpenSees structural solver so that an entire finite element program (passed in as JSON) becomes a differentiable input, with gradients coming from OpenSees’ own sensitivity analysis. Paired with a finite-difference Tesseract for corroded cross-sections, it infers where and how badly a steel structure has corroded, with uncertainty estimates, from simulated sensor data.

### Track 5: Tesseract Inverse Thermography

_Differentiable graphics & rendering_ | Author: Usi Adia-Nimuwa | [Explore Tesseract Inverse Thermography](https://github.com/il-miscusi/tesseract-inverse-thermography)

```{figure} ../static/blog/2026-09-30-tesseract-hackathon-winners/inverse-thermography.png
:alt: Grid of heat source recoveries and image residuals for a calibrated and a mis-calibrated renderer.

Heat sources recovered from the same noisy thermal image through a calibrated and a mis-calibrated renderer.
```

The project treats a thermal camera as a differentiable renderer, so gradients run from the pixels of a single image back through a coupled Fortran/JAX/PyTorch flow–heat equilibrium to the hidden heat sources, with a learned closure model in the loop. Very neat!

## Best Engineering / Tesseract Hack: Impact-Adjoint

Author: Harsh Singh ([@singhharsh1708](https://github.com/singhharsh1708)) | [Explore Impact-Adjoint](https://github.com/singhharsh1708/impact-adjoint)

This category recognizes the entry that shows the deepest engagement with the Tesseract stack. Since the hackathon began, Harsh has landed nearly 30 merged pull requests across Tesseract Core, Tesseract-JAX, Tesseract-Torch, and Tesseract-Streamlit, contributions that were hard to beat in terms of demonstrated effort, technical excellence, and overall impact on the Tesseract community.

The project itself fixes a subtle failure of naive autodiff through collisions. Differentiating a bouncing-ball simulation step by step can return a gradient of exactly zero when the true one isn’t, so Impact-Adjoint supplies exact collision sensitivities from a Julia solver instead.

## Best Visual: Normax

Track 1 winner Rafael Pastrana also took home the award for best visual in the hackathon for the [animations](https://github.com/arpastrana/normax#what-is-special-about-normax) in his project, Normax.

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
