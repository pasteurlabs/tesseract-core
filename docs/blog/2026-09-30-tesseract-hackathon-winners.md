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

## Grand prize: PRISMO

Author: Benjamin Vial | [Explore PRISMO](https://github.com/benvial/prismo)

The top winner of this year’s hackathon is PRISMO, which explores a different way of designing components for optical chips. It optimizes the distribution of electrical dopants in a silicon device that controls light, allowing the structure itself to evolve rather than restricting the design to a predefined geometry. Starting from a conventional design, the optimization discovers a ring-like structure that wraps around the region where light travels and produces more than five times the change in optical phase. It is a concrete example of how combining physical simulation with automatic differentiation can reveal device designs that are difficult to find by hand.

“Tesseract made this possible by letting PRISMO treat separate physics solvers as pieces of one differentiable model,” says author Benjamin Vial. “The semiconductor and optical simulations are written in different languages, but Tesseract gives them a common interface for both their calculations and derivatives. JAX can then connect them and propagate gradients through the whole multiphysics simulation, without having to rewrite or tightly couple the underlying solvers.”

## Second prize: δSnowSac

Author: Kamlesh Sawadekar | [Explore δSnowSac](https://github.com/kasProg/dSnowSac)

Our second prize winner, δSnowSac, is a perfect illustration of how to bridge legacy solvers (implemented in Fortran in the 1970s, no less!) with modern PyTorch models in an efficient gradient-based parameter learning pipeline. Prediction of hydrological variables is important for water resource management. Hence, NOAA have their long-established models like Snow-17 and SACSMA for streamflow prediction since 1970s. However, these models traditionally require laborious tuning basin by basin. The solution is differentiable model framework, δSnowSac that teaches a neural network to do that tuning automatically. Unlike a pure machine-learning model, it doesn’t replace the physics. It works through NOAA’s original Fortran code, left untouched, so every prediction still comes with the snowpack, soil moisture and physical settings forecasters already know how to read and check.

According to creator Kamlesh Sawadekar, “a differentiable model usually requires rewriting the physics model in PyTorch or Jax framework. This bottleneck was alleviated using Tesseract, which was wrapped around each of the NOAA’s models. The Tesseract’s PyTorch integration helps in chaining the two containers, allowing the learning signal to flow from the final streamflow loss back through both Fortran models into the network. Overall, the Tesseract kept the models separate, just as NOAA maintains them, and still let them be trained as one.”

## Best in Track Awards

### Track 1, Inverse design & shape optimization: Normax

Author: Rafael Pastrana | [Explore Normax](https://github.com/arpastrana/normax)

What if typically “hidden” requirements like building codes could be part of the design process from the start? Normax explores this and shows what’s possible when entire pipelines become differentiable, in a really creative way. We’re excited to see where it goes, and how projects like it are going to empower engineers to find better solutions faster.

### Track 2, Multi-physics & coupled systems: Coldplate

Author: Abd Elilah Tauil | [Explore Coldplate](https://github.com/TAUIL-Abd-Elilah/coldplate)

A brilliant example of how to find an optimal coupled equilibrium through Newton-Krylov iteration when Picard iteration is unstable, Coldplate leverages both forward- and reverse-mode AD and explains why ignoring the implicit function theorem can provide misleading results.

### Track 3, Hybrid ML + mechanistic models: Differentiable Silicon

Author: S M Hozaifa Hossain | [Explore Differentiable Silicon](https://github.com/hozaifa1/differentiable-silicon)

Differentiable Silicon combined a Tesseract wrapped around the thorniest piece of software in the entire hackathon (Sentaurus TCAD) and an impressive gradient update mechanism with Broyden’s method.

### Track 4, Differentiable inference & UQ: Opensees-SHM

Authors: Weipeng Xu, Ziyuan Xie, Dazhi Zhao, Tianju Xue | [Explore Opensees-SHM](https://github.com/xwpken/opensees-shm-tesseract)

OpenSees-SHM wraps the OpenSees structural solver so that an entire finite element program (passed in as JSON) becomes a differentiable input, with gradients coming from the Opensee’s own sensitivity analysis rather than finite differences.

### Track 5, Differentiable graphics & rendering: Tesseract Inverse Thermography

Author: Usi Adia-Nimuwa | [Explore Tesseract Inverse Thermography](https://github.com/il-miscusi/tesseract-inverse-thermography)

Tesseracts spanning domains (heat transport, fluid dynamics, rendering) and software frameworks (JAX, PyTorch, Fortran) are coupled to recover heat sources from thermal images in an inverse manner through gradient descent, with a learned closure model in the loop. Very neat!

## Best Engineering / Tesseract Hack: Impact-Adjoint

Author: Harsh Singh ([@singhharsh1708](https://github.com/singhharsh1708))

This category recognizes the entry that shows the deepest engagement with the Tesseract stack, in this case exemplified via dozens of code improvements to the Tesseract ecosystem. While building his project [Impact Adjoint](https://github.com/singhharsh1708/impact-adjoint), Harsh’s contributions were hard to beat in terms of demonstrated effort, technical excellence, and overall impact on the Tesseract community.

## Best Visual

Track 1 winner Rafael Pastrana also took home the award for best visual in the hackathon for his project, [Normax](https://github.com/arpastrana/normax)).

```{figure} ../static/blog/gridshell_optimization_web.gif
:alt: Example animation from Normax demonstrating a gridshell design.

Animation demonstrating a gridshell design with form and sections optimized through the full Normax pipeline.
```

## Honorable Mentions

This year’s hackathon was particularly competitive, with many more excellent projects than prizes available. The following honorable mentions were particularly compelling:

- [NeuroLocate](https://github.com/BrainCapture/NeuroLocate) (authored by Magnus Guldberg Pedersen)
- [Cadjoint](https://github.com/andrinr/cadjoint) (authored by Andrin Rehmann)

_Tesseract is a free, open-source framework for differentiable scientific computing.
[Docs](https://docs.pasteurlabs.ai/projects/tesseract-core) · [Demos](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/demo/demo/) · [GitHub](https://github.com/pasteurlabs/tesseract-core) · [Forum](https://si-tesseract.discourse.group/)_
