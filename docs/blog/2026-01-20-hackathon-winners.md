---
orphan: true
og:title: Announcing the winners of the Tesseract Hackathon
og:description: "The inaugural virtual Tesseract Hackathon concluded with impressive entries from global researchers and engineers building differentiable pipelines for scientific impact."
blog_date: "2026-01-20"
blog_author: "@dionhaefner, @samalipio"
blog_title: "Announcing the winners of the Tesseract Hackathon"
blog_description: "The inaugural Tesseract Hackathon concluded with impressive entries from global researchers and engineers."
---

# Announcing the winners of the Tesseract Hackathon

_This post originally appeared on [Pasteur Labs Insights](https://pasteurlabs.ai/insights/tesseract-hackathon-winners)._

## Introduction

The inaugural virtual Tesseract Hackathon has concluded, and we're thrilled to announce the winners! Over thirty days, participants from around the world developed applications leveraging Tesseract's differentiable pipeline capabilities to address real scientific challenges. The brief: use the differentiable pipeline capabilities of Tesseract for scientific impact, and publish all code under an open source license.

## 1st Place: Multi-Agent Differentiable Predictive Control for Zero-Shot PDE Scalability

**Authors:** Pietro Zanotta, Dibakar Roy, and Honghui Zheng

[Source Code](https://github.com/SOLARIS-JHU/Multi-Agent-DPC) | [Author submission](https://www.linkedin.com/posts/pietro-zanotta_tesseracthackathon-activity-7414046354915774464-ZMhq?)

This project explores how agents in flow fields can coordinate without direct communication. The team employed a differentiable PDE solver wrapped as a Tesseract, enabling agents to learn action policies via gradient descent. Notably, policies trained on a set of 20 agents are still useful when actually deploying 60 agents.

```{figure} ../static/blog/hackathon-multi-agent-dpc.png
:alt: Multi-Agent DPC architecture

Multi-Agent Differentiable Predictive Control architecture for zero-shot PDE scalability.
```

<figure>
<video autoplay loop muted playsinline>
  <source src="../_static/blog/hackathon-heat2d-decentralized.mp4" type="video/mp4">
</video>
<figcaption>Visualization of decentralized agents shaping a 2D heat field without direct communication.</figcaption>
</figure>

## 2nd Place: DeepSwingr -- A Differentiable Framework for Cricket Ball Swing Optimization

**Author:** Pavan Govindaraju

[Source Code](https://github.com/gpavanb1/DeepSwingr) | [Author Submission](https://www.linkedin.com/posts/activity-7410628586577002496-vu56?)

This trajectory optimization pipeline comprises four specialized Tesseracts:

1. **Physics backend** -- Calculates forces on the ball
2. **Integrator** -- Performs numerical integration of the equations of motion
3. **Swing logic** -- Higher-order Tesseract orchestrating integrator and physics components
4. **Optimizer** -- Searches for optimal parameters

The modular design demonstrates Tesseract's containerization benefits, allowing component substitution without code modifications.

```{figure} ../static/blog/hackathon-deepswingr.png
:alt: DeepSwingr system architecture

DeepSwingr system architecture with four specialized Tesseracts for cricket ball swing optimization.
```

## Social Media Bonus: PruneDeepONet

**Author:** Tomoki Koike

[Source code](https://github.com/smallpondtom/prunedeeponet) | [Author submission](https://www.linkedin.com/posts/tomoki-koike_tesseracthackathon-activity-7414068074997383168-6vgS?)

This entry became far and away the most viral project among submissions, comparing two surrogate architectures using Tesseract.

## Honorable Mentions

- **[Diffopteract](https://github.com/llueg/diffopteract)** -- Differentiable Optimization in JAX via Julia/JuMP Tesseract (Laurens Lueg)
- **[DiffPIC](https://github.com/ale-ballester/tesseract-hackathon)** -- Differentiable Particle-in-Cell Optimization (Alejo Ballester, Rushan Zhang, Tage Burnett, Harshavardhan Harish)
- **[Tesseract-pinn-inverse-burgers](https://github.com/julian-8897/tesseract-pinn-inverse-burgers)** -- Backend-Agnostic Inverse 1D Burgers Solver (Julian Chan)

---

_Tesseract is a free, open-source framework for differentiable scientific computing._
_[Docs](https://tesseract.pasteurlabs.ai) · [Demos](https://tesseract.pasteurlabs.ai/content/demo/demo.html) · [GitHub](https://github.com/pasteurlabs/tesseract-core) · [Forum](https://si-tesseract.discourse.group/)_
