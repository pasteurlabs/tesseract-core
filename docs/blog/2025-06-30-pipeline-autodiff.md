---
orphan: true
og:title: "New demo: Pipeline-level autodiff with Tesseracts"
og:description: "How Tesseracts unlock efficient, clean pipeline-level automatic differentiation."
blog_date: "2025-06-30"
blog_author: "@andrinr, @dionhaefner"
blog_title: "New demo: Pipeline-level autodiff with Tesseracts"
blog_description: "A demo highlighting how Tesseracts unlock efficient, clean pipeline-level automatic differentiation."
---

# New demo: Pipeline-level autodiff with Tesseracts

_This post originally appeared on [Pasteur Labs Insights](https://pasteurlabs.ai/insights/tesseract-pipeline)._

## Introduction

We're excited to share our new demo showcasing how Tesseracts enable pipeline-level automatic differentiation. In this demo, we use multiple Tesseracts to implement gradient-based parametric optimization of a finite element method (FEM) simulation.

## A quick refresher: What's a Tesseract?

Tesseracts are components that allow scientists to enable complex scientific workflows at scale. They are self-contained, self-documenting, and self-executing, available via command line and HTTP. Since each Tesseract is a standalone and stateless component, we can build pipelines connecting multiple Tesseracts to create complex workflows. This is particularly useful for multi-step computations, including use cases in digital engineering and machine learning.

## Tesseracts as pipeline-level autodiff enablers

Automatic differentiation (also known as "autodiff", or just "AD") is an important part of modern scientific computing. It enjoys particular success in the machine learning world, underpinning popular deep learning frameworks like PyTorch and TensorFlow.

Autodiff works really well for situations where all calculations can be expressed in a single program. However, when it comes to differentiating a system of multiple loosely coupled components, tracking gradients becomes much more difficult. Enabling this sort of system-level automatic differentiation is what motivated us to create Tesseracts in the first place. And today, we're keen to share this example highlighting Tesseracts in action.

## Demo: Parametric shape optimization with differentiable FEM simulation

Our demo is built using JAX-FEM and inspired by the 2D Topology Optimization with the SIMP Method. We've reformulated the problem and solved it as an optimization of a parameterized shape.

<figure>
<img src="../_static/blog/pipeline-data-flow.png" alt="Data flow through a Tesseract-based pipeline for parametric shape optimization">
<figcaption>Data flow through a Tesseract-based pipeline for parametric shape optimization, involving two separate Tesseracts: one for computing a signed distance field from a parametric geometry, and another for computing the compliance of a structure given a density field via finite element analysis.</figcaption>
</figure>

We've implemented this demo using multiple Tesseracts that communicate with each other, forming a multi-step computation pipeline. We then apply end-to-end automatic differentiation to carry out the optimization. The demo clearly shows the feasibility --- and efficacy --- of pipeline-level automatic differentiation with Tesseracts. In particular, Tesseracts simplify heterogeneous gradient computation, as well as managing dependencies, computing resources, and components.

<figure>
<img src="../_static/blog/pipeline-demo.png" alt="Shape optimization demo">
<figcaption>Shape optimization results showing the initial and optimized geometries.</figcaption>
</figure>

## Learn more

- Join the [Tesseract Community Forum](https://si-tesseract.discourse.group/)
- Visit [Tesseract Core on GitHub](https://github.com/pasteurlabs/tesseract-core)
- Visit [Tesseract-JAX on GitHub](https://github.com/pasteurlabs/tesseract-jax)
