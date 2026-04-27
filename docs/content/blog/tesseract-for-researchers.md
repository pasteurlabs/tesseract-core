---
html_class: blog-page
og:title: Tesseract for Researchers
og:description: "How Tesseract helps researchers in differentiable simulation, physics-informed ML, and probabilistic modeling work across language and framework boundaries."
---

# Tesseract for Researchers

_Mar 31, 2026_ · The Tesseract Team

Tesseract helps researchers in differentiable simulation, physics-informed ML, and probabilistic modeling work across language and framework boundaries without losing gradient information.

## Why Tesseract?

- **Mix AD strategies freely.** Combine JAX, PyTorch, Julia, or analytic adjoints in the same pipeline. Each Tesseract chooses its own differentiation approach; gradients compose automatically at the boundaries.
- **Share solvers without dependency hell.** Wrap any simulator as a self-contained container. Collaborators run it with `tesseract serve` -- no environment setup, no version conflicts.
- **Embed in JAX or PyTorch.** With [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), any Tesseract becomes a JAX primitive you can JIT, vmap, or differentiate through.
- **Run probabilistic inference.** Use Tesseracts as black-box likelihoods in NumPyro, Stan, or any sampling framework -- gradients included.

## Get started

- [Installation](../introduction/installation.md)
- [Get started tutorial](../introduction/get-started.md)
- [Differentiable programming primer](../misc/differentiable-programming.md)
- [End-to-end demos](../demo/demo.md)
