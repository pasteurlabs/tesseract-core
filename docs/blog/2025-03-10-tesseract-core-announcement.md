---
orphan: true
og:title: Tesseract Core is now live... and open source!
og:description: "We're announcing the public release of Tesseract Core, a free and open source application enabling scientists and engineers to build end-to-end differentiable pipelines with minimal code."
blog_date: "2025-03-10"
blog_author: Pasteur Labs
blog_title: "Tesseract Core is now live... and open source!"
blog_description: "We're announcing the public release of Tesseract Core, a free and open source application enabling scientists and engineers to build end-to-end differentiable pipelines with minimal code."
---

# Tesseract Core is now live... and open source!

_This post originally appeared on [Pasteur Labs Insights](https://pasteurlabs.ai/insights/tesseract-core-announcement)._

We've released Tesseract Core as a free and open source application to define, build, and execute Tesseracts. Alongside this release, we launched the [Tesseract dev forum](https://si-tesseract.discourse.group/) to support the growing ecosystem.

## What are Tesseracts?

One of the key challenges in scientific computing is bridging the gap between research-grade code and production systems. Tesseracts allow you to package complex scientific and machine learning code into components that are readily usable and effortless to deploy in production.

Tesseracts provide native support for endpoints that interface with automatic differentiation frameworks, enabling gradient-based, adaptive optimization involving multiple differentiable components in a pipeline --- even when distributed across different machines.

Tesseract Core is a command line app and Python SDK for building end-to-end differentiable pipelines consisting of wildly different components like physical simulators, geometric operators, differentiable meshers and renderers, scientific data transforms, neural networks, and more.

<figure>
<img src="../_static/blog/rosenbrock_optimization.gif" alt="Rosenbrock optimization">
<figcaption>Gradient-based optimization of the Rosenbrock function using a Tesseract.</figcaption>
</figure>

## Learn more

- Check out the [Tesseract Core documentation](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/)
- Join the [Tesseract Community Forum](https://si-tesseract.discourse.group/)
- Visit [Tesseract Core on GitHub](https://github.com/pasteurlabs/tesseract-core)
