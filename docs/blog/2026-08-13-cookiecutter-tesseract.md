---
orphan: true
og:title: "An easier way to build your first multi-Tesseract pipeline"
og:description: "Introducing cookiecutter-tesseract, a batteries-included starting point for building projects out of more than one Tesseract."
blog_date: "2026-08-13"
blog_author: "@samalipio"
blog_title: "An easier way to build your first multi-Tesseract pipeline"
blog_description: "Introducing cookiecutter-tesseract, a batteries-included starting point for building projects out of more than one Tesseract."
---

# An easier way to build your first multi-Tesseract pipeline

A single Tesseract packages one computation unit into a standardized component. If you're building a training loop, optimization routine, or a full application, you'll probably need to chain multiple Tesseracts into a larger workflow or integrate them with other code. To make that easier, we've published a [new guide](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/how-to/pipelines/) and the [`cookiecutter-tesseract`](https://github.com/pasteurlabs/cookiecutter-tesseract) template.

## You've built your Tesseracts, now what?

Real-world projects usually involve multiple components, for example a mesher feeding a solver, an encoder feeding a model, a simulation feeding a post-processor. Once you've packaged your code as Tesseracts, the next step is to determine how you'll call and chain your components together. Our new guide on [composing Tesseracts into pipelines](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/how-to/pipelines/) walks you through three important considerations as you build: 

1. **Calling each Tesseract:** Every Tesseract exposes a CLI, a REST API, and a Python SDK, how do you decide which is the right interface to reach for?
2. **Chaining Tesseracts:** How should data flow between Tesseracts, and how can you debug when the chain misbehaves?
3. **Building a multi-Tesseract pipeline:** How can you structure a complex project to keep things organized and consistent, share code between components, and build/test your components as a set?

## Introducing cookiecutter-tesseract

Tesseract Core is deliberately unopinionated about what you build on top, but once you’re chaining several components into an application, you end up hand-rolling the same things every time: a directory layout, a place for shared utilities, per-component test cases, CI that builds everything, and a runner to tie the pipeline together.

The [`cookiecutter-tesseract`](https://github.com/pasteurlabs/cookiecutter-tesseract) template gives you that structure as a batteries-included starting point, so you don't have to assemble it by hand. If you’re building an app out of a single Tesseract, `tesseract init` is still the right tool; if you’re building one out of many, you may want to consider this template in addition to the above guide.

### What's inside
- **Monorepo layout**: components (Tesseracts), shared code, and the pipeline app in one repo with a standardized structure.
- **A `make` workflow**: `make new`, `build`, `test`, `data`, and `run` wrap the common Tesseract commands so you don’t have to memorize them.
- **Component scaffolding**: `make new <name> [RECIPE=base|jax|pytorch]` spins up a new Tesseract, pre-wired to depend on your shared code.
- **Shared code package**: one place for utilities every Tesseract can import, installed automatically into each component.
- **Regression testing**: JSON test cases per component plus a `pytest` suite for the app, all runnable via `make test`.
- **CI/CD + pre-commit**: GitHub Actions that build components and run the full suite, with Ruff formatting and linting configured out of the box.
- **Example notebook**: an interactive notebook for running the pipeline and plotting outputs.

Elements like CI, pre-commit, and the example notebook are optional extras, so you can drop them if you don’t need them after generating your project.

### Getting started with cookiecutter-tesseract

Prerequisites are [Tesseract Core](https://github.com/pasteurlabs/tesseract-core) and a running Docker daemon. Then:

```
# Install cookiecutter (with uv, or pip)
uv tool install cookiecutter

# Generate a project from the template
cookiecutter gh:pasteurlabs/cookiecutter-tesseract

# From inside the generated project:
make new mytess RECIPE=jax   # scaffold a component
make build                   # build all components
make test                    # test components + app
make run                     # run the pipeline end-to-end
```

## Build a multi-Tesseract pipeline, compete for prizes

We're running a virtual hackathon now through August 31, 2026, with $20,000 in cash prizes and research collaboration opportunities. Your challenge? Compose your own differentiable scientific workflow from multiple Tesseracts and use end-to-end gradients to solve a real design, inference, or training problem. [Register today](https://pasteurlabs.ai/tesseract-hackathon-2026/) and try out `cookiecutter-tesseract` to get a head start.
