---
orphan: true
og:title: "Automated adjoints of coupled systems"
og:description: "Coupled multi-physics used to mean a hand-written adjoint or a monolithic rewrite. Compose independent Tesseracts and jax.grad differentiates the whole two-way coupling end-to-end."
blog_date: "2026-08-20"
blog_author: "@jpbrodrick89"
blog_title: "Automated adjoints of coupled systems"
blog_description: "Coupled multi-physics used to mean a hand-written adjoint or a monolithic rewrite. Compose independent Tesseracts and jax.grad differentiates the whole two-way coupling end-to-end."
---

# Automated adjoints of coupled systems

Real engineering systems are rarely governed by a single physics. A turbine blade is thermal _and_ structural _and_ fluid; a battery is electrochemical _and_ thermal _and_ mechanical. The behaviour that matters lives in the coupling between these processes, each usually modelled by its own specialized solver.

Designing such systems means solving inverse problems _through_ the coupled physics, so that the desired behaviour at equilibrium is obtained. Solving that efficiently needs gradients through the entire coupled chain. Historically you had two options:

- **Hand-code an adjoint.** Derive and maintain the coupled adjoint by hand, which is fragile, error-prone, and re-derived every time a solver changes.
- **Rewrite everything into one monolith.** Merge your independently developed solvers into a single differentiable codebase, throwing away the team and dependency boundaries that made them tractable in the first place.

Tesseracts offer you a third path that allows each physics component to stay exactly what it is, with its own implementation, dependencies, and container. Because each exposes its derivatives, you can compose all of the components into a single differentiable pipeline. `jax.grad` can then propagate gradients through the whole two-way coupling automatically without a manual adjoint or monolithic rewrite.

Our new [multi-physics optimization demo](../content/demo/multiphysics-optimization) shows this on the thermal–structural case from the turbine example. A thermal solver and a structural solver are built as two _independent_ Tesseracts, coupled both ways: temperature drives thermal expansion, and the resulting deformation changes how heat flows. We iterate them to a coupled equilibrium and then ask an inverse question. _Find the heat-source location and intensity that produce a target set of sensor temperatures at equilibrium._

```{figure} ../static/blog/coupled-systems-hero.png
:alt: "Two panels. Left: a temperature field from the thermal Tesseract. Right: a deformed mesh from the structural Tesseract. Two arrows between them show temperature flowing one way and displacement feeding back the other, and a dashed arrow wrapping the pair shows the jax.grad backward pass."

Two independent Tesseracts exchanging fields both ways: temperature drives the structural deformation, and the resulting displacement feeds back into the thermal solve. Because each Tesseract exposes its derivatives, `jax.grad` differentiates through the whole loop end-to-end, with no hand-written adjoint.
```

Because the composed pipeline is differentiable end-to-end, a gradient-based optimizer (L-BFGS-B) solves this in 27 evaluations, against 372 for a gradient-free baseline (Nelder-Mead) on the same problem. Both land on the same design, and the end-to-end gradients match finite differences to a relative error of about 1e-4. None of it required touching either solver's internals. They stay black boxes that exchange fields.

It also stays cheap in memory. By leveraging [optimistix](https://docs.kidger.site/optimistix/), the gradient through the coupled equilibrium is computed via the implicit function theorem (no problem-specific adjoint) in a few lines. The backward pass then solves a single linear system at the converged fixed point, instead of storing and differentiating every coupling iteration, so memory stays constant no matter how many steps convergence takes. The 30×30 demo is too small to feel this, but it is what keeps the backward pass tractable on production-scale meshes.

The demo is deliberately minimal, but the pattern is general: wrap each physics component as a Tesseract, compose them with Tesseract-JAX, and differentiate the whole thing. You can even swap a hand-written solver for a learned surrogate without touching the optimization code. Independent teams keep shipping independent solvers, and the coupling between them is differentiable end-to-end, with no adjoint to derive and no monolith to maintain.

[**Try the demo →**](../content/demo/multiphysics-optimization)
