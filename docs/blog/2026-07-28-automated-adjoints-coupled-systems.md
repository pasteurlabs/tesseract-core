---
orphan: true
og:title: "Automated adjoints of coupled systems"
og:description: "Coupled multi-physics used to mean a hand-written adjoint or a monolithic rewrite. Compose independent Tesseracts and jax.grad differentiates the whole two-way coupling end-to-end."
blog_date: "2026-07-28"
blog_author: "@jpbrodrick89"
blog_title: "Automated adjoints of coupled systems"
blog_description: "Coupled multi-physics used to mean a hand-written adjoint or a monolithic rewrite. Compose independent Tesseracts and jax.grad differentiates the whole two-way coupling end-to-end."
---

# Automated adjoints of coupled systems

Real engineering systems are rarely governed by a single physics. A turbine blade is thermal _and_ structural _and_ fluid; a battery is electrochemical _and_ thermal _and_ mechanical. The behavior that matters — and the behavior you want to design for — lives in the _coupling_ between these processes, each usually modeled by its own specialized solver.

Designing such systems means solving inverse problems _through_ the coupled physics: what inputs produce the behavior we want, once everything has settled into equilibrium? Answering that efficiently needs gradients through the entire coupled chain. Historically you had two options, both bad:

- **Hand-code an adjoint.** Derive and maintain the coupled adjoint by hand — fragile, error-prone, and re-derived every time a solver changes.
- **Rewrite everything into one monolith.** Merge your independently developed solvers into a single differentiable codebase, throwing away the team and dependency boundaries that made them tractable in the first place.

Tesseracts offer a third path. Each physics component stays exactly what it is — its own implementation, its own dependencies, its own container — but because each one exposes its derivatives, they compose into a single differentiable pipeline. `jax.grad` then propagates gradients through the whole two-way coupling automatically. No manual adjoint. No monolithic rewrite.

Our new [multi-physics optimization demo](../content/demo/multiphysics-optimization) shows this on a thermoelastic inverse-design problem. A thermal solver and a structural solver are built as two _independent_ Tesseracts, coupled both ways: temperature drives thermal expansion, and the resulting deformation changes how heat flows. We iterate them to a coupled equilibrium and then ask an inverse question — _find the heat-source location and intensity that produce a target set of sensor temperatures, after the coupling has converged._

Because the composed pipeline is differentiable end-to-end, a gradient-based optimizer solves this in a fraction of the evaluations a gradient-free method needs, and the gradients match finite differences to validation tolerance. None of it required touching either solver's internals — they stay black boxes that exchange fields.

It also scales. The gradient through the coupled equilibrium is computed via the implicit function theorem — a result nearly two centuries old — so the backward pass solves a single linear system at the fixed point instead of replaying the iteration, and memory stays constant no matter how many coupling steps convergence takes. (We lean on [optimistix](https://docs.kidger.site/optimistix/) for this; it's a few lines.)

The demo is deliberately minimal, but the pattern is general: wrap each physics component as a Tesseract, compose them with Tesseract-JAX, and differentiate the whole thing. Independent teams keep shipping independent solvers — and the coupling between them is differentiable end-to-end, with no adjoint to derive and no monolith to maintain.

That's the shift: the coupled adjoint you used to write by hand, or avoid entirely, is now just `jax.grad`.

[**Read the demo →**](../content/demo/multiphysics-optimization)
