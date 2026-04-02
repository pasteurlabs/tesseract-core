---
html_class: blog-page
hide-toc: true
og:title: Building Differentiable Multi-Physics Pipelines with Tesseract
og:description: "Gradient-based shape optimization of a rocket fin using three solvers in different languages — C++ with analytic adjoints, Julia with Enzyme, JAX — composed into an end-to-end differentiable pipeline."
---

# Building Differentiable Multi-Physics Pipelines with Tesseract

:::{div} blog-post-meta
Mar 15, 2026 · Dion Hafner
:::

One of Tesseract's key promises is composing heterogeneous simulators into end-to-end differentiable pipelines. In this post, we walk through a concrete example: gradient-based shape optimization of a rocket fin, where three separate solvers -- each in a different language, each with a different autodiff strategy -- are chained together with gradients flowing across the boundaries.

## The problem

Suppose you're designing a rocket fin and want to optimize its shape to minimize drag while maintaining structural integrity under thermal loads. This involves three physics domains:

1. **CFD solver** (C++ with analytic adjoints) computes aerodynamic forces
2. **Thermal solver** (Julia with Enzyme AD) computes temperature distribution
3. **Structural solver** (Python/JAX) computes stress and displacement

In a traditional workflow, you'd run each solver manually, pass files between them, and do gradient-free optimization (or finite differences) over the full pipeline. This is slow and doesn't scale.

## The Tesseract approach

Each solver gets wrapped as a Tesseract:

```python
# tesseract_api.py for the CFD solver
from tesseract_core import Tesseract

class InputSchema(Tesseract.InputSchema):
    shape_params: list[float]
    flow_conditions: dict

class OutputSchema(Tesseract.OutputSchema):
    forces: list[float]
    pressure_field: list[float]

def apply(inputs: InputSchema) -> OutputSchema:
    # Calls the C++ solver internally
    result = run_cfd_solver(inputs.shape_params, inputs.flow_conditions)
    return OutputSchema(
        forces=result.forces,
        pressure_field=result.pressure_field,
    )
```

The CFD Tesseract exposes a `jacobian` endpoint that calls the solver's built-in adjoint method. The thermal Tesseract uses Enzyme for automatic differentiation of the Julia code. The structural Tesseract uses JAX's native AD. None of these need to know about each other's internals.

### Composing the pipeline

With `tesseract-jax`, each Tesseract becomes a JAX primitive:

```python
import jax
import jax.numpy as jnp
from tesseract_jax import connect

# Connect to running Tesseracts
cfd = connect("http://localhost:8001")
thermal = connect("http://localhost:8002")
structural = connect("http://localhost:8003")

def full_pipeline(shape_params):
    # Chain the solvers
    aero = cfd.apply(shape_params=shape_params, flow_conditions=conditions)
    temps = thermal.apply(
        pressure_field=aero["pressure_field"],
        boundary_conditions=bc,
    )
    stress = structural.apply(
        shape_params=shape_params,
        temperature_field=temps["temperature"],
        forces=aero["forces"],
    )
    return stress["max_von_mises"]

# Gradient of the full pipeline -- flows through all three solvers
grad_fn = jax.grad(full_pipeline)
shape_gradient = grad_fn(initial_shape)
```

That's it. `jax.grad` computes the gradient of `max_von_mises` with respect to `shape_params`, automatically calling each Tesseract's `jacobian` endpoint in the right order via the chain rule. The C++ adjoint, Julia Enzyme AD, and JAX autodiff all compose seamlessly.

## Why this matters

This pattern unlocks several things that are hard or impossible without Tesseract:

- **No shared codebase required.** The CFD team ships their solver as a Tesseract. The thermal team ships theirs. Nobody needs to install anyone else's dependencies or understand their build system.

- **Mixed AD strategies.** Analytic adjoints for the CFD solver (most efficient), Enzyme for the Julia code (automatic and fast), JAX for the structural model (convenient for ML-style iteration). Each team picks the best tool for their code.

- **Scalable deployment.** Each Tesseract can run on different hardware. The CFD solver might need a GPU cluster; the structural model runs fine on a CPU node. Tesseract handles the communication.

> The key insight is that differentiation is a _local_ property. Each component only needs to know how to differentiate itself. Tesseract handles the global composition via the chain rule.

## Performance considerations

There's overhead in the HTTP communication between Tesseracts. For solvers that run in seconds or minutes, this is negligible. For sub-second kernels, you might want to batch calls or use the local-process mode.

We measured the overhead for our rocket fin pipeline:

| Component         | Solve time | Communication overhead |
| ----------------- | ---------- | ---------------------- |
| CFD solver        | 45s        | 12ms                   |
| Thermal solver    | 8s         | 8ms                    |
| Structural solver | 2s         | 5ms                    |

The total overhead is ~25ms on a ~55s pipeline -- less than 0.05%.

## Next steps

We're working on several improvements to make this workflow even smoother:

- **TesseractHub** -- a registry for discovering and sharing pre-built Tesseracts, so you can find a CFD solver without building your own
- **Async execution** -- running independent Tesseracts in parallel within a pipeline
- **Higher-order derivatives** -- Hessian-vector products for second-order optimization

If you're building multi-physics pipelines and want to try this out, start with the [Get Started guide](../introduction/get-started.md) and check out the [end-to-end demos](../demo/demo.md).

We'd love to hear about your use cases on the [Tesseract User Forums](https://si-tesseract.discourse.group/).
