# Design Patterns

This page provides guidance on common questions around Tesseract design: what should (and shouldn't) be a Tesseract, and how to structure your workflow.

## When to use a Tesseract

{doc}`/content/concepts/when-to-use` covers when a Tesseract is worth its process boundary, and when a `jax.custom_vjp`, a `torch.autograd.Function`, or a rewrite is the better answer. The same reasoning shapes how you split a problem into Tesseracts. If an inner loop calls a function millions of times, that function shouldn't be a Tesseract. Wrap the entire loop instead.

## How granular should Tesseracts be?

One of the most common questions is: _"Should I make one big Tesseract or many small ones?"_

### Prefer fewer, coarser Tesseracts when:

- Operations are tightly coupled and always run together
- Data transfer between steps would be expensive (e.g., large meshes or tensors)
- The combined operation is what users actually want to call
- You need maximum performance (fewer container invocations)

### Prefer more, finer Tesseracts when:

- Components have different hardware requirements (e.g., one needs GPU, one doesn't)
- Components have conflicting dependencies
- Different team members own different parts
- You want to swap out implementations (e.g., different solvers for the same interface)
- Components are reusable across multiple workflows

(designing-good-interfaces)=

## Designing good interfaces

### Keep schemas focused

Your `InputSchema` and `OutputSchema` should contain only what's needed for the computation. Avoid:

- Configuration that rarely changes (put it in the Tesseract itself or make it a build-time option)
- Metadata that's not used in computation
- Redundant fields that can be derived from others

### Use meaningful types

```python
# Less clear
class InputSchema(BaseModel):
    data: Array[(None, None), Float64]  # What is this?

# More clear
class InputSchema(BaseModel):
    mesh_coordinates: Array[(None, 3), Float64] = Field(
        description="Node coordinates of the mesh (N nodes x 3 dimensions)"
    )
```

### Design for composability

If your Tesseract might be chained with others, design interfaces that make this natural:

```python
# Mesh generator output
class MeshOutput(BaseModel):
    nodes: Array[(None, 3), Float64]
    elements: Array[(None, 4), Int32]

# Solver input (matches mesh output)
class SolverInput(BaseModel):
    nodes: Array[(None, 3), Float64]
    elements: Array[(None, 4), Int32]
    boundary_conditions: BoundaryConditions
```

## Example: Simulation workflow

Consider a CFD simulation workflow with these steps:

1. Generate mesh from CAD geometry
2. Run CFD solver
3. Post-process results
4. Visualize output

**Option A: One Tesseract**

```
CAD → [Mesh + Solve + Post-process + Visualize] → Report
```

Pros: Simple to deploy, no intermediate data transfer\
Cons: Can't swap solver, can't run meshing on CPU while solving on GPU

**Option B: Four Tesseracts**

```
CAD → [Mesh] → [Solve] → [Post-process] → [Visualize] → Report
```

Pros: Maximum flexibility, clear ownership\
Cons: Data transfer overhead, more complex orchestration

**Option C: Two Tesseracts (recommended)**

```
CAD → [Mesh] → [Solve + Post-process + Visualize] → Report
```

Pros: Separates geometry (often CPU-bound, different expertise) from simulation (often GPU-bound, different team), minimal data transfer for tightly coupled steps

The right choice depends on your team structure, hardware constraints, and reuse patterns. When in doubt, start with fewer Tesseracts and split them later if needed — it's easier to break apart than to combine.

```{seealso}
For a real-world example of multi-Tesseract composition, see the [QoI-based Workflows with Ansys Fluent](https://si-tesseract.discourse.group/t/qoi-based-workflows-with-ansys-fluent-and-tesseract/110) showcase, which demonstrates chaining geometry, solver, and post-processing Tesseracts into a unified pipeline.
```

## Common anti-patterns

### The "kitchen sink" Tesseract

Don't create a single Tesseract that does everything your project needs. This defeats the purpose of modularity and makes it hard to maintain or reuse.

### Overloading a single Tesseract with mode flags

Tesseracts have one `apply` function. While mode flags can be acceptable in some cases, the default should be to create separate Tesseracts for distinct operations:

```python
# Anti-pattern: mode switching
class InputSchema(BaseModel):
    mode: Literal["train", "predict", "evaluate"]
    ...

# Better: separate Tesseracts
# - model-trainer (for training)
# - model-predictor (for inference)
# - model-evaluator (for evaluation)
```

### Stateful operations

Tesseracts are designed to be stateless and context-free. Each call to `apply` should be independent. If you need state:

- Pass it explicitly in the input schema
- Store it in external storage (files, databases) and reference it by path
- Reconsider whether a Tesseract is the right abstraction
