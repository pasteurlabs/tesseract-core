# Tesseract Design Patterns

This page provides guidance on common questions around Tesseract design: what should (and shouldn't) be a Tesseract, and how to structure your workflow.

## When to use a Tesseract

Tesseracts are most useful when you need to:

- **Share software across teams** — Package your code so others can use it without understanding implementation details
- **Combine heterogeneous software** — Integrate components written in different languages/frameworks into a unified pipeline
- **Deploy to diverse hardware** — Run the same component on different machines (local, cloud, HPC) without modification
- **Enable gradient-based optimization** — Expose derivatives for use in optimization or calibration workflows
- **Ensure reproducibility** — Capture dependencies and environment in a container for consistent execution

## When NOT to use a Tesseract

Tesseracts add overhead that isn't always justified:

- **Single-user, single-environment workflows** — If you're the only one running the code on a single machine, the containerization overhead may not be worth it
- **Sub-second latency requirements** — Tesseracts are designed for compute kernels that run for at least several seconds; for very fast operations, the container and HTTP overhead becomes significant
- **Tightly coupled iterations** — If your inner loop calls a function millions of times, that function shouldn't be a Tesseract; instead, wrap the entire loop
- **Simple scripts** — A quick data transformation script that won't be reused doesn't need Tesseract packaging

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

### A practical rule of thumb

Think about the **natural unit of work** that makes sense to share. A Tesseract should wrap functionality that:

1. Has a clear, well-defined interface (inputs and outputs)
2. Represents a meaningful computation (not just a utility function)
3. Could reasonably be owned and maintained by one person or team
4. Takes at least a few seconds to run (to amortize container overhead)

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

Pros: Simple to deploy, no intermediate data transfer
Cons: Can't swap solver, can't run meshing on CPU while solving on GPU

**Option B: Four Tesseracts**

```
CAD → [Mesh] → [Solve] → [Post-process] → [Visualize] → Report
```

Pros: Maximum flexibility, clear ownership
Cons: Data transfer overhead, more complex orchestration

**Option C: Two Tesseracts (recommended)**

```
CAD → [Mesh] → [Solve + Post-process + Visualize] → Report
```

Pros: Separates geometry (often CPU-bound, different expertise) from simulation (often GPU-bound, different team), minimal data transfer for tightly coupled steps

The right choice depends on your team structure, hardware constraints, and reuse patterns. When in doubt, start with fewer Tesseracts and split them later if needed — it's easier to break apart than to combine.

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

## Common anti-patterns

### The "kitchen sink" Tesseract

Don't create a single Tesseract that does everything your project needs. This defeats the purpose of modularity and makes it hard to maintain or reuse.

### The "micro-Tesseract"

Don't wrap trivial operations like `add(a, b)` as Tesseracts. The overhead isn't justified, and such operations should just be regular functions in your pipeline code.

### Ignoring the single-entrypoint design

Tesseracts have one `apply` function. If you need multiple entrypoints, create multiple Tesseracts. Don't try to work around this with mode flags:

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
