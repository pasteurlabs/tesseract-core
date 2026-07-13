# Composing Tesseracts into pipelines

A single Tesseract packages one computation. Real work usually involves several: a mesher feeds a solver, an encoder feeds a model, a simulation feeds a post-processor. This page is about the step _after_ you've built your Tesseracts — calling them from your own code and chaining them into a larger workflow.

Tesseract Core is deliberately unopinionated about what you build on top. A project that uses Tesseracts might be a training loop, an optimization routine, a notebook, a web service, or a full application, with as much or as little logic of its own as you like. We won't prescribe that. What follows is the small amount of generic advice that applies regardless: _how to call and chain Tesseracts well_.

The [Design Patterns](design-patterns.md) page covers the complementary question of how to split a workflow into Tesseracts in the first place. Read that for the "how many, how granular" decisions; read this for "now how do I wire them together."

## Choosing how to call a Tesseract

Every Tesseract exposes the same three interfaces: a [CLI, a REST API, and a Python SDK](../using-tesseracts/use.md). For composing Tesseracts into a program, there are two approaches worth knowing, and which one you reach for depends on whether you're working inside an autodiff framework.

### If you're using JAX or PyTorch: use the framework bindings

This is the recommended default when your surrounding code already lives in JAX or PyTorch. **[Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax)** and **[Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch)** wrap a Tesseract so it behaves like a native operation in that framework — traceable, and above all _differentiable_, so gradients flow through the containerized computation and into the rest of your program.

Both expose the same one-function interface, `apply_tesseract(tesseract, inputs)`. With JAX:

```python
import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

t = Tesseract.from_image("vectoradd_jax")
t.serve()

x = jnp.ones((1000,))
y = jnp.ones((1000,))

def vector_sum(x, y):
    res = apply_tesseract(t, {"a": {"v": x}, "b": {"v": y}})
    return res["vector_add"]["result"].sum()

vector_sum(x, y)                  # call it like any JAX function
jax.jit(vector_sum)(x, y)         # ... jit it
jax.grad(vector_sum)(x, y)        # ... and differentiate through it

t.teardown()                      # stop the container when you're done
```

The PyTorch equivalent integrates with autograd instead:

```python
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

t = Tesseract.from_image("vectoradd_torch")
t.serve()

x = torch.ones(1000, requires_grad=True)
y = torch.ones(1000)

res = apply_tesseract(t, {"a": {"v": x}, "b": {"v": y}})
loss = res["vector_add"]["result"].sum()
loss.backward()                   # gradients propagate back through the Tesseract
print(x.grad)

t.teardown()                      # stop the container when you're done
```

```{tip}
These bindings are what makes Tesseracts *differentiable software components*. If your workflow is an optimization, calibration, or training problem, reaching for Tesseract-JAX or Tesseract-Torch usually means you get end-to-end gradients for free. See the [Differentiable Programming guide](../misc/differentiable-programming.md).
```

### Otherwise: use the Python SDK

When you're not inside JAX or PyTorch (a plain Python script, a service, a batch job), the [`Tesseract`](#tesseract_core.Tesseract) SDK class is the general-purpose way to invoke a Tesseract:

```python
import numpy as np
from tesseract_core import Tesseract

with Tesseract.from_image("scaler") as scaler:
    result = scaler.apply({"vector": np.array([1.0, 2.0, 3.0]), "scale_factor": 2.0})

print(result["scaled_vector"])   # -> [2. 4. 6.]
```

[`Tesseract.from_image`](#tesseract_core.Tesseract.from_image) references a built image by name; the `with` block starts the container and tears it down on exit. `apply` takes a dict of inputs and returns a dict of outputs — NumPy in, NumPy out, no manual serialization.

```{note}
To call a Tesseract that already runs elsewhere (a shared service, a GPU node, a remote deployment), use [`Tesseract.from_url(...)`](#tesseract_core.Tesseract.from_url) instead of `from_image`. The calling code is otherwise identical, so a workflow developed locally moves to distributed execution without a rewrite.
```

## Chaining Tesseracts

Chaining is deliberately unremarkable in all three cases: a Tesseract's outputs are ordinary values (a dict of arrays), so you feed them into the next call like any other Python data. There's no special pipeline object to learn.

With the SDK:

```python
with Tesseract.from_image("scaler") as scaler, \
     Tesseract.from_image("normalizer") as normalizer:
    scaled = scaler.apply({"vector": vector, "scale_factor": 2.0})
    normalized = normalizer.apply({"vector": scaled["scaled_vector"]})
```

And the same shape inside a differentiable JAX program — call `apply_tesseract` for each step and let the framework thread gradients through the whole chain:

```python
def pipeline(vector):
    scaled = apply_tesseract(scaler, {"vector": vector, "scale_factor": 2.0})
    normalized = apply_tesseract(normalizer, {"vector": scaled["scaled_vector"]})
    return normalized["normalized_vector"]

jax.grad(lambda v: pipeline(v).sum())(vector)   # gradient through both Tesseracts
```

Because chaining is just data flow, ordinary control flow works too. Put a Tesseract call in a loop, behind a conditional, inside `jax.lax.scan`, or wherever your program needs it.

When a chain misbehaves, test each Tesseract in isolation first. A component you've verified on its own against a known input/output pair is a fixed point you can trust while debugging the workflow around it.

The one thing that makes or breaks chaining is at the _interface_, not the call site: a step composes cleanly with the next only if its output fields line up with the downstream input schema. Design for that.

```{seealso}
[Designing good interfaces](design-patterns.md#designing-good-interfaces) — matching one Tesseract's `OutputSchema` to the next's `InputSchema` is what keeps chains readable. Design interfaces with the downstream consumer in mind.
```

## Building a multi-Tesseract project

Once a project grows past a couple of Tesseracts, some structure pays off: a consistent place for each component, a way to share code between them, and a build-and-test loop that covers the whole set. Rather than assemble this by hand, the [`cookiecutter-tesseract`](https://github.com/pasteurlabs/cookiecutter-tesseract) template generates a ready-made project with all of it wired up:

```bash
$ pip install cookiecutter
$ cookiecutter github:pasteurlabs/cookiecutter-tesseract
```

The generated project gives you an opinionated, working structure for a multi-Tesseract codebase, including:

- **A two-layer project layout** that separates the individual Tesseracts (each independently built and tested) from the application code that uses them.
- **A one-command workflow** for scaffolding, building, and testing components, so you don't call the underlying tooling by hand.
- **A shared-code package** that every component can depend on, so common helpers live in one place instead of being copied into each Tesseract.
- **Per-component test fixtures** for checking each Tesseract in isolation, plus a place for tests of the application layer.
- **Pre-build hooks** for components that need setup before their container builds (fetching weights, compiling an extension).
- **Continuous integration** that builds and tests every component across supported Python versions, catching a broken interface the moment it stops matching its consumer.

Take the parts that fit your project and leave the rest. For anything beyond a handful of Tesseracts, starting from the template usually beats reinventing this plumbing.

## What's next

- [Design Patterns](design-patterns.md) — how to split a workflow into Tesseracts and design their interfaces.
- [Interacting with Tesseracts](../using-tesseracts/use.md) — the full SDK, CLI, and REST interfaces.
- [Differentiable Programming](../misc/differentiable-programming.md) — propagating gradients through a composed, multi-Tesseract program.
- [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) and [Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch) — the framework bindings.
- [Performance](../misc/performance.md) — minimizing container and data-transfer overhead in chained workflows.
- Questions? Ask on the [Tesseract User Forums](https://si-tesseract.discourse.group/).
