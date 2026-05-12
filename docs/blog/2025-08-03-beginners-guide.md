---
orphan: true
og:title: "Playing Catch With Newton: a Beginner's Guide to Tesseracts"
og:description: "A hands-on tutorial introducing the Tesseract ecosystem through a physics-based example, covering containerization of computational tools and implementing derivative endpoints."
blog_date: "2025-08-03"
blog_author: "@jacanchaplais"
blog_title: "Playing Catch With Newton: a Beginner's Guide to Tesseracts"
blog_description: "A hands-on tutorial introducing the Tesseract ecosystem through a physics-based example."
---

# Playing Catch With Newton: a Beginner's Guide to Tesseracts

_This post originally appeared on [Pasteur Labs Insights](https://pasteurlabs.ai/insights/beginners-guide-tesseract)._

## Introduction

This tutorial introduces the Tesseract ecosystem through a physics-based example. We'll build a Tesseract from scratch that containerizes a computational tool and implements derivative endpoints for use in differentiable pipelines.

This is the first in a three-part beginner's guide series covering:

1. Writing a custom Tesseract to distribute a computational tool (this post)
2. Applying gradient-based optimization to a differentiable Tesseract
3. Adding an interactive web interface to share an exploratory tool

We assume basic differential calculus knowledge and Python familiarity (including NumPy-style array operations). No prior Tesseract ecosystem experience is required.

## The physics: Projectile motion

Researchers regularly use fantastic software tools --- physics simulators, machine learning models, meshers, and more. But these tools often present installation and configuration barriers for new users. With Tesseracts, you can wrap any tool so it's installable locally via a single command or accessible remotely, all sharing a uniform JSON interface.

To see how this works, let's start with a simple physics example: projectile motion.

### Newton's challenge

<figure>
<img src="../_static/blog/beginners-newton-low.png" alt="Newton at a low angle">
<figcaption>Newton throws a ball at a low angle.</figcaption>
</figure>

Imagine you're playing catch with Isaac Newton. He throws a ball from a fixed distance away, and it arrives after a certain flight time. Newton points out that given a fixed distance and flight time, "only one path" exists --- a unique combination of speed and angle.

<figure>
<img src="../_static/blog/beginners-newton-high.png" alt="Newton at a high angle">
<figcaption>The same distance can be reached at a high angle with a different speed.</figcaption>
</figure>

### The equations

Under constant gravitational acceleration, we can describe the projectile's motion with two key equations.

<figure>
<img src="../_static/blog/beginners-projectile.svg" alt="Projectile trajectory">
<figcaption>Projectile trajectory under constant gravitational acceleration.</figcaption>
</figure>

**Time of flight:**

$$t = \frac{2u \sin \theta}{g}$$

**Horizontal distance:**

$$s_x = \frac{u^2 \sin 2\theta}{g}$$

Where $u$ is the initial velocity, $\theta$ is the launch angle, and $g = 9.81 \, \text{m/s}^2$ is the gravitational acceleration.

<figure>
<img src="../_static/blog/beginners-resolve-velocity.svg" alt="Velocity vector resolution">
<figcaption>Resolving the velocity vector into horizontal and vertical components.</figcaption>
</figure>

These equations establish a one-to-one correspondence between input coordinates $(u, \theta)$ and output coordinates $(s_x, t)$.

## Building the Tesseract

### Project structure

Initialize a new Tesseract project:

```bash
$ tesseract init
```

This generates three files: `tesseract_api.py`, `tesseract_config.yaml`, and `tesseract_requirements.txt`.

### Configuration

Name the Tesseract "projectile" and describe its purpose:

```yaml
name: "projectile"
version: "0.0.1"
description: |
  Given a projectile with an initial speed and angle, computes the
  horizontal distance and time-of-flight for it to return to zero
  vertical displacement.
```

Add JAX as a dependency in `tesseract_requirements.txt`:

```
jax[cuda]==0.5.3
```

### Schema definition

Define the input and output schemas using Pydantic models with `Differentiable` type annotations:

```python
from pydantic import BaseModel
from tesseract_core.runtime import Differentiable, Float32

class InputSchema(BaseModel):
    speed: Differentiable[Float32]
    angle: Differentiable[Float32]

class OutputSchema(BaseModel):
    distance: Differentiable[Float32]
    time: Differentiable[Float32]
```

### Physics functions

Implement the core physics using JAX for automatic differentiation support:

```python
import jax.numpy as jnp

RECIP_GRAV_STRENGTH = 1.0 / 9.81

OUTPUT_FUNCS = {
    "distance": lambda speed, angle: (
        speed * speed * RECIP_GRAV_STRENGTH * jnp.sin(2.0 * angle)
    ),
    "time": lambda speed, angle: (
        2.0 * speed * RECIP_GRAV_STRENGTH * jnp.sin(angle)
    ),
}
```

### Automatic differentiation

Using `jax.grad()`, we can automatically compute the Jacobian matrix:

$$
J(u, \theta) = \begin{bmatrix} \partial s_x / \partial u & \partial s_x / \partial \theta \\ \partial t / \partial u & \partial t / \partial \theta \end{bmatrix}
$$

```python
GRAD_FUNCS = {
    output_name: {
        input_name: jax.grad(func, argnums=num)
        for num, input_name in enumerate(arg_names(func))
    }
    for output_name, func in OUTPUT_FUNCS.items()
}
```

### Endpoints

The `apply` endpoint computes forward evaluation:

```python
def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(
        distance=OUTPUT_FUNCS["distance"](
            speed=inputs.speed, angle=inputs.angle
        ),
        time=OUTPUT_FUNCS["time"](speed=inputs.speed, angle=inputs.angle),
    )
```

The `jacobian` endpoint computes derivatives:

```python
def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> dict[str, dict[str, float]]:
    grads = defaultdict(dict)
    for output_name, input_name in product(jac_outputs, jac_inputs):
        grads[output_name][input_name] = GRAD_FUNCS[output_name][input_name](
            inputs.speed, inputs.angle
        ).item()
    return dict(grads)
```

## Testing it out

Build the Tesseract:

```bash
$ tesseract build .
```

Test with the CLI:

```bash
$ tesseract run projectile apply '{"inputs": {"speed": 10.0, "angle": 0.75}}' | jq .
```

Verify the Jacobian computation at $u = 10$, $\theta = 0.75$ rad --- the numerical results should match the analytical derivatives.

## What's next

By wrapping even a simple physics model in a Tesseract, we get a containerized, self-documenting component with a standard JSON interface and built-in derivative support. The next installments will explore gradient-based optimization and interactive visualization for these differentiable components.

<figure>
<img src="../_static/blog/beginners-newton-gravity.jpg" alt="Isaac Newton discovers gravity, 1936">
<figcaption>"Isaac Newton discovers gravity", 1936.</figcaption>
</figure>

## Learn more

- Check out the [Tesseract Core documentation](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/)
- Join the [Tesseract Community Forum](https://si-tesseract.discourse.group/)
- Visit [Tesseract Core on GitHub](https://github.com/pasteurlabs/tesseract-core)
