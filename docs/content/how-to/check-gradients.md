---
og:title: "Checking gradients"
og:description: "Compare a Tesseract's Jacobian, JVP, and VJP endpoints against finite differences of its apply endpoint, with or without a container, and check a pipeline link by link."
---

# Checking gradients

A wrong gradient rarely crashes anything. The optimizer slows down or settles somewhere odd, the training loss plateaus, and nothing in the forward result says why. `check-gradients` compares each derivative endpoint of a Tesseract against finite differences of its `apply` endpoint, so a wrong VJP can be caught on the day it is written rather than weeks into a training run.

## Check a `tesseract_api.py`

No container is needed. Install the runtime into an environment that also has your Tesseract's dependencies, point `TESSERACT_API_PATH` at the file, and pass the inputs at which to check the gradients:

```bash
$ pip install "tesseract-core[runtime]"
$ pip install -r tesseract_requirements.txt
$ export TESSERACT_API_PATH=/path/to/tesseract_api.py
$ tesseract-runtime check-gradients '{"inputs": {"x": [1.0, 2.0, 3.0]}}'
✅ Gradient check for vector_jacobian_product passed ✅ (0 failures / 1000 checks)
```

Every gradient endpoint the Tesseract implements is checked. Prefix the payload with `@` to read it from a file, as in `check-gradients @inputs.json`.

## Check a built image

A Tesseract image runs the same command. Options for the check go through `--runtime-args`:

```bash
$ tesseract run my-tesseract check-gradients '{"inputs": {"x": [1.0, 2.0, 3.0]}}' \
    --runtime-args '--eps 1e-3 --seed 0'
```

The command exits with a non-zero status if any check fails, so either form can run as a CI step.

## What it compares

For each sampled element of each differentiable input, the check perturbs that element by `±eps`, calls `apply` on both sides, and takes the central difference. That gives one column of the Jacobian, the sensitivity of every differentiable output to that input element. The same column is then computed from the endpoint under test:

- `jacobian` is called directly.
- `jacobian_vector_product` is called with a one-hot tangent.
- `vector_jacobian_product` is called with one one-hot cotangent per output element, which is why VJP checks get expensive for large outputs. `--max-output-samples` caps how many output elements are swept.

The two columns pass if they agree to within `--rtol` (default `0.1`) and an absolute tolerance of `1e-8`. `--max-evals` (default `1000`) sets the number of sampled input elements, spread across the inputs in proportion to their size and drawn with replacement, so a small input is sampled many times over.

## Read a failure

Here the VJP returns `3 * x * cotangent` where it should return `2 * x * cotangent`:

```
⚠️ Gradient check for vector_jacobian_product failed ⚠️ (1000 failures / 1000 checks)
First 2 failures:
  Input path: 'x', Output path: 'y', Index: (0,)
  vector_jacobian_product value: [3. 0. 0.]
  Finite difference value: [2.0003319 0.        0.       ]

  Input path: 'x', Output path: 'y', Index: (1,)
  vector_jacobian_product value: [0. 6. 0.]
  Finite difference value: [0.        3.9982796 0.       ]

❌ Some gradient checks failed ❌
```

Each failure names the input path, the output path, and the input index that was perturbed. Both arrays hold the sensitivity of every output element to that input element. A constant ratio between them, 1.5 in this case, usually points to a missing or extra factor. A sign flip points to a transposed or negated term, and agreement in some columns but not others points to a coupling or indexing mistake.

If an endpoint raises an exception, the check reports it with the file and line instead of the values.

## Choose a step size

`--eps` (default `1e-4`) is an absolute step, applied unscaled to every input. No single step suits every problem. A step that is too large measures curvature as well as slope, while one that is too small drowns the difference in rounding error. The second problem is common with `float32` inputs. With `--eps 1e-7`, the correct VJP above fails:

```
  vector_jacobian_product value: [0. 4. 0.]
  Finite difference value: [0.        2.3841858 0.       ]
```

When finite differences come back as zero or as suspiciously round numbers, the step is too small for the input's precision. Try `1e-3` or `1e-2` for `float32`.

Inputs whose magnitudes differ by orders of magnitude need their own steps. `--eps-for` sets one per input path and can be repeated, with every other input falling back to `--eps`:

```bash
$ tesseract-runtime check-gradients @inputs.json --eps 1e-4 --eps-for pressure=1e2 --eps-for viscosity=1e-8
```

A useful habit is to rerun a failing check with the step ten times larger and ten times smaller. If the finite differences change noticeably, the step is the problem. If they stay put and still disagree with the endpoint, the gradient is.

## Narrow the check

- `--input-paths` and `--output-paths` restrict the check to some differentiable fields, which helps when `apply` is expensive or when you are working on one output.
- `--endpoints` checks only the endpoints you name, for example `--endpoints vector_jacobian_product`.
- `--seed` makes the sampled indices reproducible, so a failure can be rerun exactly.
- `--max-failures` (default `10`) controls how many failures are printed per endpoint.

## Check a pipeline link by link

`check-gradients` checks one Tesseract at a time, at the inputs you give it. In a pipeline, check each Tesseract at the inputs it actually receives there, since a gradient can be right in one region of input space and wrong in another. Save the inputs that reach each Tesseract during a forward pass, for instance by converting arrays with `.tolist()` and writing them to JSON, and pass that file to `check-gradients`.

Links that pass on their own can still compose into a wrong gradient if the code between them is wrong. To test the chain as a whole, compare the gradient of the final loss with a central difference along a random direction:

```python
import jax
import jax.numpy as jnp
import numpy as np


def directional_check(loss, x, eps=1e-3, seed=0):
    """Compare jax.grad of a scalar loss with a central difference along a random direction."""
    v = np.random.default_rng(seed).standard_normal(x.shape).astype(x.dtype)
    v /= np.linalg.norm(v)
    fd = (loss(x + eps * v) - loss(x - eps * v)) / (2 * eps)
    ad = jnp.vdot(jax.grad(loss)(x), v)
    return float(fd), float(ad)
```

The two numbers should agree to a few digits. This costs three forward passes and one backward pass however many inputs there are, which is what makes it affordable for a whole pipeline.

## Limitations

- Finite differences are approximations, so a pass is evidence that the gradient is right and not proof. Functions with kinks, branches, or discontinuities near the checked point can fail even when the endpoint returns the correct one-sided or subgradient.
- The check runs in the Tesseract's own process and environment. It cannot yet be pointed at a Tesseract that someone else serves over HTTP.
- `apply` is called twice per sampled input element. For an expensive solver, start with a small `--max-evals` and a few targeted `--input-paths`.

To implement gradients in the first place, see {doc}`/content/concepts/differentiable-programming` and the {doc}`finite-difference helpers </content/examples/building-blocks/finitediff>`.
