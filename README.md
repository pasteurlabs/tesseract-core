<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/pasteurlabs/tesseract-core/blob/main/docs/static/logo-dark.png" width="128" align="right">
  <img alt="" src="https://github.com/pasteurlabs/tesseract-core/blob/main/docs/static/logo-light.png" width="128" align="right">
</picture>

### Tesseract Core

Universal components for differentiable scientific computing 📦

[Read the docs](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/) |
[Demos & tutorials](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/demo/demo/) |
[Blog](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/blog/) |
[Report an issue](https://github.com/pasteurlabs/tesseract-core/issues) |
[Contribute](https://github.com/pasteurlabs/tesseract-core/blob/main/CONTRIBUTING.md)

---

[![DOI](https://joss.theoj.org/papers/10.21105/joss.08385/status.svg)](https://doi.org/10.21105/joss.08385)
[![SciPy](https://img.shields.io/badge/SciPy-2025-blue)](https://proceedings.scipy.org/articles/kvfm5762)

You have a model in PyTorch or JAX and a solver that lives somewhere else, perhaps a Fortran code, a Julia package, a FEniCS model, or a commercial tool. You want to train or optimize through both, and the two won't share an environment. Tesseract wraps the solver behind a typed interface with derivative endpoints, runs it in its own process, and hands it to `jax.grad` or `torch.autograd` as a single differentiable operation.

If you already call the solver through files and `subprocess.run`, that holds up for a few dozen calls and gives out once the prototype becomes a training loop. Tesseract is the persistent, typed, binary version of that glue, with tooling to check that its gradients are right.

## What people have built with it

Each caption names the boundary the gradient crosses. Most of these projects are entries to the [2026 Tesseract Hackathon](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/blog/2026-09-30-tesseract-hackathon-winners/).

<table>
<tr>
<td width="33%" valign="top">
<a href="https://github.com/benvial/prismo"><img src="https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/docs/img/gallery/prismo.jpg" alt="Optimized dopant layout around a silicon waveguide core"></a>
<b><a href="https://github.com/benvial/prismo">PRISMO</a></b><br>
A Julia semiconductor solver and a FEniCS optics solver in one JAX gradient, reaching more than five times the optical phase shift of the starting design.
</td>
<td width="33%" valign="top">
<a href="https://github.com/kasProg/dSnowSac"><img src="https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/docs/img/gallery/dsnowsac.jpg" alt="Simulated and observed streamflow in a held-out basin"></a>
<b><a href="https://github.com/kasProg/dSnowSac">δSnowSac</a></b><br>
A PyTorch network learns to calibrate NOAA's operational Fortran models, Snow-17 and SAC-SMA, with gradients flowing through the untouched Fortran.
</td>
<td width="33%" valign="top">
<a href="https://docs.pasteurlabs.ai/projects/tesseract-core/latest/blog/2025-11-28-rocket-fin-optimization/"><img src="https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/docs/img/gallery/rocket-fin.jpg" alt="Optimized rocket grid fin geometry"></a>
<b><a href="https://docs.pasteurlabs.ai/projects/tesseract-core/latest/blog/2025-11-28-rocket-fin-optimization/">Rocket grid fins</a></b><br>
Ansys SpaceClaim geometry and PyMAPDL structural analysis in one optimization loop, giving a design 24% stiffer than the baseline at the same mass.
</td>
</tr>
<tr>
<td width="33%" valign="top">
<a href="https://github.com/hozaifa1/differentiable-silicon"><img src="https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/docs/img/gallery/differentiable-silicon.jpg" alt="Transistor transfer curves and fabrication parameters during optimization"></a>
<b><a href="https://github.com/hozaifa1/differentiable-silicon">Differentiable Silicon</a></b><br>
Sentaurus TCAD, a closed-source binary driven over SSH, differentiated with finite differences and Broyden updates to tune a transistor for a spiking network.
</td>
<td width="33%" valign="top">
<a href="https://github.com/TAUIL-Abd-Elilah/coldplate"><img src="https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/docs/img/gallery/coldplate.jpg" alt="Cold plate material layout, coolant flow, and chip temperature"></a>
<b><a href="https://github.com/TAUIL-Abd-Elilah/coldplate">Coldplate</a></b><br>
A C++ fluid solver coupled both ways to a thermal solver in Fortran (via Enzyme) or JAX. Leaving the coupling out of the gradient flips a third of its signs.
</td>
<td width="33%" valign="top">
<a href="https://github.com/arpastrana/normax"><img src="https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/docs/img/gallery/normax.jpg" alt="Three gridshell designs from different optimization strategies"></a>
<b><a href="https://github.com/arpastrana/normax">Normax</a></b><br>
JAX form finding, OpenSees structural analysis, and a Eurocode 3 check in one gradient, using 29–67% less material than resizing the tubes of a fixed shape.
</td>
</tr>
<tr>
<td width="33%" valign="top">
<a href="https://github.com/xwpken/opensees-shm-tesseract"><img src="https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/docs/img/gallery/opensees-shm.jpg" alt="Truss bridge response with candidate damage segments"></a>
<b><a href="https://github.com/xwpken/opensees-shm-tesseract">OpenSees-SHM</a></b><br>
An OpenSees finite element model, differentiated by its own sensitivity analysis, infers where and how badly a steel truss has corroded, with uncertainty estimates.
</td>
<td width="33%" valign="top">
<a href="https://github.com/BrainCapture/NeuroLocate"><img src="https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/docs/img/gallery/neurolocate.jpg" alt="Recovered brain source location inside a head model"></a>
<b><a href="https://github.com/BrainCapture/NeuroLocate">NeuroLocate</a></b><br>
A PyTorch network proposes EEG sources, and JAX refines them with gradients through OpenMEEG, a C++ boundary element solver with no autodiff of its own.
</td>
<td width="33%" valign="top">
<a href="https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/demo/enzyme-lfortran/"><img src="https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/docs/img/gallery/enzyme-fortran.jpg" alt="Recovered and true initial temperature fields"></a>
<b><a href="https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/demo/enzyme-lfortran/">Differentiable Fortran</a></b><br>
A Fortran heat solver, differentiated by Enzyme at the LLVM level, recovers a 900-element initial temperature field from 100 sensors through JAX.
</td>
</tr>
</table>

## Quick start

No Docker needed. You need Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/), which Tesseract uses to build an isolated environment for each component.

```bash
$ pip install tesseract-core tesseract-jax
$ tesseract init --name my-tesseract
```

Replace the generated `tesseract_api.py` with this stand-in for a solver:

```python
# tesseract_api.py
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType


class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float32]]


class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float32]]


def apply(inputs: InputSchema) -> OutputSchema:
    # Call your solver, mesh generator, or surrogate here
    return OutputSchema(y=inputs.x**2)


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    # Hand-written, autodiff, adjoint solve, finite differences, ...
    return {"x": 2 * inputs.x * cotangent_vector["y"]}


def abstract_eval(abstract_inputs):
    # Output shapes from input shapes, so JAX can trace without running apply
    return {"y": ShapeDType(shape=abstract_inputs.x.shape, dtype="float32")}
```

Run it in its own process and differentiate through it with JAX:

```python
import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

with Tesseract.from_source("tesseract_api.py") as solver:
    loss = lambda x: apply_tesseract(solver, {"x": x})["y"].sum()
    print(jax.grad(loss)(jnp.array([1.0, 2.0, 3.0])))  # [2. 4. 6.]
```

[Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch) does the same for `torch.autograd`. Before an optimizer relies on the hand-written VJP, check it against finite differences:

```bash
$ pip install "tesseract-core[runtime]"
$ TESSERACT_API_PATH=tesseract_api.py tesseract-runtime check-gradients '{"inputs": {"x": [1.0, 2.0, 3.0]}}'
✅ Gradient check for vector_jacobian_product passed ✅ (0 failures / 1000 checks)
```

When it's time to share the component or deploy it, the same folder builds into a container image (this step requires [Docker](https://docs.docker.com/engine/install/)), and `Tesseract.from_image("my-tesseract")` replaces `from_source`:

```bash
$ tesseract build .
$ tesseract run my-tesseract apply '{"inputs": {"x": [1.0, 2.0, 3.0]}}'
```

## How it works

- **A component declares its interface** as Pydantic input and output schemas and marks which fields are differentiable.
- **It implements `apply`** and any of `jacobian`, `jacobian_vector_product`, and `vector_jacobian_product`, with whatever produces the derivative: a hand-written adjoint, JAX or PyTorch autodiff, Enzyme, or the built-in finite-difference helpers.
- **It runs in its own process**, as a subprocess (`from_source`), a container (`tesseract build`), or a remote service (`from_url`), exposing the same endpoints over HTTP and through the Python SDK in each case.
- **[Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) and [Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch)** turn it into a JAX primitive or a PyTorch operator, so one gradient can cross components written in different languages and frameworks.
- **`check-gradients`** compares every derivative endpoint against finite differences, and schemas and API docs are generated from the code, so a component can be tested and used without reading its source.

## What the boundary costs

Every call crosses a process boundary. On one machine that costs a few milliseconds of HTTP plus the time it takes to move the arrays, which disappears next to a solver that runs for seconds and dominates one that runs for microseconds. For large arrays, Tesseract can pass data through shared memory instead of the request body. On an Apple silicon laptop, a 40 MB round trip to a `from_source` component took about 26 ms that way, against 168 ms with the default base64 encoding. The [performance guide](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/concepts/performance/) and [fast same-machine runs](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/how-to/fast-local-runs/) have the details.

## Do you need it?

If the component imports into your model's environment, a `jax.custom_vjp` or `torch.autograd.Function` is simpler and faster, and you should use that. Tesseract is the better fit when a rewrite of the component would not be accepted as the same thing, because it is validated, closed-source, trained on data you don't have, or the product of years of work, and when it also needs its own interpreter, dependencies, or framework. The [decision guide](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/concepts/when-to-use/) walks through the cases.

There is also a smaller reason to use it. Putting a differentiable component behind a schema and running `check-gradients` on it is a quick way to test it in isolation, even if it later runs in-process via `Tesseract.from_tesseract_api`.

## Ecosystem

- **[Tesseract Core](https://github.com/pasteurlabs/tesseract-core)**: CLI, Python SDK, and runtime (this repo).
- **[Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax)**: embed Tesseracts as JAX primitives, compatible with `jit`, `grad`, and `vmap`.
- **[Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch)**: embed Tesseracts as PyTorch operators with `torch.autograd` support.
- **[Tesseract-Streamlit](https://github.com/pasteurlabs/tesseract-streamlit)**: generate interactive web apps from Tesseracts.

## Learn more

- [Get started](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/tutorials/get-started/)
- [Checking gradients](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/how-to/check-gradients/)
- [Differentiable programming guide](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/concepts/differentiable-programming/)
- [Demos & tutorials](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/demo/demo/)
- [Deploying Tesseracts](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/how-to/deploy/)

## Citing Tesseract

If you use Tesseract in your research, please cite:

```bibtex
@article{TesseractCore,
  doi = {10.21105/joss.08385},
  url = {https://doi.org/10.21105/joss.08385},
  year = {2025},
  publisher = {The Open Journal},
  volume = {10},
  number = {111},
  pages = {8385},
  author = {Häfner, Dion and Lavin, Alexander},
  title = {Tesseract Core: Universal, autodiff-native software components for Simulation Intelligence},
  journal = {Journal of Open Source Software}
}
```

## License

Tesseract Core is licensed under the [Apache License 2.0](https://github.com/pasteurlabs/tesseract-core/blob/main/LICENSE) and is free to use, modify, and distribute (under the terms of the license).

Tesseract is a registered trademark of Pasteur Labs, Inc. and may not be used without permission.
