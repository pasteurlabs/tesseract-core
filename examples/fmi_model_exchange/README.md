# FMI Model Exchange Tesseract

Wraps an [FMI](https://fmi-standard.org/) 2.0 **Model Exchange (ME)** Functional Mockup
Unit (FMU) as a differentiable Tesseract, using [FMPy](https://github.com/CATIA-Systems/FMPy).
Ships the Modelica [VanDerPol reference FMU](https://github.com/modelica/Reference-FMUs).

## Right-hand side, not a simulation

A Co-Simulation FMU integrates internally. A Model Exchange FMU instead exposes the model's
_right-hand side_, and this Tesseract surfaces exactly that as a single, stateless evaluation:

```
dx/dt = f(t, x, u; p)
```

Time integration is left to the caller. The intended use is to integrate _over_ the Tesseract
from an external JAX/[diffrax](https://github.com/patrick-kidger/diffrax) program via
[tesseract-jax](https://github.com/pasteurlabs/tesseract-jax) — see
[`parameter_estimation.py`](./parameter_estimation.py). Keeping `apply` a pure function of its
inputs is what makes it differentiable and safe to call concurrently inside a solver.

## Schema

| field            | role                                                                            |
| ---------------- | ------------------------------------------------------------------------------- |
| `t`              | simulation time (scalar)                                                        |
| `x`              | continuous state vector                                                         |
| `u`              | continuous input vector (empty for VanDerPol)                                   |
| `parameters`     | tunable real parameters exposed for differentiation (`["mu"]` by default)       |
| `dx_dt` (output) | state derivatives `f(t, x, u; p)` — **differentiable**                          |
| `y` (output)     | declared FMU outputs (non-differentiable; for VanDerPol these equal the states) |

State/input/output orderings and FMI value references are read from `modelDescription.xml`
at import, so the schema sizes adapt to whatever FMU is loaded.

## Differentiability

FMI's `fmi2GetDirectionalDerivative` computes a Jacobian-vector product (`J @ v`) — i.e. a
forward-mode **JVP** — exactly. The endpoints are wired accordingly:

- `jacobian_vector_product` → `fmi2GetDirectionalDerivative` directly (states/inputs).
- `jacobian` → swept unit tangents through the directional derivative.
- `vector_jacobian_product` → derived from the Jacobian (`vjp_from_jacobian`). This is the local
  VJP that diffrax's reverse-mode adjoint consumes when integrating over the Tesseract.

Directions the FMU cannot differentiate this way fall back to **finite differences**: the time
`t` (FMI has no value reference for it) and `parameters` (VanDerPol declares no derivative
dependency on `mu`, and `mu` has `fixed` variability). For VanDerPol, the state Jacobian matches
the closed form `[[0, 1], [-2·mu·x0·x1 - 1, mu·(1 - x0²)]]` to machine precision.

> FMI 3.0 adds `fmi3GetAdjointDerivative` (a native VJP), which would back
> `vector_jacobian_product` directly without materializing the Jacobian — worthwhile only at high
> state dimension. For VanDerPol (2 states) the forward-derivative route is equivalent.

## Build and run

```bash
tesseract build examples/fmi_model_exchange
tesseract run fmi-model-exchange apply @examples/fmi_model_exchange/example_inputs.json
# dx_dt = [1.0, -11.0] for x=[2,1], mu=3
```

## Parameter estimation demo (host-side)

```bash
pip install "tesseract-jax>=0.3" diffrax jax
python examples/fmi_model_exchange/parameter_estimation.py
```

`tesseract-jax >= 0.3` is required (it targets the current `openapi_schema` client API).
With `uv` and a private index that doesn't mirror it, force PyPI for that one run:

```bash
uv run --index https://pypi.org/simple --with "tesseract-jax>=0.3" --with diffrax \
  python examples/fmi_model_exchange/parameter_estimation.py --steps 3
```

Integrates the FMU RHS with diffrax, then recovers `mu` from a noisy synthetic trajectory by
`jax.grad` through the solve — gradients flow from diffrax's adjoint into the Tesseract's
`vector_jacobian_product`, and from there into FMI directional derivatives (for `d/dx`) and
finite differences (for `d/dmu`).

## Swapping in a different FMU

Drop a Model Exchange `.fmu` into [`fmus/`](./fmus/) and point at it (and, if needed, declare
which parameters to expose) — no endpoint code changes:

```yaml
# tesseract_config.yaml
build_config:
  package_data:
    - ["fmus/MyModel.fmu", "fmus/MyModel.fmu"]
```

```bash
FMU_PATH=fmus/MyModel.fmu FMU_PARAMETERS=k1,k2 tesseract run ...
```
