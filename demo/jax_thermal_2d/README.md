# JAX: Differentiable 2D Thermal Solver

This example is a **JAX reimplementation** of the [Enzyme Thermal 2D](../enzyme_thermal_2d/) solver. It solves the same physics with an identical Tesseract interface, but obtains derivatives via `jax.vjp` / `jax.jvp` instead of Enzyme's LLVM-level AD.

## What it does

Same governing equation as the Enzyme version:

```
rho * cp * dT/dt = div( k(T) * grad(T) ) + Q
```

with temperature-dependent conductivity `k(T) = k0 + k1*T`, mixed boundary conditions (Dirichlet, convection, insulated), and explicit Euler time integration.

## Why both versions?

The two implementations represent different strategies for making legacy solvers differentiable:

|                          | Enzyme (Fortran)                                           | JAX (Python)                  |
| ------------------------ | ---------------------------------------------------------- | ----------------------------- |
| **Approach**             | Keep existing Fortran code, differentiate at LLVM IR level | Rewrite solver in JAX         |
| **AD mechanism**         | Enzyme LLVM pass                                           | `jax.vjp` / `jax.jvp`         |
| **Build complexity**     | LFortran + LLVM + Enzyme toolchain                         | `pip install jax`             |
| **Legacy code reuse**    | Full — no solver rewrite needed                            | None — complete rewrite       |
| **JIT compilation**      | Ahead-of-time (LLVM)                                       | XLA JIT (first call slower)   |
| **Gradient correctness** | Exact (compiler-generated)                                 | Exact (source transformation) |

## Usage

```python
from tesseract_core import Tesseract
import numpy as np

with Tesseract.from_image("jax-thermal-2d:latest") as t:
    nx, ny = 20, 20
    result = t.apply(inputs={
        "T_init": np.full(nx * ny, 293.15),
        "Q": np.zeros(nx * ny),
        "nx": nx, "ny": ny, "n_steps": 100,
        "k0": 45.0, "k1": -0.01,
        "rho": 7850.0, "cp": 460.0,
        "h_conv": 25.0, "T_inf": 293.15, "T_hot": 373.15,
        "Lx": 0.1, "Ly": 0.05, "dt": 0.01,
    })
    T_final = result["T_final"].reshape(ny, nx)
```

The interface is identical to `enzyme-thermal-2d` — you can swap one for the other by changing only the image name.

## File structure

```
jax_thermal_2d/
├── README.md
├── tesseract_api.py              # JAX solver + auto-derived AD endpoints
├── tesseract_config.yaml         # Build config (just JAX + equinox)
└── tesseract_requirements.txt    # jax[cpu], equinox
```
