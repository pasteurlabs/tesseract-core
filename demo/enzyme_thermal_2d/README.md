# Enzyme AD: Differentiable 2D Thermal Solver

This example demonstrates how to obtain **exact automatic derivatives** of a production-style Fortran thermal simulation without writing any manual adjoint code, using [Enzyme](https://enzyme.mit.edu/) for automatic differentiation at the LLVM IR level.

## What it does

The Tesseract wraps a 2D transient heat conduction solver:

```
rho * cp * dT/dt = div( k(T) * grad(T) ) + Q
```

with:

- **Temperature-dependent conductivity**: `k(T) = k0 + k1*T` (nonlinear material model)
- **Mixed boundary conditions**: Dirichlet (hot wall), convection (Robin), and insulated (Neumann)
- **Multi-step explicit time integration**: the Fortran kernel runs the full time-stepping loop internally

This is representative of the thermal solvers CAE engineers run in production — a structured-grid finite difference code with nonlinear material properties, mixed BCs, and a time integration loop. The key difference from a toy example is that **the nonlinear conductivity makes the Jacobian solution-dependent**, so you can't derive it analytically and hand-coding the adjoint is error-prone.

Enzyme differentiates through the **entire time-stepping loop** (including the nonlinear stencil operations at each step), giving exact gradients of the final temperature field with respect to all material properties, boundary conditions, and initial conditions.

## How it works

The compilation pipeline is identical to the [1D Enzyme example](../enzyme_ad/):

```
thermal_2d.f90         (Fortran source — 2D solver with k(T), mixed BCs, time loop)
    │
    ▼  lfortran --show-llvm
thermal_2d.ll          (LLVM IR)
    │
    ▼  opt -O1
thermal_2d_opt.ll      (cleaned-up IR, ready for Enzyme)
    │
    ▼  llvm-link with wrapper.c
combined.ll            (Fortran IR + C wrapper with __enzyme_autodiff / __enzyme_fwddiff)
    │
    ▼  opt --load-pass-plugin=LLVMEnzyme-19.so -passes=enzyme
ad.ll                  (LLVM IR with compiler-generated forward and reverse mode derivatives)
    │
    ▼  clang -shared
libthermal_2d_ad.so    (shared library: forward, JVP, VJP entry points)
```

At runtime, `tesseract_api.py` loads `libthermal_2d_ad.so` via `ctypes` and exposes:

- **`apply`** — runs the full forward simulation (all time steps)
- **`jacobian_vector_product`** — Enzyme forward-mode (JVP) through the full simulation
- **`vector_jacobian_product`** — Enzyme reverse-mode (VJP) through the full simulation

## Physics

### Governing equation

2D transient heat conduction with temperature-dependent conductivity and volumetric heat source:

```
rho * cp * dT/dt = d/dx(k(T) * dT/dx) + d/dy(k(T) * dT/dy) + Q
```

### Material model

```
k(T) = k0 + k1 * T
```

Default values model mild steel: `k0 = 45 W/(m·K)`, `k1 = -0.01 W/(m·K²)`.

### Boundary conditions

| Boundary     | Type                | Condition            |
| ------------ | ------------------- | -------------------- |
| Bottom (y=0) | Dirichlet           | T = T_hot            |
| Top (y=Ly)   | Convection (Robin)  | -k ∂T/∂n = h(T - T∞) |
| Left (x=0)   | Insulated (Neumann) | ∂T/∂x = 0            |
| Right (x=Lx) | Insulated (Neumann) | ∂T/∂x = 0            |

### Discretization

- Structured rectangular grid (nx × ny)
- Central differences in space with harmonic-mean conductivity at cell faces
- Explicit Euler in time

## Why AD matters here

With constant thermal conductivity, the Jacobian of the heat equation is a constant tridiagonal (or banded) matrix that you could derive by hand. But with `k(T) = k0 + k1*T`:

1. The stencil coefficients depend on the current temperature field
2. The Jacobian changes at every time step
3. Hand-coding the adjoint through the nonlinear stencil and multi-step loop is tedious and error-prone
4. Finite differences require N+1 forward solves for N parameters

Enzyme gives you exact gradients through the entire nonlinear time-stepping loop for the cost of roughly one additional forward solve (for reverse mode). This enables:

- **Material property calibration**: fit k0, k1 to match experimental temperature measurements
- **Boundary condition estimation**: infer h_conv or T_hot from sensor data (inverse heat transfer)
- **Design optimization**: optimize geometry (Lx, Ly) or heat source placement (Q) to achieve a target temperature distribution
- **Sensitivity analysis**: understand how uncertainties in material properties propagate to the temperature field

## Usage

```python
from tesseract_core import Tesseract
import numpy as np

with Tesseract.from_image("enzyme-thermal-2d:latest") as t:
    nx, ny = 20, 20
    n = nx * ny

    # Uniform initial temperature
    T_init = np.full(n, 293.15)  # 20°C everywhere
    Q = np.zeros(n)               # no internal heating

    # Forward solve
    result = t.apply(inputs={
        "T_init": T_init, "Q": Q,
        "nx": nx, "ny": ny, "n_steps": 100,
        "k0": 45.0, "k1": -0.01,
        "rho": 7850.0, "cp": 460.0,
        "h_conv": 25.0, "T_inf": 293.15, "T_hot": 373.15,
        "Lx": 0.1, "Ly": 0.05, "dt": 0.01,
    })
    T_final = result["T_final"].reshape(ny, nx)

    # Gradient of average temperature w.r.t. material properties
    # (e.g., for calibrating k0 and h_conv from experimental data)
    cotangent = np.full(n, 1.0 / n)  # gradient of mean(T_final)
    vjp = t.vector_jacobian_product(
        inputs={
            "T_init": T_init, "Q": Q,
            "nx": nx, "ny": ny, "n_steps": 100,
            "k0": 45.0, "k1": -0.01,
            "rho": 7850.0, "cp": 460.0,
            "h_conv": 25.0, "T_inf": 293.15, "T_hot": 373.15,
            "Lx": 0.1, "Ly": 0.05, "dt": 0.01,
        },
        vjp_inputs=["k0", "k1", "h_conv", "T_hot"],
        vjp_outputs=["T_final"],
        cotangent_vector={"T_final": cotangent},
    )
    print(f"d(mean T)/d(k0)    = {vjp['k0']:.6f}")
    print(f"d(mean T)/d(k1)    = {vjp['k1']:.6f}")
    print(f"d(mean T)/d(h_conv)= {vjp['h_conv']:.6f}")
    print(f"d(mean T)/d(T_hot) = {vjp['T_hot']:.6f}")
```

## Toolchain

All tools are installed from prebuilt binaries during the Docker build (no source compilation of the toolchain itself):

| Tool                                        | Source          | Purpose                           |
| ------------------------------------------- | --------------- | --------------------------------- |
| [LFortran](https://lfortran.org/) 0.61.0    | conda-forge     | Fortran → LLVM IR frontend        |
| [LLVM](https://llvm.org/) 19                | apt.llvm.org    | IR optimization, linking, codegen |
| [Enzyme](https://enzyme.mit.edu/) (nightly) | GitHub releases | LLVM AD pass                      |

## Key design choices

- **Black-box solver.** The Fortran subroutine takes initial conditions and parameters, runs the full time integration internally, and returns the final temperature field. This mirrors how legacy Fortran solvers are structured — the user doesn't need to refactor their code to expose individual time steps.
- **Temperature-dependent conductivity via harmonic mean at cell faces.** This is the standard approach in finite volume / finite difference thermal codes — it ensures continuity of heat flux across cells with different conductivities.
- **No array intrinsics in the Fortran kernel.** Explicit `do` loops produce clean LLVM IR that Enzyme handles reliably.
- **`--no-array-bounds-checking`** is passed to LFortran. Bounds checks emit calls to LFortran's runtime which Enzyme cannot differentiate through.
- **Enzyme differentiates through the full time-stepping loop** using a store-all (tape) strategy for reverse mode. During the forward pass, Enzyme caches ~148 intermediate values per time step into dynamically allocated tapes (O(n_steps) memory). For the default problem size (~400 grid points, 100 time steps), this is ~115 KB — trivial. For very long simulations (e.g., 100,000 steps), tape memory grows linearly to ~115 MB. Enzyme's checkpointing annotations (`__enzyme_checkpoint`) could be used to trade recomputation for memory in such cases, but are not needed here.
- **Work arrays are passed from C, not allocated in Fortran.** LFortran emits `_lfortran_malloc` for variable-length arrays, which Enzyme cannot differentiate through. The C wrapper allocates `T_cur` and `T_new` on the heap and passes them to the Fortran subroutine, avoiding this issue.

## File structure

```
enzyme_thermal_2d/
├── README.md
├── tesseract_api.py              # Python API wrapping ctypes calls
├── tesseract_config.yaml         # Build config with LLVM/LFortran/Enzyme toolchain
├── tesseract_requirements.txt    # numpy
└── enzyme/
    ├── thermal_2d.f90            # Fortran solver (2D, nonlinear, multi-step)
    ├── wrapper.c                 # C wrapper declaring Enzyme AD entry points
    └── build.sh                  # Full compilation pipeline script
```
