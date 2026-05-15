# Enzyme AD: Differentiable Fortran Simulator

This example demonstrates how to obtain **exact automatic derivatives** of a Fortran simulation without writing any manual adjoint code, using [Enzyme](https://enzyme.mit.edu/) for automatic differentiation at the LLVM IR level.

## What it does

The Tesseract wraps a single explicit Euler step of the 1D heat equation:

```
dT/dt = alpha * d^2T/dx^2
```

discretized as:

```
T_out[i] = T_in[i] + r * (T_in[i-1] - 2*T_in[i] + T_in[i+1])
```

where `r = alpha * dt / dx^2`.

All three derivative endpoints (`apply`, `jacobian_vector_product`, `vector_jacobian_product`) are provided. The JVP and VJP are computed by Enzyme — not by finite differences or hand-written adjoints.

## How it works

The compilation pipeline runs at Docker build time:

```
heat_step.f90          (Fortran source — pure numerical kernel)
    │
    ▼  lfortran --show-llvm
heat_step.ll           (LLVM IR)
    │
    ▼  opt -O1
heat_step_opt.ll       (cleaned-up IR, ready for Enzyme)
    │
    ▼  llvm-link with wrapper.c
combined.ll            (Fortran IR + C wrapper with __enzyme_autodiff / __enzyme_fwddiff)
    │
    ▼  opt --load-pass-plugin=LLVMEnzyme-19.so -passes=enzyme
ad.ll                  (LLVM IR with compiler-generated forward and reverse mode derivatives)
    │
    ▼  clang -shared
libheat_ad.so          (shared library: forward, JVP, VJP entry points)
```

At runtime, `tesseract_api.py` loads `libheat_ad.so` via `ctypes` and exposes:

- **`apply`** — calls the forward Fortran kernel
- **`jacobian_vector_product`** — calls the Enzyme forward-mode (JVP) wrapper
- **`vector_jacobian_product`** — calls the Enzyme reverse-mode (VJP) wrapper

## Toolchain

All tools are installed from prebuilt binaries during the Docker build (no source compilation of the toolchain itself):

| Tool                                        | Source          | Purpose                           |
| ------------------------------------------- | --------------- | --------------------------------- |
| [LFortran](https://lfortran.org/) 0.61.0    | conda-forge     | Fortran → LLVM IR frontend        |
| [LLVM](https://llvm.org/) 19                | apt.llvm.org    | IR optimization, linking, codegen |
| [Enzyme](https://enzyme.mit.edu/) (nightly) | GitHub releases | LLVM AD pass                      |

## Key design choices

- **No array intrinsics in the Fortran kernel.** The heat step uses explicit `do` loops rather than whole-array operations. This produces clean LLVM IR (simple GEP + load/store patterns) that Enzyme handles reliably.
- **`--no-array-bounds-checking`** is passed to LFortran. Bounds checks emit calls to LFortran's runtime (`_lcompilers_runtime_error`, `exit`) which Enzyme cannot differentiate through.
- **The C wrapper (`wrapper.c`) bridges the Fortran ABI and Enzyme.** It copies scalar arguments to the stack (Fortran passes everything by pointer), declares `__enzyme_autodiff` / `__enzyme_fwddiff` calls with the appropriate `enzyme_dup` / `enzyme_const` annotations, and exports clean C-callable functions for Python ctypes.

## Usage

```python
from tesseract_core import Tesseract
import numpy as np

with Tesseract.from_image("enzyme-ad:latest") as t:
    T_in = np.array([0.0, 0.0, 100.0, 0.0, 0.0])

    # Forward pass
    result = t.apply(inputs={"T_in": T_in, "alpha": 0.01, "dx": 0.25, "dt": 0.001})
    print(result["T_out"])

    # Reverse-mode gradient of T_out[2] w.r.t. all inputs
    vjp = t.vector_jacobian_product(
        inputs={"T_in": T_in, "alpha": 0.01, "dx": 0.25, "dt": 0.001},
        vjp_inputs=["T_in", "alpha", "dx", "dt"],
        vjp_outputs=["T_out"],
        cotangent_vector={"T_out": np.array([0.0, 0.0, 1.0, 0.0, 0.0])},
    )
    print(vjp["T_in"])    # [0.0, 0.00016, 0.99968, 0.00016, 0.0]
    print(vjp["alpha"])   # -3.2
```

## File structure

```
enzyme_ad/
├── README.md
├── tesseract_api.py              # Python API wrapping ctypes calls
├── tesseract_config.yaml         # Build config with LLVM/LFortran/Enzyme toolchain
├── tesseract_requirements.txt    # numpy
└── enzyme/
    ├── heat_step.f90             # Fortran kernel (no I/O, no allocations)
    ├── wrapper.c                 # C wrapper declaring Enzyme AD entry points
    └── build.sh                  # Full compilation pipeline script
```
