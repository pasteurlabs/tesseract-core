# Differentiating Compiled Code (Enzyme + Fortran)

[View on GitHub](https://github.com/pasteurlabs/tesseract-core/tree/main/examples/fortran_enzyme)

## Context

The {doc}`Fortran Integration <fortran>` example wraps a compiled solver as a Tesseract, but stops at the forward pass. This example adds **exact, machine-precision derivatives** with no hand-written adjoint code: the solver is compiled to LLVM IR with [LFortran](https://lfortran.org/), [Enzyme](https://enzyme.mit.edu/) generates forward- and reverse-mode derivatives directly from that IR, and the result is a shared library exposing all three differentiable endpoints. The pattern applies to any code that lowers to LLVM IR (C, C++, Rust).

## Example Tesseract (examples/fortran_enzyme)

### The Fortran kernel

The solver computes a single explicit Euler step of the 1D heat equation:

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$$

discretized with central differences as $T_\text{out}(i) = T_\text{in}(i) + r \, (T_\text{in}(i-1) - 2 T_\text{in}(i) + T_\text{in}(i+1))$, where $r = \alpha \, \Delta t / \Delta x^2$:

```{literalinclude} ../../../../examples/fortran_enzyme/enzyme/heat_step.f90
:language: fortran
:lines: 21-39
```

Two details keep the generated IR clean enough for Enzyme to differentiate reliably: the stencil uses explicit `do` loops (not whole-array intrinsics), and LFortran is run with `--no-array-bounds-checking`, since bounds checks emit runtime calls Enzyme cannot trace through.

### Input and output schemas

Every differentiable field is marked with `Differentiable[...]`. The `InputSchema` also enforces positivity and the CFL stability condition $r \leq 0.5$, skipping the check during `abstract_eval` when only shapes are known:

```{literalinclude} ../../../../examples/fortran_enzyme/tesseract_api.py
:pyobject: InputSchema
:language: python
```

```{literalinclude} ../../../../examples/fortran_enzyme/tesseract_api.py
:pyobject: OutputSchema
:language: python
```

### The compilation pipeline

The Enzyme magic happens at image build time. A C wrapper (`wrapper.c`) bridges the Fortran ABI and Enzyme: it copies scalar arguments to the stack (Fortran passes everything by pointer) and declares `__enzyme_autodiff` / `__enzyme_fwddiff` calls annotated with `enzyme_dup` (differentiate) or `enzyme_const` (hold fixed). These sentinel calls are replaced by generated derivative code when the Enzyme pass runs:

```{literalinclude} ../../../../examples/fortran_enzyme/enzyme/wrapper.c
:language: c
:lines: 44-63
```

The `build.sh` script chains the toolchain together, emitting `libheat_ad.so` with three entry points---`heat_step_forward` (primal), `heat_step_jvp` (forward-mode), and `heat_step_vjp` (reverse-mode):

```{literalinclude} ../../../../examples/fortran_enzyme/enzyme/build.sh
:language: bash
:lines: 21-41
```

### Wiring the library into the Tesseract API

`tesseract_api.py` loads the shared library via `ctypes` and declares the signatures of the three entry points. The `apply` function calls the primal kernel:

```{literalinclude} ../../../../examples/fortran_enzyme/tesseract_api.py
:pyobject: apply
:language: python
```

The differentiable endpoints call the Enzyme-generated wrappers. Reverse mode threads cotangents into shadow arrays that Enzyme accumulates gradients into; the `vector_jacobian_product` endpoint then returns only the requested inputs. The `jacobian_vector_product` endpoint is the forward-mode mirror image, calling `heat_step_jvp` with input tangents instead.

```{literalinclude} ../../../../examples/fortran_enzyme/tesseract_api.py
:pyobject: vector_jacobian_product
:language: python
```

### Build configuration

The `tesseract_config.yaml` installs the LLVM 19 toolchain, LFortran (via micromamba from conda-forge), and the prebuilt Enzyme plugin as `custom_build_steps`, then runs `build.sh` to produce the differentiated library. All tools are installed from prebuilt binaries---the toolchain itself is never compiled from source:

```{literalinclude} ../../../../examples/fortran_enzyme/tesseract_config.yaml
:language: yaml
```

## Adapting this pattern

To differentiate your own compiled code with Enzyme:

1. **Write a C wrapper** declaring `__enzyme_autodiff` / `__enzyme_fwddiff` calls, annotating each argument `enzyme_dup` or `enzyme_const`, against a clean kernel (explicit loops, no runtime checks Enzyme can't trace through).
2. **Assemble the pipeline** as `custom_build_steps`: lower to LLVM IR, link the wrapper, run the Enzyme pass, and compile to a shared library.
3. **Wire it up**: load the library with `ctypes`, mark differentiable fields with `Differentiable[...]`, and call the JVP/VJP wrappers from the gradient endpoints.

## See also

- {ref}`tr-autodiff` for background on differentiable programming in Tesseracts
- The {doc}`Fortran Integration <fortran>` building block for wrapping a compiled solver without gradients
- The {doc}`Finite Difference Gradients <finitediff>` building block for an inexact, framework-free alternative
