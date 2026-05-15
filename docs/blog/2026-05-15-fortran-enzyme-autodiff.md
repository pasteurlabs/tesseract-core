---
orphan: true
og:title: "From Fortran to JAX autodiff via LLVM and Enzyme"
og:description: "We duct-taped LFortran, LLVM, and Enzyme together to get exact gradients from a Fortran thermal solver, then solved inverse problems with jax.grad. A field report."
blog_date: "2026-05-15"
blog_author: "@dionhaefner"
blog_title: "From Fortran to JAX autodiff via LLVM and Enzyme"
blog_description: "We duct-taped LFortran, LLVM, and Enzyme together to get exact gradients from a Fortran thermal solver, then solved inverse problems with jax.grad. A field report."
---

# From Fortran to JAX autodiff via LLVM and Enzyme

We duct-taped four compilers together and got exact gradients out of a Fortran thermal solver. Then we used those gradients to reconstruct a 900-element temperature field from 100 noisy sensor readings. No adjoint code was written by hand. The whole thing is held together with shell scripts and ctypes. If you maintain a Fortran, C, or C++ simulation and have wished you could just call `grad` on it, this is a rough field report on one way to get there.

The mechanism: [Enzyme](https://enzyme.mit.edu/) works at the LLVM IR level, so it can differentiate anything that compiles to LLVM. We used it on a 2D Fortran heat solver with nonlinear material properties and mixed boundary conditions, compiled through [LFortran](https://lfortran.org/). This post walks through the compilation pipeline (warts and all), verifies the gradients, and then uses them to solve two inverse problems of increasing ambition.

## The problem

Scientific computing is full of Fortran, C, and C++ code that people need gradients of. Optimization, inverse problems, ML integration, uncertainty quantification: all of these require derivatives of a simulation's outputs with respect to its inputs. The options today are all bad in their own way:

- **Hand-written adjoints.** Months of expert effort, error-prone, and a maintenance nightmare that drifts out of sync with the forward code.
- **Finite differences.** Slow (O(n) evaluations for n parameters), inaccurate (the truncation-vs-roundoff tradeoff means there's always a sweet spot you have to find), and poorly conditioned for stiff problems.
- **Rewrite in JAX or PyTorch.** Impractical for existing codebases with tens of thousands of lines of validated physics.

What if you could just compile the derivatives automatically, from the existing source? That's the promise, anyway. Here's what it actually looks like in practice.

## The Fortran code

The solver is `thermal_2d.f90`, about 220 lines of vanilla Fortran 90. It solves 2D transient heat conduction with temperature-dependent conductivity:

$$\rho \, c_p \frac{\partial T}{\partial t} = \nabla \cdot \big( k(T) \, \nabla T \big) + Q$$

where the conductivity follows a linear material model: $k(T) = k_0 + k_1 \cdot T$. Time integration is explicit Euler over `n_steps` steps.

Here's the subroutine signature and the interior stencil loop:

```fortran
subroutine thermal_2d_solve(n, nx, ny, n_steps, &
                            T_init, T_final, T_cur, T_new, &
                            k0, k1, rho, cp, &
                            h_conv, T_inf, T_hot, &
                            Q, Lx, Ly, dt)
  implicit none
  integer, intent(in) :: n, nx, ny, n_steps
  double precision, intent(in) :: T_init(n)
  double precision, intent(out) :: T_final(n)
  ! ... (work arrays, scalars)

  ! Time integration loop
  do step = 1, n_steps
    do j = 2, ny - 1
      do i = 2, nx - 1
        idx = (j - 1) * nx + i

        T_c = T_cur(idx)
        T_e = T_cur(idx + 1)
        T_w = T_cur(idx - 1)
        T_nn = T_cur(idx + nx)
        T_s = T_cur(idx - nx)

        ! Harmonic-mean conductivity at cell faces
        kx_east = 2.0d0 * (k0 + k1*T_c) * (k0 + k1*T_e) &
                  / ((k0 + k1*T_c) + (k0 + k1*T_e))
        kx_west = 2.0d0 * (k0 + k1*T_c) * (k0 + k1*T_w) &
                  / ((k0 + k1*T_c) + (k0 + k1*T_w))
        ! ... (ky_north, ky_south similarly)

        flux_x = (kx_east*(T_e - T_c) - kx_west*(T_c - T_w)) / (dx*dx)
        flux_y = (ky_north*(T_nn - T_c) - ky_south*(T_c - T_s)) / (dy*dy)

        T_new(idx) = T_c + dt/(rho*cp) * (flux_x + flux_y + Q(idx))
      end do
    end do
    ! ... (boundary conditions, swap T_cur <- T_new)
  end do
```

The stencil uses harmonic-mean conductivity at cell faces, the standard approach in finite difference thermal codes for ensuring flux continuity across cells with different conductivities. Boundary conditions are mixed: Dirichlet (hot wall at the bottom), convection/Robin (top), and insulated/Neumann (sides).

The key point for why AD matters here: with $k(T)$ nonlinear, the stencil coefficients depend on the current temperature field. The Jacobian changes at every time step. Hand-coding the adjoint through this nonlinear stencil and multi-step loop is tedious and error-prone.

This is vanilla Fortran. No special annotations, no AD-aware constructs. Explicit `do` loops, no array intrinsics. But "vanilla" required some effort. The original version used local allocatable arrays for `T_cur` and `T_new`. LFortran compiles those into `_lfortran_malloc` calls, which Enzyme can't differentiate through — the AD pass just crashes. The workaround was to pass work arrays in from C, pre-allocated on the heap. Not a huge change, but the kind of thing you discover only after staring at an unhelpful LLVM error message. We also avoid array intrinsics and bounds checking (`--no-array-bounds-checking`) for the same reason: they emit runtime calls that Enzyme doesn't know how to handle.

This is an honest preview of what "differentiate existing Fortran" looks like today. The source stays recognizably Fortran, but you'll likely need to massage it — eliminating allocations and runtime calls that the AD toolchain can't see through. For a 220-line solver, that's an afternoon of work. For a large legacy codebase, it's an open question.

## The pipeline

Enzyme works at the LLVM IR level, not the source level. Anything that compiles to LLVM IR — C, C++, Rust, Fortran — can, in theory, be differentiated. In practice there are caveats (we'll get to those). But the idea is sound, and the compilation chain is surprisingly straightforward if you're comfortable staring at LLVM IR when things go wrong.

Six steps:

<img src="../_static/blog/enzyme-pipeline.png" alt="Compilation pipeline: Fortran → Enzyme AD → shared library">

**1. Fortran → LLVM IR** via LFortran:

```bash
lfortran --show-llvm --no-array-bounds-checking thermal_2d.f90 > thermal_2d.ll
```

LFortran is a modern Fortran compiler that emits clean LLVM IR. It's also still maturing — not all Fortran features are supported yet ([compilation status](https://lfortran.org/progress/)). We chose it over gfortran because its LLVM IR output is much cleaner, but that means your Fortran needs to stay within what LFortran can handle. Enzyme also works with GCC/Clang frontends for broader language coverage at the cost of messier IR.

**2. Optimize the IR:**

```bash
opt -O1 -S thermal_2d.ll -o thermal_2d_opt.ll
```

We use `-O1` here rather than `-O3` deliberately, and it took us a day to figure out why. With `-O3`, the VJP returned NaN on certain inputs while the forward pass worked fine. The root cause was an interaction between LLVM's vectorization/code-motion passes and Enzyme's reverse-mode analysis: aggressive transforms produced IR patterns that Enzyme mishandled when adjacent cell temperatures were equal and intermediate terms canceled. The fix was to keep pre-Enzyme optimization mild and save `-O3` for after the AD pass (step 6). If you're building a similar pipeline, be aware that "the forward pass works" does not imply "the gradients are correct."

**3. Compile the C wrapper to LLVM IR:**

```bash
clang -emit-llvm -S -O1 wrapper.c -o wrapper.ll
```

The C wrapper bridges Fortran's by-pointer ABI to a C-callable interface with Enzyme annotations. It declares three entry points — `thermal_2d_forward`, `thermal_2d_vjp`, and `thermal_2d_jvp` — using Enzyme's `__enzyme_autodiff` and `__enzyme_fwddiff` intrinsics to mark which arguments get shadow (gradient) buffers. Here's the core of the VJP entry point:

```c
void thermal_2d_vjp(int nx, int ny, int n_steps,
                    const double* T_init,  double* dT_init,
                    const double* T_final, double* dT_final,
                    double k0,     double* dk0,
                    double k1,     double* dk1,
                    /* ... */)
{
    double* T_cur  = calloc(n, sizeof(double));
    double* dT_cur = calloc(n, sizeof(double));
    /* ... */

    __enzyme_autodiff((void*)thermal_2d_solve,
        enzyme_const, &n_,
        enzyme_const, &nx_,
        enzyme_dup, (double*)T_init, dT_init,
        enzyme_dup, (double*)T_final, dT_final,
        enzyme_dup, &k0_, dk0,
        enzyme_dup, &k1_, dk1,
        /* ... */);
}
```

The wrapper also allocates work arrays on the heap rather than letting Fortran allocate them, avoiding the `_lfortran_malloc` issue.

**4. Link the IR modules:**

```bash
llvm-link wrapper.ll thermal_2d_opt.ll -S -o combined.ll
```

**5. Run the Enzyme AD pass:**

```bash
opt --load-pass-plugin=LLVMEnzyme-19.so -passes=enzyme -S combined.ll -o ad.ll
```

This is the step that does the actual work. Enzyme analyzes the LLVM IR and synthesizes forward- and reverse-mode derivative code. For reverse mode, it uses a store-all (tape) strategy, caching intermediate values at each time step. When it works, it's genuinely impressive. When it doesn't, you're reading LLVM IR diffs at 2 AM (see [Limitations](#limitations-and-whats-next)).

**6. Optimize and compile to a shared library:**

```bash
opt -O3 -S ad.ll -o ad_opt.ll
clang -shared -O3 ad_opt.ll -o libthermal_2d_ad.so -lm
```

The result is a single `.so` file with three entry points callable from Python via ctypes: forward evaluation, JVP, and VJP. The entire pipeline runs during `tesseract build` and takes about 30 seconds. It's not elegant, but there's something satisfying about a shell script that turns Fortran into exact gradients.

## Does it actually work?

The pipeline compiles. But does it produce correct gradients? We called the VJP from Python and compared against central finite differences at various step sizes:

<img src="../_static/blog/enzyme-fd-convergence.png" alt="Enzyme vs. finite difference gradient accuracy">

Finite differences have a sweet spot: too large an $\epsilon$ and you get truncation error; too small and you get roundoff error. The best relative error is typically around $10^{-8}$. Enzyme's gradients agree to machine precision ($\sim 10^{-15}$), because they're computing the analytically correct derivative, just synthesized by a compiler rather than a human.

To be clear about what's being differentiated here: the entire multi-step time loop with nonlinear stencil updates at each step. Not a single-step toy.

| Method                    | Relative error vs. exact |
| ------------------------- | -----------------------: |
| Enzyme AD                 |           ~1e-15 (exact) |
| FD (best $\epsilon$)      |                    ~1e-8 |
| FD ($\epsilon$ too large) |                    ~1e-2 |
| FD ($\epsilon$ too small) |                    ~1e-4 |

## Now do something useful with it

Exact gradients from a compiler pass are a neat trick. But the question that matters is whether you can actually _use_ them — plug them into an optimizer, solve a real inverse problem, compose them with other code.

To wire the Enzyme gradients into JAX, the solver needs to look like a differentiable JAX primitive. We used [Tesseract](https://github.com/pasteurlabs/tesseract-core) for this: it wraps the compiled library (LFortran, LLVM 19, Enzyme, the whole toolchain) into a container with autodiff endpoints, so `jax.value_and_grad` routes VJP calls to the Enzyme-generated code. The setup is two commands:

```bash
tesseract build demo/enzyme_thermal_2d/
tesseract serve enzyme-thermal-2d
```

With that, we can throw optimization problems at it and see what breaks.

### Scalar calibration: recovering 2 material parameters

A steel plate is heating up. We have thermocouple readings at 9 sensor locations, but we don't know the exact material properties. Can we recover $k_0$ (base conductivity) and $k_1$ (temperature coefficient) from sparse, noisy observations?

We generate synthetic "observed" data by running the solver with known true values ($k_0 = 45$, $k_1 = -0.02$), sample at sensor locations, and add 0.5 K of Gaussian noise. Then we start from a deliberately wrong initial guess — 33% off on $k_0$, wrong sign on $k_1$ — and run L-BFGS-B.

One VJP call gives gradients with respect to both parameters simultaneously. The optimizer doesn't need to know that the gradients come from a compiled Fortran solver differentiated by an LLVM pass.

```python
def objective_and_gradient(params, tesseract):
    k0, k1 = params
    result = tesseract.apply(inputs=make_inputs(k0, k1))
    T_pred = np.array(result["T_final"])

    residuals = T_pred[sensor_indices] - T_obs
    loss = 0.5 * np.sum(residuals**2)

    # Cotangent: residuals at sensor locations, zero elsewhere
    cotangent = np.zeros(n, dtype=np.float64)
    cotangent[sensor_indices] = residuals

    # One VJP call → gradients w.r.t. both k0 and k1
    vjp = tesseract.vector_jacobian_product(
        inputs=make_inputs(k0, k1),
        vjp_inputs=["k0", "k1"],
        vjp_outputs=["T_final"],
        cotangent_vector={"T_final": cotangent},
    )
    return loss, np.array([vjp["k0"], vjp["k1"]])
```

<img src="../_static/blog/enzyme-part1-convergence.png" alt="Scalar calibration convergence: loss, k0, k1">

<img src="../_static/blog/enzyme-part1-temperature-fields.png" alt="Temperature field comparison: initial guess, recovered, ground truth, error">

L-BFGS-B converges in about 15 iterations. The recovered parameters match the true values to within the noise floor. This is a gentle problem — only 2 unknowns, well-conditioned — so it's more of a sanity check than a stress test. The real question is whether the gradients hold up when we push harder.

### Thermal forensics: recovering a 900-element initial temperature field

A steel plate was subjected to an unmonitored heating event — say, a laser pulse or a localized defect generating heat. Five seconds later, you measure temperatures at 100 sensor locations. Can you reconstruct what the initial temperature distribution looked like?

The true initial condition has two Gaussian hot spots on a warm background. This is 900 unknowns from 100 noisy observations, through a nonlinear PDE. An ill-posed problem by any measure.

This is where reverse-mode AD becomes essential:

| Method                | Forward solves per iteration |
| --------------------- | ---------------------------: |
| Finite differences    |                  901 (n + 1) |
| VJP (reverse-mode AD) |        2 (forward + reverse) |
| **Speedup**           |                    **~450×** |

Finite differences would need 901 forward solves per iteration to get all 900 gradients. One VJP gives them all at once, for roughly the cost of two forward passes. The trade-off is memory: Enzyme's reverse mode tapes intermediate values at each time step. For this problem (~900 grid points, 100 steps), the tape is a few hundred kilobytes. For production codes with large state vectors and thousands of time steps, tape memory becomes the dominant constraint. Enzyme supports checkpointing annotations to trade recomputation for memory, but we haven't needed or tested them here.

<img src="../_static/blog/enzyme-part2-forensics.png" alt="Thermal forensics: recovering 900 initial temperature values from 100 sensors">

L-BFGS-B recovers the hot spot locations and magnitudes. The correlation between recovered and true initial temperature fields exceeds 0.99. The error is largest at the edges of the hot spots, where the signal has diffused most. It's not perfect — the reconstruction is smoothed relative to the true field, as you'd expect from an ill-posed problem with diffusion — but it's far better than we had any right to expect from a pipeline held together with shell scripts.

That's `jax.grad` flowing through compiled Fortran, with no adjoint code written by hand.

### JAX integration via tesseract-jax

With [tesseract-jax](https://github.com/pasteurlabs/tesseract-jax), the Tesseract becomes a JAX primitive. `jax.value_and_grad` just works:

```python
from tesseract_jax import apply_tesseract

def loss_fn(k0, k1, tesseract):
    inputs = {**base_inputs, "k0": k0, "k1": k1}
    T_pred = apply_tesseract(tesseract, inputs)["T_final"]
    residuals = T_pred[sensor_indices] - jnp.array(T_obs)
    return 0.5 * jnp.sum(residuals**2)

# This differentiates through: Python → JAX → HTTP → Enzyme → Fortran
loss, (dk0, dk1) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
    jnp.float64(60.0), jnp.float64(0.01), enzyme_tess
)
```

From JAX's perspective, the Fortran solver is just another differentiable function. You can swap `enzyme_tess` for a pure-JAX reimplementation and the optimization loop doesn't change — only the container behind the HTTP call does. Whether that abstraction is beautiful or horrifying depends on your tolerance for gradients that traverse six layers of indirection (Python → JAX → HTTP → ctypes → Enzyme → Fortran).

## Where this could go

We should be careful about extrapolating from a 220-line solver we wrote to fit the pipeline. Enzyme handles loops, conditionals, and function calls, and the Enzyme team has demonstrated it on larger codes including [BUDE molecular docking](https://enzyme.mit.edu/getting_started/UsingEnzyme/#bude) and [LULESH hydrodynamics](https://enzyme.mit.edu/getting_started/UsingEnzyme/#lulesh), with derivative overhead factors typically between 1× and 4×. But "works on LULESH" and "works on your 30-year-old Fortran codebase" are different claims.

That said, the composability angle is interesting. Imagine a Fortran CFD solver (Enzyme-differentiated) feeding into a JAX neural net surrogate:

```python
cfd_output = apply_tesseract(cfd_tess, cfd_inputs)     # Fortran + Enzyme VJP
surrogate_loss = neural_net(cfd_output["pressure"])      # JAX AD
total_loss = surrogate_loss + regularizer(cfd_inputs)    # JAX AD

grads = jax.grad(total_loss)  # chains Enzyme + JAX AD automatically
```

Each component uses its native AD. Tesseract handles the composition. We built a version of this pattern for the [rocket fin optimization](2025-11-28-rocket-fin-optimization.md) post, where analytical adjoints, finite differences, and JAX AD coexisted in one pipeline. It worked, but each new gradient source added its own class of debugging problems.

Because Enzyme works at the LLVM IR level, nothing here is Fortran-specific. The same pipeline should apply to C, C++, Rust, or any language with an LLVM frontend — though "should" is doing some work in that sentence.

## What's next (and what we'd do differently)

Most of the sharp edges — the `_lfortran_malloc` workaround, the `-O3` NaN disaster, LFortran's incomplete coverage — are already described in context above. To summarize: this works, but it's not turnkey. You should expect to adapt your Fortran, pin your compiler versions, and debug at the IR level at least once.

A few things we haven't tried yet:

**Implicit time integration.** Backward Euler, Crank-Nicolson, and other implicit schemes require differentiating through iterative linear solves (CG, GMRES). Enzyme can handle this in principle, but tape memory grows with solver iteration count, and we haven't tested this path with LFortran. This is where we're headed next.

**Third-party code.** We wrote this solver to work with the pipeline. We haven't yet run Enzyme + LFortran on a Fortran codebase we didn't author. The "will it work on _my_ code?" question is fair. The honest answer is "probably, with adaptation" — the kind of adaptation we described above, and possibly more that we haven't encountered yet.

**Solvers at scale.**. MPI-parallel codes, GPU kernels, and multi-physics coupling are also untested with this pipeline. Enzyme has support for some of these; we just haven't tried.

## Try it yourself

The full source — Fortran solver, Enzyme pipeline, inverse problem notebooks, and the shell scripts holding it all together — is [on GitHub](https://github.com/pasteurlabs/tesseract-core/tree/main/demo/enzyme_thermal_2d). If you have a Fortran, C, or C++ solver you want gradients for and a healthy appetite for compiler debugging, this is a starting point.
