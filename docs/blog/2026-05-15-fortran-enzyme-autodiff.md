---
orphan: true
og:title: "Fortran solvers, JAX autodiff, and the magic of Enzyme"
og:description: "We duct-taped LFortran, LLVM, and Enzyme together to get exact gradients from a Fortran thermal solver, then solved inverse problems with jax.grad. Here's how."
blog_date: "2026-05-15"
blog_author: "@dionhaefner"
blog_title: "Fortran solvers, JAX autodiff, and the magic of Enzyme"
blog_description: "We duct-taped LFortran, LLVM, and Enzyme together to get exact gradients from a Fortran thermal solver, then solved inverse problems with jax.grad. Here's how."
---

# Fortran solvers, JAX autodiff, and the magic of Enzyme

What if you could do autodiff through existing Fortran, C, or C++ simulation code, embed it into JAX and torch, and use it as a high-performance differentiable physics engine? Turns out, you can -- if you're brave enough. The key insight is that [Enzyme](https://enzyme.mit.edu/) allows us to apply AD at the LLVM IR level, which means we can differentiate any code that compiles to LLVM, and that Tesseract enables us to wrap anything as a [JAX](https://jax.readthedocs.io/en/latest/) primitive, granting full access to JAX's autodiff capabilities from Python.

All you need is to duct-tape [LFortran](https://lfortran.org/), LLVM, and Enzyme together, point the result at a Fortran thermal solver, and get exact gradients out the other end. Package it as a Tesseract, use Tesseract-JAX to register it as a custom JAX primitive, and you're off to the races: Fortran solvers acting as differentiable layers in arbitrary JAX code. From Fortran to LLVM IR to Enzyme AD to C wrapper to shared library to Tesseract to JAX primitive to XLA to Python to your optimization loop. Sounds easy, right? Well, it is and isn't, but at any rate it's pretty amazing that a stack combining some of the oldest and newest technologies out there can work together so seamlessly.

But let's start a the beginning. What follows is the full walkthrough — the compilation pipeline, the sharp edges we hit, and two inverse problems that made the whole effort worth it (and that wouldn't be possible without AD).

## The problem

If you work in scientific computing, you might have run into this: you have a simulation written in Fortran (or C, or C++), and now someone needs gradients. Maybe it's for optimization, maybe inverse problems, maybe plugging the sim into an ML pipeline. Derivatives of the simulation's outputs with respect to its inputs. And your options are:

- **Hand-written adjoints.** Months of expert effort. Error-prone. A maintenance nightmare that slowly drifts out of sync with the forward code.
- **Finite differences.** Slow (O(n) evaluations for n parameters), inaccurate (you're always hunting for the truncation-vs-roundoff sweet spot), and poorly conditioned for stiff problems.
- **Rewrite in JAX or PyTorch.** Sure, if you want to rewrite tens of thousands of lines of validated physics.

What if you could just compile the derivatives automatically, from the existing source? That's the pitch. Here's what it actually looks like when you try.

## The Fortran code

Our test subject is `thermal_2d.f90`, about 220 lines of vanilla Fortran 90. It solves 2D transient heat conduction with temperature-dependent conductivity:

$$\rho \, c_p \frac{\partial T}{\partial t} = \nabla \cdot \big( k(T) \, \nabla T \big) + Q$$

The conductivity follows a linear material model, $k(T) = k_0 + k_1 \cdot T$, and time integration is explicit Euler over `n_steps` steps. Nothing exotic.

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

The stencil uses harmonic-mean conductivity at cell faces (the standard approach for flux continuity across cells with different conductivities). Boundary conditions are mixed: Dirichlet (hot wall at the bottom), convection/Robin (top), and insulated/Neumann (sides).

The reason AD matters here: with $k(T)$ nonlinear, the stencil coefficients depend on the current temperature field. The Jacobian changes at every time step. Hand-coding the adjoint through this nonlinear stencil and multi-step time loop is the kind of thing that sounds doable until you actually sit down to do it.

Now, we keep calling this "vanilla Fortran," and it is — no special annotations, no AD-aware constructs, explicit `do` loops, no array intrinsics. But getting to "vanilla" took some work. Our first version used local allocatable arrays for `T_cur` and `T_new`. LFortran compiles those into `_lfortran_malloc` calls, and Enzyme has no idea what to do with them. The AD pass just crashes. The fix was to pass work arrays in from C, pre-allocated on the heap. Not a huge change, but we only figured that out after staring at an unhelpful LLVM error for longer than we'd like to admit. We also had to disable array intrinsics and bounds checking (`--no-array-bounds-checking`) for the same reason: they emit runtime calls that Enzyme can't see through.

This is an honest preview of what "differentiate existing Fortran" looks like today. The source stays recognizably Fortran, but you'll need to massage it, eliminating allocations and runtime calls that break the AD toolchain. For a 220-line solver, that's an afternoon. For a large legacy codebase, it's an open question.

## The pipeline

Here's the key insight that makes all of this possible: Enzyme works at the LLVM IR level, not the source level. Anything that compiles to LLVM IR — C, C++, Rust, Fortran — can be differentiated. In practice there are caveats (oh, we'll get to those). But the compilation chain is surprisingly straightforward, as long as you're comfortable staring at LLVM IR when things go wrong.

Six steps:

<img src="../_static/blog/enzyme-pipeline.png" alt="Compilation pipeline: Fortran → Enzyme AD → shared library">

**1. Fortran → LLVM IR** via LFortran:

```bash
lfortran --show-llvm --no-array-bounds-checking thermal_2d.f90 > thermal_2d.ll
```

LFortran is a modern Fortran compiler that emits clean LLVM IR — arrays become plain pointers with GEP/load/store patterns (much like C), rather than the multi-field descriptor structs and runtime library calls you get from Flang. That matters because Enzyme needs to trace through every memory access for its activity analysis, and simple pointer arithmetic is much easier to analyze than opaque `fir.box` descriptors. LFortran is still maturing, though — not all Fortran features are supported yet ([compilation status](https://lfortran.org/progress/)), so your Fortran needs to stay within what LFortran can handle.

**2. Optimize the IR:**

```bash
opt -O1 -S thermal_2d.ll -o thermal_2d_opt.ll
```

Notice that's `-O1`, not `-O3`. We learned this the hard way.

Our first pipeline used `-O3` here, and the forward pass worked perfectly. The VJP, however, returned NaN on certain inputs. We spent a full day tracking this down. The root cause: LLVM's aggressive vectorization and code-motion passes at `-O3` produced IR patterns that Enzyme mishandled during reverse-mode analysis, specifically when adjacent cell temperatures were equal and intermediate terms canceled. The fix was embarrassingly simple — keep pre-Enzyme optimization mild and save `-O3` for after the AD pass (step 6). But the lesson was not simple at all: "the forward pass works" does not mean "the gradients are correct." If you're building a similar pipeline, test the gradients early and often.

**3. Compile the C wrapper to LLVM IR:**

```bash
clang -emit-llvm -S -O1 wrapper.c -o wrapper.ll
```

We need a thin C wrapper to bridge Fortran's by-pointer ABI to a C-callable interface with Enzyme annotations. It declares three entry points — `thermal_2d_forward`, `thermal_2d_vjp`, and `thermal_2d_jvp` — using Enzyme's `__enzyme_autodiff` and `__enzyme_fwddiff` intrinsics to mark which arguments get shadow (gradient) buffers. Here's the core of the VJP entry point:

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

This is also where we allocate the work arrays on the heap, sidestepping the `_lfortran_malloc` issue from earlier.

**4. Link the IR modules:**

```bash
llvm-link wrapper.ll thermal_2d_opt.ll -S -o combined.ll
```

**5. Run the Enzyme AD pass:**

```bash
opt --load-pass-plugin=LLVMEnzyme-19.so -passes=enzyme -S combined.ll -o ad.ll
```

This is where the magic happens. Enzyme analyzes the LLVM IR and synthesizes forward- and reverse-mode derivative code. For reverse mode, it uses a store-all (tape) strategy, caching intermediate values at each time step. When it works, it's genuinely impressive — you get an adjoint of your entire time-stepping loop for free. When it doesn't, you're reading LLVM IR diffs at 2 AM wondering where your life went wrong (see [What's next](#whats-next-and-what-wed-do-differently)).

**6. Optimize and compile to a shared library:**

```bash
opt -O3 -S ad.ll -o ad_opt.ll
clang -shared -O3 ad_opt.ll -o libthermal_2d_ad.so -lm
```

Out comes a single `.so` file with three entry points callable from Python via ctypes: forward evaluation, JVP, and VJP. The entire pipeline runs during `tesseract build` and takes about 30 seconds. It's a shell script that turns Fortran into exact gradients. We're not going to pretend that's elegant, but it is deeply satisfying.

## Does it actually work?

OK, it compiles. But after the `-O3` NaN incident, we weren't about to trust it without checking. We called the VJP from Python and compared against central finite differences at various step sizes:

<img src="../_static/blog/enzyme-fd-convergence.png" alt="Enzyme vs. finite difference gradient accuracy">

Finite differences have a sweet spot: too large an $\epsilon$ and you get truncation error, too small and you get roundoff. The best you can typically do is around $10^{-8}$ relative error. Enzyme's gradients agree to machine precision ($\sim 10^{-15}$) — because they're computing the analytically correct derivative, just synthesized by a compiler instead of a human.

And this isn't differentiating a single matrix multiply. It's the entire multi-step time loop with nonlinear stencil updates at each step.

| Method                    | Relative error vs. exact |
| ------------------------- | -----------------------: |
| Enzyme AD                 |           ~1e-15 (exact) |
| FD (best $\epsilon$)      |                    ~1e-8 |
| FD ($\epsilon$ too large) |                    ~1e-2 |
| FD ($\epsilon$ too small) |                    ~1e-4 |

## Now do something useful with it

Correct gradients are great, but they're not the point. The point is using them — plugging them into an optimizer, solving a real inverse problem, composing them with other differentiable code.

To wire the Enzyme gradients into JAX, the solver needs to look like a differentiable JAX primitive. This is essentially what Tesseract was made for: it wraps the compiled library (LFortran, LLVM 19, Enzyme, the whole toolchain) into a container with autodiff endpoints, so `jax.value_and_grad` routes VJP calls to the Enzyme-generated code. Two commands:

```bash
tesseract build demo/enzyme_thermal_2d/
tesseract serve enzyme-thermal-2d
```

With that running, we can throw optimization problems at it.

### Scalar calibration: recovering 2 material parameters

Setup: a steel plate is heating up. We have thermocouple readings at 9 sensor locations, but we don't know the exact material properties. Can we recover $k_0$ (base conductivity) and $k_1$ (temperature coefficient) from sparse, noisy observations?

We generate synthetic "observed" data by running the solver with known true values ($k_0 = 45$, $k_1 = -0.02$), sample at sensor locations, and add 0.5 K of Gaussian noise. Then we start from a deliberately wrong initial guess — 33% off on $k_0$, wrong sign on $k_1$ — and hand it to L-BFGS-B.

One VJP call gives gradients with respect to both parameters simultaneously. The optimizer doesn't care that those gradients come from a compiled Fortran solver differentiated by an LLVM pass.

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

L-BFGS-B converges in about 15 iterations. The recovered parameters match the true values to within the noise floor. This is a gentle problem — only 2 unknowns, well-conditioned — so it's really a sanity check. The interesting question is what happens when we push harder.

### Thermal forensics: recovering a 900-element initial temperature field

Now for something harder. A steel plate was subjected to an unmonitored heating event — a laser pulse, a localized defect, something. Five seconds later, you measure temperatures at 100 sensor locations. Can you figure out what the initial temperature distribution looked like?

The true initial condition has two Gaussian hot spots on a warm background. That's 900 unknowns from 100 noisy observations, through a nonlinear PDE — ill-posed by any measure.

This is where reverse-mode AD becomes essential:

| Method                | Forward solves per iteration |
| --------------------- | ---------------------------: |
| Finite differences    |                  901 (n + 1) |
| VJP (reverse-mode AD) |        2 (forward + reverse) |
| **Speedup**           |                    **~450×** |

Finite differences would need 901 forward solves per iteration to get all 900 gradients. One VJP gives them all at once, for roughly the cost of two forward passes. The trade-off is memory: Enzyme's reverse mode tapes intermediate values at each time step. For this problem (~900 grid points, 100 steps), the tape is a few hundred kilobytes. For production codes with large state vectors and thousands of time steps, tape memory becomes the dominant constraint. Enzyme supports checkpointing annotations to trade recomputation for memory, but we haven't needed or tested them here.

<img src="../_static/blog/enzyme-part2-forensics.png" alt="Thermal forensics: recovering 900 initial temperature values from 100 sensors">

L-BFGS-B recovers the hot spot locations and magnitudes. Correlation between recovered and true initial temperature fields exceeds 0.99. The error is largest at the edges of the hot spots, where the signal has diffused most. The reconstruction is smoothed relative to the true field, as you'd expect from an ill-posed problem with diffusion, but it's far better than we had any right to expect from a pipeline held together with shell scripts.

That's `jax.grad` flowing through compiled Fortran, with no adjoint code written by hand.

### JAX integration via tesseract-jax

You can also go one step further with [tesseract-jax](https://github.com/pasteurlabs/tesseract-jax), which turns the Tesseract into a proper JAX primitive. Then `jax.value_and_grad` just works:

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

From JAX's perspective, the Fortran solver is just another differentiable function. You could swap `enzyme_tess` for a pure-JAX reimplementation and the optimization loop wouldn't change — only the container behind the HTTP call. Whether that abstraction is beautiful or horrifying probably depends on how you feel about gradients traversing six layers of indirection (Python → JAX → HTTP → ctypes → Enzyme → Fortran).

## Where this could go

We should be upfront: this is a 220-line solver we wrote to fit the pipeline. Enzyme handles loops, conditionals, and function calls, and the Enzyme team has demonstrated it on larger codes including [BUDE molecular docking](https://enzyme.mit.edu/getting_started/UsingEnzyme/#bude) and [LULESH hydrodynamics](https://enzyme.mit.edu/getting_started/UsingEnzyme/#lulesh), with derivative overhead factors typically between 1x and 4x. But "works on LULESH" and "works on your 30-year-old Fortran codebase" are very different claims.

What gets us excited is the composability. Imagine a Fortran CFD solver (Enzyme-differentiated) feeding into a JAX neural net surrogate:

```python
cfd_output = apply_tesseract(cfd_tess, cfd_inputs)     # Fortran + Enzyme VJP
surrogate_loss = neural_net(cfd_output["pressure"])      # JAX AD
total_loss = surrogate_loss + regularizer(cfd_inputs)    # JAX AD

grads = jax.grad(total_loss)  # chains Enzyme + JAX AD automatically
```

Each component uses its native AD. Tesseract handles the composition. We built a version of this pattern for the [rocket fin optimization](2025-11-28-rocket-fin-optimization.md) post, where analytical adjoints, finite differences, and JAX AD coexisted in one pipeline. It worked, though each new gradient source brought its own class of debugging problems.

And because Enzyme works at the LLVM IR level, none of this is Fortran-specific. The same pipeline should apply to C, C++, Rust, or any language with an LLVM frontend — though "should" is doing a lot of work in that sentence.

## What's next (and what we'd do differently)

We've described most of the sharp edges inline — the `_lfortran_malloc` crash, the `-O3` NaN disaster, LFortran's incomplete coverage. To be blunt: this works, but it's not turnkey. Expect to adapt your Fortran, pin your compiler versions, and debug at the IR level at least once.

A few things we haven't tried yet:

**Implicit time integration.** Backward Euler, Crank-Nicolson, and other implicit schemes require differentiating through iterative linear solves (CG, GMRES). Enzyme can handle this in principle, but tape memory grows with solver iteration count, and we haven't tested this path with LFortran. This is where we're headed next.

**Third-party code.** We wrote this solver to work with the pipeline. We haven't yet tried Enzyme + LFortran on a Fortran codebase we didn't author. The "will it work on _my_ code?" question is fair. Honest answer: probably, with adaptation — the kind we described above, and possibly more we haven't encountered yet.

**Solvers at scale.** MPI-parallel codes, GPU kernels, multi-physics coupling — all untested with this pipeline. Enzyme has support for some of these; we just haven't tried.

## Try it yourself

The full source — Fortran solver, Enzyme pipeline, inverse problem notebooks, and the shell scripts holding it all together — is [on GitHub](https://github.com/pasteurlabs/tesseract-core/tree/main/demo/enzyme_thermal_2d). If you have a Fortran, C, or C++ solver you want gradients for and don't mind some quality time with LLVM IR, this is a starting point.

---

_Tesseract is a free, open-source framework for differentiable scientific computing. `pip install tesseract-core`.
[Docs](https://tesseract.pasteurlabs.ai) · [Demos](https://tesseract.pasteurlabs.ai/content/demo/demo.html) · [GitHub](https://github.com/pasteurlabs/tesseract-core) · [Forum](https://si-tesseract.discourse.group/)_
