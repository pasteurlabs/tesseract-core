---
orphan: true
og:title: "Differentiable Fortran via LFortran and Enzyme"
og:description: "How we duct-taped a compiler pipeline together to get exact gradients out of a Fortran thermal solver, then used it to solve real inverse problems from Python."
blog_date: "2026-05-15"
blog_author: "@dionhaefner"
blog_title: "Differentiable Fortran via LFortran and Enzyme"
blog_description: "How we duct-taped a compiler pipeline together to get exact gradients out of a Fortran thermal solver, then used it to solve real inverse problems from Python."
---

# Differentiable Fortran via LFortran and Enzyme

What if you could do autodiff through existing Fortran, C, or C++ simulation code, embed it into JAX and torch, and use it as a high-performance differentiable physics engine? Turns out, you can.

Decades of validated physics code within CFD, climate, aerospace, nuclear, sit behind a wall that modern ML pipelines can't cross: no gradients. The usual answer is to rewrite it all in JAX or PyTorch. The alternative we explore here is to leave the code where it is and get exact gradients out anyway, thanks to some LLVM-level magic. [Enzyme](https://enzyme.mit.edu/) applies AD at the LLVM IR level, so we can differentiate any code that compiles to LLVM.

So all we need is to duct-tape [LFortran](https://lfortran.org/), LLVM, and Enzyme together, point the result at a Fortran thermal solver, and get exact gradients out the other end. From there, [Tesseract](https://github.com/pasteurlabs/tesseract-core) (whose blog you're reading) wraps the result as a custom [JAX](https://jax.readthedocs.io/en/latest/) primitive, so a Fortran solver becomes a differentiable layer in arbitrary JAX code. Sounds easy, right?

This is all pretty experimental, so we had to spend a half day chasing a gradient that returned NaN and painstakingly compare LLVM IR diffs to make it work. But the gradients come out of the entire multi-step time loop matching the analytic derivative to machine precision (~1e-15 relative error), where finite differences top out around 1e-8. And it's amazing to see that a stack combining some of the oldest and newest technologies out there can work together at all.

So let's start at the beginning. What follows is the full walkthrough: the compilation pipeline, the sharp edges we hit, and two inverse problems that made the whole effort worth it (and that wouldn't be possible without AD).

## The problem

If you work in scientific computing, you might have run into this: you have a simulation written in Fortran (or C, or C++), and now someone needs gradients (derivatives of the simulation's outputs with respect to its inputs). Maybe it's for optimization, maybe inverse problems, maybe plugging the sim into an ML pipeline. Now your options are:

- **Hand-written adjoints.** Months of expert effort. Error-prone. A maintenance nightmare that slowly drifts out of sync with the forward code.
- **Finite differences.** Slow (O(n) evaluations for n parameters), inaccurate (you're always hunting for the truncation-vs-roundoff sweet spot), and poorly conditioned for stiff problems.
- **Rewrite in JAX or PyTorch.** Sure, if you want to rewrite tens of thousands of lines of validated physics.

What if you could just compile the derivatives automatically, from the existing source? Here's what it actually looks like when you try.

## The Fortran code

Our test subject is `thermal_2d.f90`, about 220 lines of vanilla Fortran 90. It solves 2D transient heat conduction with temperature-dependent conductivity:

$$\rho \, c_p \frac{\partial T}{\partial t} = \nabla \cdot \big( k(T) \, \nabla T \big) + Q$$

The conductivity follows a linear material model, $k(T) = k_0 + k_1 \cdot T$, and time integration is explicit Euler over `n_steps` steps. Nothing exotic here.

Here's what the subroutine signature and the interior stencil loop look like:

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

Boundary conditions are mixed — Dirichlet (hot wall at the bottom), convection/Robin (top), insulated/Neumann (sides) — and the faces use harmonic-mean conductivity, the standard trick for flux continuity across cells with different $k$. This is exactly where AD earns its keep: because $k(T)$ is nonlinear, the stencil coefficients depend on the current temperature field, so the Jacobian changes at every time step. Hand-coding the adjoint through that, and re-deriving it every time the forward code changes, is the kind of work AD exists to delete.

It's vanilla Fortran — explicit `do` loops, no annotations, no AD-aware constructs — but getting to "vanilla" took some massaging. Allocatables, array intrinsics, and bounds checking all compile down to runtime calls (`_lfortran_malloc` and friends) that Enzyme can't see through, so they had to go: work arrays get passed in pre-allocated from C, and we compile with `--no-array-bounds-checking`. For a 220-line solver that's an afternoon's work; for a large legacy codebase, it's an open question.

## The pipeline

Enzyme works at the LLVM IR level, not the source level, so anything that compiles to LLVM IR (C, C++, Rust, Fortran) can be differentiated. The chain is six `opt`/`clang` invocations, surprisingly straightforward as long as you're comfortable inspecting IR when things go wrong:

<img src="../_static/blog/enzyme-pipeline.png" alt="Compilation pipeline: Fortran → Enzyme AD → shared library">

Most of those steps are plumbing — lower the Fortran to IR, compile a thin C wrapper, `llvm-link` the two, run Enzyme, optimize, emit a `.so`. Two of them are where all the interesting failures live.

**Why LFortran.** The first step (`lfortran --show-llvm`) is also the reason the whole thing is tractable: LFortran emits remarkably clean IR. Arrays come out as plain pointers with the usual GEP/load/store patterns, much like C, instead of the multi-field descriptor structs and runtime calls that Flang produces. That matters enormously, because Enzyme has to trace every memory access to figure out what's active, and it has a far easier time with pointer arithmetic than with opaque `fir.box` descriptors. The catch: LFortran is still maturing, so your code has to stay inside what it [supports](https://lfortran.org/progress/).

**Why `-O1`, not `-O3`.** The pre-Enzyme optimization step looks innocuous, and our first pipeline used `-O3` there. The forward pass worked perfectly, so we moved on — but the VJP returned NaN on certain inputs, and it took hours to find why. At `-O3`, LLVM's aggressive vectorization and code-motion produce IR patterns Enzyme mishandles in reverse mode; in our case it bit when adjacent cell temperatures were equal, intermediate terms cancelled, and a rearrangement turned that into a division by zero. The fix is to keep optimization mild _before_ the AD pass and save `-O3` for _after_ it. If you build something like this: test the gradients early and often.

In between sits a thin C wrapper that bridges Fortran's by-pointer ABI to a C interface Enzyme can annotate. It allocates the work arrays on the heap (sidestepping the `_lfortran_malloc` issue) and marks which arguments get shadow buffers via Enzyme's intrinsics:

```c
void thermal_2d_vjp(/* ... nx, ny, n_steps ... */)
{
    double* T_cur = calloc(n, sizeof(double));   // heap, not allocatable
    /* ... */
    __enzyme_autodiff((void*)thermal_2d_solve,
        enzyme_const, &nx_,                       // not differentiated
        enzyme_dup,   (double*)T_init, dT_init,   // value + shadow buffer
        enzyme_dup,   &k0_, dk0,
        /* ... */);
}
```

The Enzyme pass (`-passes=enzyme`) is where the real work happens: it analyzes the linked IR and synthesizes the derivative code, using a store-all tape strategy that caches intermediates at each time step. When it works, you get an adjoint of your entire time-stepping loop for free. When it doesn't, you're reading IR diffs at 2 AM (see [What's next](#whats-next-and-what-wed-do-differently)). The final `-O3 + clang -shared` step emits a single `.so` with three entry points — forward, JVP, VJP — callable from Python via ctypes. The whole pipeline runs during `tesseract build`, about 30 seconds end to end.

## Does it actually work?

OK, it compiles. But after the `-O3` NaN incident, trusting it without checking was off the table. So the VJP gets called from Python and compared against central finite differences at various step sizes:

<img src="../_static/blog/enzyme-fd-convergence.png" alt="Enzyme vs. finite difference gradient accuracy">

Finite differences essentially compute $\big(f(x + \epsilon) - f(x - \epsilon)\big) / 2\epsilon$, and they only work well in a narrow sweet spot. Pick too large an $\epsilon$ and truncation error dominates; pick too small and roundoff takes over. In practice the best you can hope for is around $10^{-8}$ relative error. Enzyme's gradients, on the other hand, agree to machine precision (about $10^{-15}$), and that's no accident: they're the analytically correct derivative, just synthesized by a compiler rather than worked out by hand. And that holds across the entire multi-step time loop, with a nonlinear stencil update at every step.

| Method                    | Relative error vs. exact |
| ------------------------- | -----------------------: |
| Enzyme AD                 |           ~1e-15 (exact) |
| FD ($\epsilon$ too small) |                    ~1e-4 |
| FD (best $\epsilon$)      |                    ~1e-8 |
| FD ($\epsilon$ too large) |                    ~1e-2 |

Error is worst at both ends of the $\epsilon$ range and best in the middle, the classic truncation-vs-roundoff trade-off. Enzyme sidesteps it entirely.

## Now do something useful with it

Correct gradients are only worth something once you put them to work, whether that means dropping them into an optimizer, solving a real inverse problem, or composing them with other differentiable code.

To wire the Enzyme gradients into JAX, the solver has to look like a differentiable JAX primitive, and this is more or less exactly what Tesseract was built for. It wraps the compiled library, with LFortran, LLVM 19, Enzyme, and the whole toolchain inside it, into a container that exposes autodiff endpoints. Building and serving it is two commands:

```bash
$ tesseract build demo/enzyme_thermal_2d/
$ tesseract serve enzyme-thermal-2d
```

The last piece is [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), which promotes a served Tesseract to a proper JAX primitive. Wrap a call in `apply_tesseract` and the Fortran solver becomes an ordinary differentiable JAX function: `jax.value_and_grad` routes its VJP calls straight through the HTTP boundary to the Enzyme-generated code, and the optimizer never has to know.

```python
from tesseract_jax import apply_tesseract

def solve(k0, k1):
    inputs = {**base_inputs, "k0": k0, "k1": k1}
    return apply_tesseract(enzyme_tess, inputs)["T_final"]  # differentiable
```

With that in place, it's finally time to point some real optimization problems at it and find out whether all the effort actually paid off.

### Scalar calibration: recovering 2 material parameters

A sanity check first: a steel plate heats up, 9 thermocouples read its temperature, and the material properties are unknown. Can we recover $k_0$ (base conductivity) and $k_1$ (its temperature coefficient) from those sparse, noisy readings? To keep the test honest, the "observed" data is synthetic — the solver run at known true values ($k_0 = 45$, $k_1 = -0.02$), sampled at the sensors, plus 0.5 K of Gaussian noise — and the optimizer starts from a deliberately bad guess: 33% off on $k_0$, the wrong sign entirely on $k_1$. The loss is a plain JAX function that happens to call the Fortran solver:

```python
def loss_fn(params):
    k0, k1 = params
    T_pred = solve(k0, k1)                       # apply_tesseract under the hood
    residuals = T_pred[sensor_indices] - T_obs
    return 0.5 * jnp.sum(residuals**2)

# One reverse sweep → gradients w.r.t. both k0 and k1
value_and_grad = jax.value_and_grad(loss_fn)
loss, grad = value_and_grad(jnp.array([60.0, 0.01]))
```

<img src="../_static/blog/enzyme-part1-convergence.png" alt="Scalar calibration convergence: loss, k0, k1">

L-BFGS-B converges in about 20 iterations, nailing $k_0$ to within a few percent. $k_1$ lands further out — the temperature field is only weakly sensitive to it, so 9 noisy sensors barely constrain it — but the recovered field still matches the observations. The more interesting question is what happens once you push the method a lot harder.

### Thermal forensics: recovering a 900-element initial temperature field

Imagine a steel plate that went through some unmonitored heating event, maybe a laser pulse, maybe a localized defect, nobody knows. Five seconds later you get to measure temperatures at 100 sensor locations, and the question is whether you can reconstruct what the initial temperature distribution must have looked like.

The true initial condition has two Gaussian hot spots sitting on a warm background. So the ask is 900 unknowns out of 100 noisy observations, run backwards through a nonlinear PDE. A small Tikhonov term (penalizing departure from the ambient prior) regularizes the otherwise ill-posed inversion, and since the loss is just a JAX function, that term is one extra line that `jax.value_and_grad` differentiates for free. The optimization is the same `jax.value_and_grad` + L-BFGS-B as before; only the parameter vector grows from 2 to 900. This is exactly the regime where reverse-mode AD (or its cousin the adjoint method) stops being a nice-to-have and becomes essential:

| Method                | Forward solves per iteration |
| --------------------- | ---------------------------: |
| Finite differences    |                  901 (n + 1) |
| VJP (reverse-mode AD) |        2 (forward + reverse) |
| **Speedup**           |                    **~450×** |

Finite differences would have to do 901 forward solves every iteration just to assemble all 900 gradients, while one VJP hands them all back at once, for roughly the cost of two forward passes. What you pay for that is memory, since Enzyme's reverse mode has to tape intermediate values at each time step. For a problem this size (~900 grid points, 100 steps) the tape is only a few hundred kilobytes, small enough to ignore. On production codes with large state vectors and thousands of time steps, though, that tape is usually the thing that ends up dominating the memory budget. Enzyme does have checkpointing annotations that trade recomputation for memory, but they weren't needed here and remain untested on this path.

<img src="../_static/blog/enzyme-part2-forensics.png" alt="Thermal forensics: recovering 900 initial temperature values from 100 sensors">

And it works! L-BFGS-B finds both hot spots, gets their locations and magnitudes right, and the correlation between the recovered and true initial fields comes out around 0.98. Where it struggles is at the edges of the hot spots, which is exactly where the signal has had the most time to diffuse away. The reconstruction comes out a little smoothed compared to the truth, which is about what you'd expect from an ill-posed problem with diffusion in it, but it's honestly far better than a pipeline held together with shell scripts has any right to deliver.

That's `jax.grad` flowing all the way through compiled Fortran, without a single line of adjoint code written by hand. The same `value_and_grad` call powered both inverse problems above; only the size of the parameter vector changed.

And because the solver is just another JAX function behind `apply_tesseract`, you could swap `enzyme_tess` out for a pure-JAX reimplementation tomorrow and nothing in the optimization loop would change, only the container sitting behind the HTTP call. Whether you find that abstraction beautiful or horrifying probably comes down to how you feel about your gradients quietly traversing six layers of indirection on the way through (Python → JAX/XLA → HTTP → ctypes → Enzyme → Fortran).

## Where this could go

The real value-add here is composability. Picture an Enzyme-differentiated Fortran CFD solver feeding straight into a JAX neural-net surrogate:

```python
cfd_output = apply_tesseract(cfd_tess, cfd_inputs)     # Fortran + Enzyme VJP
surrogate_loss = neural_net(cfd_output["pressure"])      # JAX AD
total_loss = surrogate_loss + regularizer(cfd_inputs)    # JAX AD

grads = jax.grad(total_loss)  # chains Enzyme + JAX AD automatically
```

Every component differentiates itself with whatever AD is native to it, and Tesseract is the thing that stitches the pieces together. This is similar to the pattern showcased in the [rocket fin optimization](2025-11-28-rocket-fin-optimization.md) post, where analytical adjoints, finite differences, and JAX AD all coexisted in a single pipeline. It works, though every new gradient source tends to bring along its own fresh class of debugging headaches.

Because Enzyme works at the LLVM IR level, none of this is Fortran-specific. The same pipeline should apply to C, C++, Rust, or any language with an LLVM frontend, which we graciously leave as an exercise for the reader.

(whats-next-and-what-wed-do-differently)=

## What's next (and what we'd do differently)

A reality check before getting carried away: this is a 220-line solver written specifically to fit the pipeline. Enzyme itself handles loops, conditionals, and function calls just fine, and the Enzyme team has run it on much larger codes, including [BUDE molecular docking](https://enzyme.mit.edu/getting_started/UsingEnzyme/#bude) and [LULESH hydrodynamics](https://enzyme.mit.edu/getting_started/UsingEnzyme/#lulesh), with derivative overhead that usually lands somewhere between 1x and 4x. Still, "works on LULESH" and "works on your 30-year-old Fortran codebase" are two very different claims.

We've flagged the sharp edges as we went, from the `_lfortran_malloc` crash to the `-O3` NaN disaster to LFortran's still-incomplete coverage. So the approach works, but it is nowhere near turnkey yet. Plan on adapting your Fortran, pinning your compiler versions, and dropping down to the IR level to debug at least once before you're done.

The path we're most interested in next is **implicit time integration**. Backward Euler, Crank-Nicolson, and the other implicit schemes all require differentiating through an iterative linear solve (CG, GMRES, and friends). Enzyme can do this in principle, but the tape memory grows with the solver's iteration count, which comes with its own set of challenges.

Two bigger questions stay open past that. One is **third-party code**: since this solver was written to fit the pipeline, the obvious "but will it work on _my_ code?" is a perfectly fair thing to ask. The honest guess is that it probably will, given the kind of adaptation described earlier and likely a few surprises beyond it. The other is **scale**, where MPI-parallel codes, GPU kernels, and multi-physics coupling all remain untested here. Enzyme has support for some of them; that's just territory this pipeline hasn't reached.

## Try it yourself

The full source is [on GitHub](https://github.com/pasteurlabs/tesseract-core/tree/main/demo/enzyme_thermal_2d), the Fortran solver, the Enzyme pipeline, the inverse-problem notebooks, and the shell scripts holding it all together. If you've got a Fortran, C, or C++ solver you'd like gradients for and you don't mind spending some quality time with LLVM IR, this is a pretty good place to start.

The pieces are here, and the open questions above (implicit solvers, third-party code, scale) are exactly the kind of thing that's more fun with company. If you take it somewhere, whether you get it running on your own solver or hit a wall we didn't, come tell us on the [forum](https://si-tesseract.discourse.group/). We'd genuinely like to see how far it goes!

---

_Tesseract is a free, open-source framework for differentiable scientific computing. `pip install tesseract-core`.
[Docs](https://tesseract.pasteurlabs.ai) · [Demos](https://tesseract.pasteurlabs.ai/content/demo/demo.html) · [GitHub](https://github.com/pasteurlabs/tesseract-core) · [Forum](https://si-tesseract.discourse.group/)_
