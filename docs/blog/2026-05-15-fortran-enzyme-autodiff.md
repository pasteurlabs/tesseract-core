---
orphan: true
og:title: "Differentiable Fortran with LFortran and Enzyme"
og:description: "How we built a pipeline to get exact gradients out of a Fortran thermal solver, then used it to solve real inverse problems from Python."
blog_date: "2026-06-15"
blog_author: "@dionhaefner"
blog_title: "Differentiable Fortran with LFortran and Enzyme"
blog_description: "How we built a pipeline to get exact gradients out of a Fortran thermal solver, then used it to solve real inverse problems from Python."
---

# Differentiable Fortran with LFortran and Enzyme

What if you could backpropagate through existing Fortran, C, or C++ simulation code, embed it into JAX and torch, and use it as a high-performance differentiable physics engine? Turns out, you can — if you're brave enough...

Decades of validated physics code in CFD, climate, aerospace, and nuclear sit behind a wall that modern ML pipelines can't cross, because they don't expose gradients. The usual answer is to rewrite it all in JAX or PyTorch. The alternative we explore here is to leave the code where it is and get exact gradients out anyway, thanks to some LLVM-level magic. [Enzyme](https://enzyme.mit.edu/) applies autodiff at the LLVM IR level, so we can differentiate any code that compiles to LLVM.

All we need is to duct-tape [LFortran](https://lfortran.org/), LLVM, and Enzyme together, point the result at a Fortran thermal solver, and get exact gradients out the other end. From there, [Tesseract](https://github.com/pasteurlabs/tesseract-core) (whose blog you're reading) wraps the result as a custom [JAX](https://jax.readthedocs.io/en/latest/) primitive, so a Fortran solver becomes a differentiable layer in arbitrary JAX code.

This is all pretty experimental, so you may have to spend a half day chasing a gradient that returned NaN and painstakingly compare LLVM IR diffs to make it work. But the gradients come out the other end of the entire multi-step time loop matching an analytic gradient. And it's amazing to see that a stack combining some of the oldest and newest technologies can work together at all.

But let's start at the beginning. What follows is the full walkthrough, including the compilation pipeline, the sharp edges we hit, and an inverse problem that made the whole effort worth it (and that wouldn't be possible without AD).

## The problem

If you work in scientific computing, you might have run into this: you have a simulation written in Fortran (or C, or C++), and now someone needs gradients, the derivatives of the simulation's outputs with respect to its inputs. Maybe it's for optimization, maybe inverse problems, maybe plugging the sim into an ML pipeline. Now your options are:

- **Hand-written adjoints.** Boils down to hand-implementing the derivative of every line of code, which means months of expert effort. Error-prone and a maintenance nightmare that slowly drifts out of sync with the forward code.
- **Finite differences.** Perturb every input and difference the outputs. Slow ($O(n)$ evaluations for n parameters), inaccurate, and poorly conditioned for stiff problems.
- **Rewrite in JAX or PyTorch and use autodiff.** Sure, if you want to rewrite tens of thousands of lines of validated physics.

What if you could just compile the derivatives automatically, from the existing source? Here's what it actually looks like when you try.

(a-fortran-heat-solver)=

## A Fortran heat solver

Our test subject is `thermal_2d.f90`, about 220 lines of vanilla Fortran 90 that we wrote for this experiment. It solves 2D transient heat conduction with temperature-dependent conductivity:

$$\rho \, c_p \frac{\partial T}{\partial t} = \nabla \cdot \big( k(T) \, \nabla T \big) + Q$$

The conductivity $k$ follows a linear material model, $k(T) = k_0 + k_1 \cdot T$, and time integration is explicit Euler over `n_steps` steps. Nothing exotic here.

Here's what the subroutine signature and the interior stencil loop look like:

```fortran
! The solver we differentiate: a 2D heat conduction step loop in plain Fortran 90
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

Boundary conditions are mixed: Dirichlet (hot wall at the bottom), convection/Robin (top), and insulated/Neumann (sides). The faces use harmonic-mean conductivity, the standard trick for flux continuity across cells with different $k$.

For this setup, there are no easy shortcuts to get derivatives: because $k(T)$ is nonlinear, the stencil coefficients depend on the current temperature field, so the Jacobian changes at every time step. Hand-coding the adjoint through that, and re-deriving it every time the forward code changes, is the kind of work AD exists to delete.

This is vanilla Fortran with no annotations or AD-aware constructs, but even getting to "vanilla" took some massaging. Allocatables, array intrinsics, and bounds checking all compile down to runtime calls (`_lfortran_malloc` and friends) that Enzyme can't see through, so they had to go. Work arrays get passed in pre-allocated from C, and we compile with `--no-array-bounds-checking`. For a 220-line solver that's an hour of work; for a large legacy codebase, it's an open question.

## The compilation pipeline

This is where it gets technical, so if you're not into the low-level details, feel free to [skip to the next section](#does-it-actually-work) where we benchmark the gradients.

Enzyme works at the LLVM IR level, not the source level, so anything that compiles to LLVM IR (C, C++, Rust, Fortran) can be differentiated. The chain is six `opt`/`clang` invocations, surprisingly straightforward as long as you're comfortable inspecting IR when things go wrong.

```{figure} ../static/blog/enzyme-pipeline.png
:alt: "Compilation pipeline: Fortran → Enzyme AD → shared library"

The six-step compilation pipeline, from Fortran source to a differentiable shared library. LFortran lowers the solver to LLVM IR, a thin C wrapper is linked in, Enzyme synthesizes the derivative code, and a final optimization pass emits a single `.so` exposing forward, JVP, and VJP entry points. The whole chain runs in about 30 seconds during `tesseract build`.
```

Most of those steps are plumbing: lower the Fortran to IR, compile a thin C wrapper, `llvm-link` the two, run Enzyme, optimize, emit a `.so`. The three that actually matter are the LFortran IR dump, the pre-Enzyme `-O1` cleanup, and the Enzyme pass itself.

```bash
# The compilation pipeline: Fortran source → differentiated shared library

# Fortran → LLVM IR (clean enough for Enzyme to trace)
$ lfortran --show-llvm --no-array-bounds-checking thermal_2d.f90 > thermal_2d.ll

# Mild cleanup BEFORE the AD pass — -O1, deliberately not -O3 (see below)
$ opt -O1 -S thermal_2d.ll -o thermal_2d_opt.ll

# ... link in the C wrapper etc ...

# Synthesize the derivative
$ opt --load-pass-plugin=LLVMEnzyme-19.so -passes=enzyme combined.ll -o ad.ll

# ... final optimization and emit a shared library ...
```

A couple of decisions along the way shaped how the whole thing holds together, though, and they're worth dwelling on.

(why-lfortran)=

**Why LFortran.** The first step (`lfortran --show-llvm`) is also the reason the whole thing is tractable, since LFortran emits remarkably clean IR. Arrays come out as plain pointers with the usual GEP/load/store patterns, much like C, instead of the multi-field descriptor structs and runtime calls that Flang produces. That matters enormously, because Enzyme has to trace every memory access to figure out what's active, and it has a far easier time with pointer arithmetic than with opaque `fir.box` descriptors.

"Clean" is doing a lot of work in that sentence, so it's worth pinning down what it means here. Three things, concretely. Every array reference lowers to a plain `getelementptr` + `load`/`store` on a bare pointer. The 19-argument subroutine takes 19 plain `ptr` arguments, with no descriptor struct wrapping any of them. And the whole module contains zero function calls: no `_lfortran_malloc`, no runtime helpers, nothing opaque for Enzyme to trace through. Here's the IR LFortran emits for the start of the stencil body, covering the `idx = (j-1)*nx + i` index and the `T_c = T_cur(idx)` / `kx_east = ...` lines from the source above:

```llvm
; What LFortran emits for the stencil body: plain pointer math, no descriptors
; idx = (j - 1) * nx + i
%56 = sub i32 %55, 1                       ; j - 1
%58 = mul i32 %56, %57                      ; (j-1) * nx
%60 = add i32 %58, %59                      ; + i
store i32 %60, ptr %idx

; T_c = T_cur(idx)
%67 = getelementptr inbounds double, ptr %t_cur, i32 %65   ; &T_cur[idx]
%68 = load double, ptr %67                                 ; T_cur[idx]
store double %68, ptr %t_c

; kx_east = 2*(k0+k1*T_c)*(k0+k1*T_e) / ((k0+k1*T_c)+(k0+k1*T_e))
%110 = fmul double %108, %109              ; k1 * T_c
%111 = fadd double %107, %110              ; k0 + k1*T_c
%112 = fmul double 2.0, %111
; ... k0 + k1*T_e, the product, the sum ...
%130 = fdiv double %118, %129              ; the harmonic mean
store double %130, ptr %kx_east
```

That's it. Array access is a `getelementptr` + `load`, arithmetic is a flat sequence of `fmul`/`fadd`/`fdiv` that maps line-for-line onto the Fortran. Flang, by contrast, would wrap those same arrays in descriptor structs and reach for runtime helpers, and Enzyme would have to see through every one of them.

The catch is that LFortran is still maturing, so your code has to stay inside what it [supports](https://lfortran.org/progress/).

**Why `-O1`, not `-O3`.** The `opt -O1` line above looks innocuous, and our first pipeline used `-O3` there. The forward pass worked perfectly, so we moved on. Later, the VJP returned NaN on certain inputs, and it took hours to find why. At `-O3`, LLVM's aggressive vectorization and code-motion produce IR patterns Enzyme apparently mishandles in reverse mode. In our case it bit when adjacent cell temperatures were equal, which caused intermediate terms cancel and a compiler rearrangement turned that into a division by zero. The fix is to keep optimization mild _before_ the AD pass and save `-O3` for _after_ it.

We certainly learned to test the gradients early and often.

**C bridge.** In between Fortran and Enzyme sits a thin C wrapper that bridges Fortran's by-pointer ABI to a C interface Enzyme can annotate. It allocates the work arrays on the heap (sidestepping the `_lfortran_malloc` issue) and marks which arguments get shadow buffers via Enzyme's intrinsics.

```c
// The C bridge: heap-allocates work arrays and tells Enzyme what to differentiate
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

**AD tracing via Enzyme.** The Enzyme pass (`-passes=enzyme`) is where the real AD work happens. It analyzes the linked IR and synthesizes the derivative code, using a store-all tape strategy that caches intermediates at each time step. When it works, you get an adjoint of your entire time-stepping loop for free. When it doesn't, you're reading IR diffs at 2 AM (see [What's next](#whats-next-and-what-wed-do-differently)).

A final `-O3 + clang -shared` step emits a single `.so` with three entry points (forward, JVP, VJP) callable from Python via ctypes. The whole pipeline runs during `tesseract build`, about 30 seconds end to end.

### What Enzyme actually does to the IR

Let's have a look at what Enzyme does to the IR. By this point the [clean LFortran IR](#why-lfortran) has been linked together with the C wrapper into one module, and that's what Enzyme consumes. Going _into_ Enzyme (step 4 output), the VJP wrapper is just an ordinary call: the `__enzyme_autodiff` intrinsic, handed a pointer to the solver and an `enzyme_const`/`enzyme_dup` marker in front of each argument (loaded here from globals as `%50`/`%51`). Each `dup` argument is a value/shadow pair, and the buffer that will receive its gradient comes right after it:

```llvm
; combined.ll — before the Enzyme pass
%50 = load i32, ptr @enzyme_const          ; the enzyme_const marker
%51 = load i32, ptr @enzyme_dup            ; the enzyme_dup marker
call void (ptr, ...) @__enzyme_autodiff(
    ptr nonnull @thermal_2d_solve,         ; function to differentiate
    i32 %50, ptr nonnull %33,              ; const:  n_steps  (no shadow)
    i32 %51, ptr %3, ptr %4,               ; dup:    T_init,  dT_init  (shadow)
    i32 %51, ptr %5, ptr %6,               ; dup:    k0,      dk0
    ; ... 15 more dup pairs ...
    i32 %51, ptr nonnull %43, ptr %28)
```

After the pass (step 5 output), that intrinsic is gone, and `__enzyme_autodiff` appears nowhere in the module except its now-unused declaration. In its place is a function Enzyme synthesized from scratch, `@diffethermal_2d_solve`. Its signature is the giveaway: every active argument from the original solver now appears twice, the value followed immediately by a `'`-suffixed shadow that will receive its gradient.

```llvm
; ad.ll — the synthesized function's signature
define internal void @diffethermal_2d_solve(
    ptr %n, ptr %nx, ptr %ny, ptr %n_steps,    ; integer args — no shadows (inactive)
    ptr %t_init, ptr %"t_init'",                ; T_init  +  its shadow
    ptr %t_final, ptr %"t_final'",
    ptr %t_cur, ptr %"t_cur'",
    ptr %t_new, ptr %"t_new'",
    ptr %k0, ptr %"k0'",                        ; k0      +  its shadow
    ptr %k1, ptr %"k1'", ...)                   ; ...one pair per active argument
```

The named shadows (`%"t_init'"`, `%"k0'"`) are where the gradients ultimately land. The interesting part is the body: Enzyme grew a whole reverse pass onto the function, a set of basic blocks (`invertloop.body`, `invert.entry`, …) that replay the time loop _backwards_ and feed those shadows. Here's one iteration of the innermost reverse loop, untouched from the generated IR — the temporaries it works through carry the same `'` mark:

```llvm
; The reverse sweep Enzyme synthesized: the time loop, replayed backwards
invertloop.body:
  %914 = load double, ptr %"'ipg_unwrap"             ; read adjoint of T_cur[idx]
  store double 0.0,  ptr %"'ipg_unwrap"              ; ...then zero it (it's consumed)
  %915 = load double, ptr %"'de5"
  %916 = fadd fast double %915, %914                 ; accumulate into a temporary
  store double %916, ptr %"'de5"
  %917 = load double, ptr %"'de5"
  store double 0.0,  ptr %"'de5"
  %919 = load double, ptr %"'ipg6_unwrap"            ; read adjoint of T_init[idx] (slot of t_init')
  %920 = fadd fast double %919, %917                 ; propagate it upstream
  store double %920, ptr %"'ipg6_unwrap"
  ; ...
  %925 = add nsw i64 %924, -1                        ; loop counter runs DOWNWARD
  br label %invertloop.body
```

The loop induction variable counts _down_ (`add nsw i64 ..., -1`), and the body is the textbook reverse-mode dance: read an adjoint, zero it so it isn't double-counted, accumulate it into the upstream variable's shadow. Nobody wrote any of this — it's the chain rule, applied statement by statement to the solver's own IR. The full `@diffethermal_2d_solve` is about 6,900 lines. The 865-line input module comes out the other side at nearly 9,900.

(does-it-actually-work)=

## Does it actually work?

OK, it compiles. But after the `-O3` NaN incident, trusting it without checking is off the table. The safest test is to compare Enzyme's gradient against a ground-truth derivative we can compute _independently_, without Enzyme anywhere in the loop.

For one regime we can do exactly that. If we set the conductivity's temperature coefficient $k_1 = 0$, the solver becomes linear: each explicit step is an affine map of the temperature field, $T^{s+1} = A(k_0)\,T^s + c(k_0)$, where the operators $A$ and $c$ encode the real stencil and the real mixed boundary conditions, and depend affinely on $k_0$. (Verified numerically: one step is affine in $k_0$ to a relative residual of ~1e-16.) That structure hands us the exact derivative of the _whole_ multi-step trajectory for free, via the tangent recurrence

$$\frac{\partial T^{s+1}}{\partial k_0} = A_1\,T^s + A(k_0)\,\frac{\partial T^s}{\partial k_0} + c_1,$$

which we iterate in plain NumPy. The reconstructed affine model reproduces the solver's actual 500-step trajectory to a relative error of ~1e-14, so it really is differentiating the discrete solver and not an approximation of it. Then we compare against Enzyme's VJP of $\sum_i T_{\mathrm{final},i}$:

| Method                    | Relative error vs. analytic gradient |
| ------------------------- | -----------------------------------: |
| **Enzyme AD**             |              **~6e-12** (≈11 digits) |
| FD (best $\epsilon$)      |                               ~1e-10 |
| FD ($\epsilon$ too small) |                                ~1e-2 |
| FD ($\epsilon$ too large) |                                ~1e-6 |

Enzyme reproduces the independently-derived gradient to about 11 significant digits. The small gap from full machine precision is likely roundoff in the reference itself, accumulated over 500 steps of operator iteration. This is the linear regime, so it stresses the full multi-step time loop and the real boundary conditions, though not the nonlinear stencil.

Finite differences, by contrast, never get this close. They essentially compute $\big(f(x + \epsilon) - f(x - \epsilon)\big) / 2\epsilon$, and that only works in a narrow sweet spot: for too-large $\epsilon$ truncation error dominates, for too-small $\epsilon$ roundoff takes over, and even at the best step size they bottom out orders of magnitude above Enzyme. You can watch the trade-off play out across step sizes, with Enzyme's flat line sitting underneath the entire FD curve.

```{figure} ../static/blog/enzyme-analytic_benchmark.png
:alt: "Enzyme vs. finite differences, both measured against an independent analytic gradient"

Relative error of Enzyme AD and central finite differences, both measured against an independent analytic gradient derived from the solver's affine structure in the linear ($k_1 = 0$) regime. The finite-difference error traces the classic U-shaped trade-off between truncation error (large step sizes) and roundoff (small step sizes), bottoming out near $10^{-10}$. Enzyme's error is a flat line at ~$6\times10^{-12}$, below the entire FD curve and free of any step size to tune.
```

That U-shaped FD curve is the classic truncation-vs-roundoff trade-off. Enzyme sidesteps it by computing the analytic derivative directly, at any input, with no step size to tune, and with O(1) scaling in the number of parameters vs. O(n) for finite differences.

## Wiring things into JAX

Correct gradients are only worth something once you put them to work, whether that means dropping them into an optimizer, solving a real inverse problem, or composing them with other differentiable code. This is most easily done from Python, so the next step is to get the Enzyme gradients out of the compiled library and into JAX.

To wire the Enzyme gradients into JAX, the solver has to look like a differentiable JAX primitive, and this is more or less exactly what Tesseract was built for. It wraps the build process and compiled library, with LFortran, LLVM 19, Enzyme, and the whole toolchain inside it, into a container that exposes autodiff endpoints. Building and serving it is two commands:

```bash
# Package the compiled solver as a served, differentiable JAX primitive
$ tesseract build demo/enzyme_thermal_2d/
$ tesseract serve enzyme-thermal-2d
```

The last piece is [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), which promotes a served Tesseract to a proper JAX primitive. Wrap a call in `apply_tesseract` and the Fortran solver becomes an ordinary differentiable JAX function: `jax.value_and_grad` routes its VJP calls straight through the HTTP boundary to the Enzyme-generated code, and the optimizer never has to know.

```python
# The Fortran solver, now callable as a differentiable JAX function
from tesseract_jax import apply_tesseract

def solve(T_init):
    inputs = {**base_inputs, "T_init": T_init}
    return apply_tesseract(enzyme_tess, inputs)["T_final"]  # differentiable Fortran!
```

With that in place, it's finally time to point a real optimization problem at it and find out whether all the effort actually paid off.

## Thermal forensics: recovering a 900-element initial temperature field

Imagine a steel plate that went through some unmonitored heating event, maybe a laser pulse, maybe a localized defect, nobody knows. Five seconds later you get to measure temperatures at 100 sensor locations, and the question is whether you can reconstruct what the initial temperature distribution must have looked like.

The true initial condition has two Gaussian hot spots sitting on a warm background. So the ask is 900 unknowns out of 100 noisy observations, run backwards through a nonlinear PDE. A small Tikhonov term (penalizing departure from the ambient prior) regularizes the otherwise ill-posed inversion, and since the loss is just a JAX function, that term is one extra line that `jax.value_and_grad` differentiates for free:

```python
# The inverse-problem loss: data misfit + Tikhonov prior, differentiated end to end
def loss_fn(T_init):
    T_pred = solve(T_init)                        # apply_tesseract under the hood
    residuals = T_pred[sensor_indices] - T_obs
    data = 0.5 * jnp.sum(residuals**2)
    reg = lam * jnp.sum((T_init - T_ambient)**2)  # Tikhonov regularization
    return data + reg

# gradients w.r.t. all 900 initial-field values
value_and_grad = jax.value_and_grad(loss_fn)
```

One `value_and_grad` call hands back the gradient with respect to all 900 unknowns at once. This is exactly the regime where reverse-mode AD (or its cousin the adjoint method) becomes essential:

| Method                | Forward solves per iteration |
| --------------------- | ---------------------------: |
| Finite differences    |                  901 (n + 1) |
| VJP (reverse-mode AD) |        2 (forward + reverse) |
| **Speedup**           |                    **~450×** |

Finite differences would have to do 901 forward solves every iteration just to assemble all 900 gradients, while one VJP hands them all back at once, for roughly the cost of two forward passes. What you pay for that is memory, since Enzyme's reverse mode has to tape intermediate values at each time step. For a problem this size (~900 grid points, 100 steps) the tape is only a few hundred kilobytes, small enough to ignore. On production codes with large state vectors and thousands of time steps, though, that tape is usually the thing that ends up dominating the memory budget. Enzyme does have checkpointing annotations that trade recomputation for memory, but they weren't needed here and remain untested on this path.

SciPy's [L-BFGS-B optimizer](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) takes the value and gradient together via `jac=True`, so the entire Enzyme-through-JAX path is just the objective it calls:

```python
from scipy.optimize import minimize

# scipy wants plain floats/NumPy; value_and_grad returns both in one sweep
def objective(T_init_vec):
    loss, grad = value_and_grad(jnp.asarray(T_init_vec))
    return float(loss), np.asarray(grad, dtype=np.float64)

result = minimize(
    objective,
    x0=np.full(900, T_ambient),   # start from uniform ambient: the wrong answer
    method="L-BFGS-B",
    jac=True,                     # objective returns (value, gradient)
    bounds=[(250.0, 450.0)] * 900,
)
```

```{figure} ../static/blog/enzyme-thermal_forensics.png
:alt: "Thermal forensics: recovering 900 initial temperature values from 100 sensors"
:class: blog-img-full

The thermal forensics inverse problem: recovering a 900-element initial temperature field from 100 noisy sensor readings taken five seconds later. L-BFGS-B, driven by Enzyme's reverse-mode gradients through the Fortran solver, locates both Gaussian hot spots and recovers their magnitudes, reaching a correlation of ~0.98 with the true field. The reconstruction is slightly smoothed at the hot-spot edges, where diffusion has erased the most information — the expected signature of an ill-posed problem.
```

And it works! L-BFGS-B finds both hot spots, gets their locations and magnitudes right, and the correlation between the recovered and true initial fields comes out around 0.98. Where it struggles is at the edges of the hot spots, which is exactly where the signal has had the most time to diffuse away. The reconstruction comes out a little smoothed compared to the truth, which is about what you'd expect from an ill-posed problem with diffusion in it. That's `jax.grad` flowing all the way through compiled Fortran, without a single line of adjoint code written by hand, delivering 900 gradient components from a single reverse sweep.

And because the solver is just another JAX function behind `apply_tesseract`, you could swap `enzyme_tess` out for a pure-JAX reimplementation tomorrow and nothing in the optimization loop would change, only the container sitting behind the HTTP call. Whether you find that abstraction beautiful or horrifying probably comes down to how you feel about your gradients quietly traversing six layers of indirection on the way through (Python → JAX/XLA → HTTP → ctypes → Enzyme → Fortran).

(whats-next-and-what-wed-do-differently)=

## What's next

A reality check before getting carried away: this is a 220-line solver written specifically to fit the pipeline. Enzyme itself handles loops, conditionals, and function calls just fine, and the Enzyme team has run it on much larger codes, including [BUDE molecular docking](https://enzyme.mit.edu/getting_started/UsingEnzyme/#bude) and [LULESH hydrodynamics](https://enzyme.mit.edu/getting_started/UsingEnzyme/#lulesh), with derivative overhead that usually lands somewhere between 1x and 4x. Still, "works on LULESH" and "works on your 30-year-old Fortran codebase" are two very different claims.

We've flagged the sharp edges as we went, from the `_lfortran_malloc` crash to the `-O3` NaN incident to LFortran's still-incomplete coverage; the approach works, but it is nowhere near turnkey yet. Plan on adapting your Fortran, pinning your compiler versions, and dropping down to the IR level to debug at least once before you're done.

Two bigger questions stay open. One is **third-party code**: "but will it work on _my_ code?" is a perfectly fair thing to ask. The honest guess is that it probably will, after the kind of adaptation described earlier and likely a few surprises beyond it. The other is **scale**, where checkpointing schemes, MPI-parallel codes, GPU kernels, and multi-physics coupling all remain untested here. Enzyme has support for some of them, that's just territory this pipeline hasn't reached.

Where this gets genuinely exciting, though, is **composability**. Picture an Enzyme-differentiated Fortran CFD solver feeding straight into a JAX neural-net surrogate:

```python
# Composability: chain Enzyme-differentiated Fortran with native JAX AD

cfd_output = apply_tesseract(cfd_tess, cfd_inputs)     # Fortran + Enzyme VJP
surrogate_loss = neural_net(cfd_output["pressure"])      # JAX AD
total_loss = surrogate_loss + regularizer(cfd_inputs)    # JAX AD

grads = jax.grad(total_loss)  # chains Enzyme + JAX AD automatically
```

Every component differentiates itself with whatever AD is native to it, and Tesseract is the thing that stitches the pieces together. This is similar to the pattern showcased in the [rocket fin optimization](2025-11-28-rocket-fin-optimization.md) post, where analytical adjoints, finite differences, and JAX AD all coexisted in a single pipeline. It works, though every new gradient source tends to bring along its own fresh class of debugging headaches.

Because Enzyme works at the LLVM IR level, none of this is Fortran-specific. The same pipeline should apply to C, C++, Rust, or any language with an LLVM frontend.

## Try it yourself

The full source is [on GitHub](https://github.com/pasteurlabs/tesseract-core/tree/main/demo/enzyme_thermal_2d), including the Fortran solver, the Enzyme pipeline, the inverse-problem notebooks, and the LLVM build script holding it all together. If you've got a Fortran, C, or C++ solver you'd like gradients for and you don't mind spending some quality time with LLVM IR, this is a pretty good place to start.

The pieces are here, and the remaining questions are exactly the kind of thing that's more fun with company. If you take it somewhere, whether you get it running on your own solver or hit a wall we didn't, come tell us on the [Forum](https://si-tesseract.discourse.group/). We'd genuinely like to see how far it goes!
