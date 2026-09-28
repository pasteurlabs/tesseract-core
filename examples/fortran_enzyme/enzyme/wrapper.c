/* Copyright 2025 Pasteur Labs. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * C wrapper that declares Enzyme AD entry points for the Fortran heat_step
 * subroutine.  After the Enzyme LLVM pass runs, the __enzyme_autodiff and
 * __enzyme_fwddiff calls are replaced with compiler-generated derivative
 * code -- no manual adjoint implementation required.
 *
 * The resulting shared library exports three functions callable from Python
 * via ctypes:
 *
 *   heat_step_forward  -- primal forward evaluation
 *   heat_step_vjp      -- reverse-mode AD  (vector-Jacobian product)
 *   heat_step_jvp      -- forward-mode AD  (Jacobian-vector product)
 */

/* Enzyme annotation sentinels (resolved by the Enzyme LLVM pass) */
int enzyme_dup;
int enzyme_const;

/* Fortran subroutine (Fortran ABI: everything by pointer) */
extern void heat_step(int* n,
                      double* T_in, double* T_out,
                      double* alpha, double* dx, double* dt);

/* Enzyme magic functions -- replaced by generated code after the pass */
extern void __enzyme_autodiff(void*, ...);
extern void __enzyme_fwddiff(void*, ...);


/* ── Forward evaluation ─────────────────────────────────────────────── */

void heat_step_forward(int n,
                       const double* T_in, double* T_out,
                       double alpha, double dx, double dt)
{
    /* Copy scalars to stack so we can take their address (Fortran ABI) */
    int n_ = n;
    double alpha_ = alpha, dx_ = dx, dt_ = dt;
    heat_step(&n_, (double*)T_in, T_out, &alpha_, &dx_, &dt_);
}


/* ── Reverse mode (VJP) ────────────────────────────────────────────── */

void heat_step_vjp(int n,
                   const double* T_in,  double* dT_in,
                   const double* T_out, double* dT_out,
                   double alpha,  double* dalpha,
                   double dx,     double* ddx,
                   double dt,     double* ddt)
{
    int n_ = n;
    double alpha_ = alpha, dx_ = dx, dt_ = dt;

    __enzyme_autodiff((void*)heat_step,
        enzyme_const, &n_,
        enzyme_dup, (double*)T_in,  dT_in,
        enzyme_dup, (double*)T_out, dT_out,
        enzyme_dup, &alpha_,        dalpha,
        enzyme_dup, &dx_,           ddx,
        enzyme_dup, &dt_,           ddt);
}


/* ── Forward mode (JVP) ────────────────────────────────────────────── */

void heat_step_jvp(int n,
                   const double* T_in,  const double* dT_in,
                   double* T_out,       double* dT_out,
                   double alpha,  double dalpha,
                   double dx,     double ddx,
                   double dt,     double ddt)
{
    int n_ = n;
    double alpha_ = alpha, dx_ = dx, dt_ = dt;
    double dalpha_ = dalpha, ddx_ = ddx, ddt_ = ddt;
    int dn_ = 0;  /* n is not differentiated */

    __enzyme_fwddiff((void*)heat_step,
        enzyme_dup, &n_,            &dn_,
        enzyme_dup, (double*)T_in,  (double*)dT_in,
        enzyme_dup, T_out,          dT_out,
        enzyme_dup, &alpha_,        &dalpha_,
        enzyme_dup, &dx_,           &ddx_,
        enzyme_dup, &dt_,           &ddt_);
}
