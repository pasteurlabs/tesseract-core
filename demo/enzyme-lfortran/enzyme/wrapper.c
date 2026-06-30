/* Copyright 2025 Pasteur Labs. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * C wrapper that declares Enzyme AD entry points for the 2D thermal solver.
 * After the Enzyme LLVM pass runs, the __enzyme_autodiff and __enzyme_fwddiff
 * calls are replaced with compiler-generated derivative code.
 *
 * Work arrays (T_cur, T_new) are allocated here and passed to the Fortran
 * subroutine to avoid _lfortran_malloc calls in the differentiated code.
 *
 * The resulting shared library exports three functions callable from Python
 * via ctypes:
 *
 *   thermal_2d_forward  -- primal forward evaluation
 *   thermal_2d_vjp      -- reverse-mode AD  (vector-Jacobian product)
 *   thermal_2d_jvp      -- forward-mode AD  (Jacobian-vector product)
 */

#include <stdlib.h>
#include <string.h>

/* Enzyme annotation sentinels (resolved by the Enzyme LLVM pass) */
int enzyme_dup;
int enzyme_const;

/* Fortran subroutine (Fortran ABI: everything by pointer)
 * Note: n = nx*ny is passed explicitly so the Fortran side sees fixed-size
 * arrays and never calls _lfortran_malloc. */
extern void thermal_2d_solve(int* n, int* nx, int* ny, int* n_steps,
                             double* T_init, double* T_final,
                             double* T_cur, double* T_new,
                             double* k0, double* k1,
                             double* rho, double* cp,
                             double* h_conv, double* T_inf, double* T_hot,
                             double* Q, double* Lx, double* Ly, double* dt);

/* Enzyme magic functions -- replaced by generated code after the pass */
extern void __enzyme_autodiff(void*, ...);
extern void __enzyme_fwddiff(void*, ...);


/* -- Forward evaluation --------------------------------------------------- */

void thermal_2d_forward(int nx, int ny, int n_steps,
                        const double* T_init, double* T_final,
                        double k0, double k1,
                        double rho, double cp,
                        double h_conv, double T_inf, double T_hot,
                        const double* Q, double Lx, double Ly, double dt)
{
    int n = nx * ny;
    int nx_ = nx, ny_ = ny, n_steps_ = n_steps, n_ = n;
    double k0_ = k0, k1_ = k1, rho_ = rho, cp_ = cp;
    double h_conv_ = h_conv, T_inf_ = T_inf, T_hot_ = T_hot;
    double Lx_ = Lx, Ly_ = Ly, dt_ = dt;

    double* T_cur = (double*)calloc(n, sizeof(double));
    double* T_new = (double*)calloc(n, sizeof(double));

    thermal_2d_solve(&n_, &nx_, &ny_, &n_steps_,
                     (double*)T_init, T_final, T_cur, T_new,
                     &k0_, &k1_, &rho_, &cp_,
                     &h_conv_, &T_inf_, &T_hot_,
                     (double*)Q, &Lx_, &Ly_, &dt_);

    free(T_cur);
    free(T_new);
}


/* -- Reverse mode (VJP) --------------------------------------------------- */

void thermal_2d_vjp(int nx, int ny, int n_steps,
                    const double* T_init,  double* dT_init,
                    const double* T_final, double* dT_final,
                    double k0,     double* dk0,
                    double k1,     double* dk1,
                    double rho,    double* drho,
                    double cp,     double* dcp,
                    double h_conv, double* dh_conv,
                    double T_inf,  double* dT_inf,
                    double T_hot,  double* dT_hot,
                    const double* Q, double* dQ,
                    double Lx,     double* dLx,
                    double Ly,     double* dLy,
                    double dt,     double* ddt)
{
    int n = nx * ny;
    int nx_ = nx, ny_ = ny, n_steps_ = n_steps, n_ = n;
    double k0_ = k0, k1_ = k1, rho_ = rho, cp_ = cp;
    double h_conv_ = h_conv, T_inf_ = T_inf, T_hot_ = T_hot;
    double Lx_ = Lx, Ly_ = Ly, dt_ = dt;

    /* Work arrays and their shadows (zero-initialized) */
    double* T_cur  = (double*)calloc(n, sizeof(double));
    double* dT_cur = (double*)calloc(n, sizeof(double));
    double* T_new  = (double*)calloc(n, sizeof(double));
    double* dT_new = (double*)calloc(n, sizeof(double));

    __enzyme_autodiff((void*)thermal_2d_solve,
        enzyme_const, &n_,
        enzyme_const, &nx_,
        enzyme_const, &ny_,
        enzyme_const, &n_steps_,
        enzyme_dup, (double*)T_init,  dT_init,
        enzyme_dup, (double*)T_final, dT_final,
        enzyme_dup, T_cur,            dT_cur,
        enzyme_dup, T_new,            dT_new,
        enzyme_dup, &k0_,     dk0,
        enzyme_dup, &k1_,     dk1,
        enzyme_dup, &rho_,    drho,
        enzyme_dup, &cp_,     dcp,
        enzyme_dup, &h_conv_, dh_conv,
        enzyme_dup, &T_inf_,  dT_inf,
        enzyme_dup, &T_hot_,  dT_hot,
        enzyme_dup, (double*)Q, dQ,
        enzyme_dup, &Lx_,     dLx,
        enzyme_dup, &Ly_,     dLy,
        enzyme_dup, &dt_,     ddt);

    free(T_cur);
    free(dT_cur);
    free(T_new);
    free(dT_new);
}


/* -- Forward mode (JVP) --------------------------------------------------- */

void thermal_2d_jvp(int nx, int ny, int n_steps,
                    const double* T_init,  const double* dT_init,
                    double* T_final,       double* dT_final,
                    double k0,     double dk0,
                    double k1,     double dk1,
                    double rho,    double drho,
                    double cp,     double dcp,
                    double h_conv, double dh_conv,
                    double T_inf,  double dT_inf,
                    double T_hot,  double dT_hot,
                    const double* Q, const double* dQ,
                    double Lx,     double dLx,
                    double Ly,     double dLy,
                    double dt,     double ddt)
{
    int n = nx * ny;
    int nx_ = nx, ny_ = ny, n_steps_ = n_steps, n_ = n;
    int dnx_ = 0, dny_ = 0, dn_steps_ = 0, dn_ = 0;
    double k0_ = k0, k1_ = k1, rho_ = rho, cp_ = cp;
    double h_conv_ = h_conv, T_inf_ = T_inf, T_hot_ = T_hot;
    double Lx_ = Lx, Ly_ = Ly, dt_ = dt;
    double dk0_ = dk0, dk1_ = dk1, drho_ = drho, dcp_ = dcp;
    double dh_conv_ = dh_conv, dT_inf_ = dT_inf, dT_hot_ = dT_hot;
    double dLx_ = dLx, dLy_ = dLy, ddt_ = ddt;

    /* Work arrays and their tangent shadows */
    double* T_cur  = (double*)calloc(n, sizeof(double));
    double* dT_cur = (double*)calloc(n, sizeof(double));
    double* T_new  = (double*)calloc(n, sizeof(double));
    double* dT_new = (double*)calloc(n, sizeof(double));

    __enzyme_fwddiff((void*)thermal_2d_solve,
        enzyme_dup, &n_,       &dn_,
        enzyme_dup, &nx_,      &dnx_,
        enzyme_dup, &ny_,      &dny_,
        enzyme_dup, &n_steps_, &dn_steps_,
        enzyme_dup, (double*)T_init,  (double*)dT_init,
        enzyme_dup, T_final,          dT_final,
        enzyme_dup, T_cur,            dT_cur,
        enzyme_dup, T_new,            dT_new,
        enzyme_dup, &k0_,     &dk0_,
        enzyme_dup, &k1_,     &dk1_,
        enzyme_dup, &rho_,    &drho_,
        enzyme_dup, &cp_,     &dcp_,
        enzyme_dup, &h_conv_, &dh_conv_,
        enzyme_dup, &T_inf_,  &dT_inf_,
        enzyme_dup, &T_hot_,  &dT_hot_,
        enzyme_dup, (double*)Q,  (double*)dQ,
        enzyme_dup, &Lx_,     &dLx_,
        enzyme_dup, &Ly_,     &dLy_,
        enzyme_dup, &dt_,     &ddt_);

    free(T_cur);
    free(dT_cur);
    free(T_new);
    free(dT_new);
}
