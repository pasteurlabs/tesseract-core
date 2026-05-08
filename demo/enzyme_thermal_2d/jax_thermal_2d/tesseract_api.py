# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX reimplementation of the 2D thermal solver from enzyme_thermal_2d.

This Tesseract solves the same physics as its Enzyme/Fortran counterpart:
  rho * cp * dT/dt = div( k(T) * grad(T) ) + Q
with identical boundary conditions, discretization, and interface.

The key difference: instead of compiling Fortran with Enzyme, the solver is
written directly in JAX. Derivatives (JVP, VJP, Jacobian) are obtained via
jax.jvp / jax.vjp — no manual adjoint code, no compilation pipeline.

This enables a direct performance comparison between:
  - Enzyme: exact AD on compiled Fortran (legacy solver story)
  - JAX: native Python AD with XLA JIT compilation (rewrite story)
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

# -- Schemas (identical to enzyme_thermal_2d) ----------------------------------


class InputSchema(BaseModel):
    """Input for a 2D transient heat conduction solver.

    Solves rho*cp*dT/dt = div(k(T)*grad(T)) + Q on a rectangular domain
    with Dirichlet (hot wall), convection, and insulated boundary conditions.
    """

    T_init: Differentiable[Array[(None,), Float64]] = Field(
        description=(
            "Initial temperature field [K]. Flattened row-major array of "
            "shape (nx*ny,). Index (i,j) maps to j*nx+i."
        ),
    )
    nx: int = Field(
        default=20, description="Number of grid points in x direction.", ge=3
    )
    ny: int = Field(
        default=20, description="Number of grid points in y direction.", ge=3
    )
    n_steps: int = Field(
        default=100, description="Number of explicit Euler time steps.", ge=1
    )
    dt: Differentiable[Float64] = Field(
        default=0.01, description="Time step size [s].", gt=0.0
    )
    Lx: Differentiable[Float64] = Field(
        default=0.1, description="Domain length in x [m].", gt=0.0
    )
    Ly: Differentiable[Float64] = Field(
        default=0.05, description="Domain length in y [m].", gt=0.0
    )
    k0: Differentiable[Float64] = Field(
        default=45.0,
        description="Base thermal conductivity [W/(m*K)]. k(T) = k0 + k1*T.",
        gt=0.0,
    )
    k1: Differentiable[Float64] = Field(
        default=-0.01,
        description="Temperature coefficient of conductivity [W/(m*K^2)]. k(T) = k0 + k1*T.",
    )
    rho: Differentiable[Float64] = Field(
        default=7850.0, description="Density [kg/m^3].", gt=0.0
    )
    cp: Differentiable[Float64] = Field(
        default=460.0, description="Specific heat capacity [J/(kg*K)].", gt=0.0
    )
    h_conv: Differentiable[Float64] = Field(
        default=25.0,
        description="Convective heat transfer coefficient at top boundary [W/(m^2*K)].",
        gt=0.0,
    )
    T_inf: Differentiable[Float64] = Field(
        default=293.15, description="Ambient temperature for convection BC [K]."
    )
    T_hot: Differentiable[Float64] = Field(
        default=373.15,
        description="Fixed temperature at bottom (Dirichlet) boundary [K].",
    )
    Q: Differentiable[Array[(None,), Float64]] = Field(
        description=(
            "Volumetric heat source [W/m^3]. Flattened row-major array of "
            "shape (nx*ny,). Use zeros for no internal heating."
        ),
    )

    @model_validator(mode="after")
    def check_array_sizes(self) -> Self:
        expected = self.nx * self.ny
        if len(self.T_init) != expected:
            raise ValueError(
                f"T_init has {len(self.T_init)} elements, expected nx*ny = {expected}."
            )
        if len(self.Q) != expected:
            raise ValueError(
                f"Q has {len(self.Q)} elements, expected nx*ny = {expected}."
            )
        return self

    @model_validator(mode="after")
    def check_stability(self) -> Self:
        dx = self.Lx / (self.nx - 1)
        dy = self.Ly / (self.ny - 1)
        k_max = self.k0 + self.k1 * self.T_hot
        if k_max <= 0:
            raise ValueError(f"Conductivity k(T_hot) = {k_max:.4f} <= 0.")
        r = k_max * self.dt / (self.rho * self.cp) * (1.0 / (dx * dx) + 1.0 / (dy * dy))
        if r > 0.5:
            raise ValueError(f"CFL stability condition violated: r = {r:.4f} > 0.5.")
        return self


class OutputSchema(BaseModel):
    T_final: Differentiable[Array[(None,), Float64]] = Field(
        description="Temperature field after n_steps time steps [K]. Flattened row-major (nx*ny,).",
    )


# -- JAX solver ----------------------------------------------------------------


def _harmonic_mean(ka, kb):
    """Harmonic mean of two conductivities (standard for cell-face averaging)."""
    return 2.0 * ka * kb / (ka + kb)


def _thermal_2d_step(T, nx, ny, dx, dy, k0, k1, rho, cp, h_conv, T_inf, T_hot, Q, dt):
    """One explicit Euler step of the 2D thermal solver.

    T is a 2D array of shape (ny, nx). Returns updated T_new of same shape.
    """
    # Conductivity field
    K = k0 + k1 * T

    # Harmonic-mean conductivities at cell faces (interior)
    kx_east = _harmonic_mean(K[:, :-1], K[:, 1:])  # (ny, nx-1)
    ky_north = _harmonic_mean(K[:-1, :], K[1:, :])  # (ny-1, nx)

    # x-direction flux: d/dx(k dT/dx)
    flux_x_faces = kx_east * (T[:, 1:] - T[:, :-1]) / dx  # (ny, nx-1)
    flux_x = jnp.zeros_like(T)
    # Interior: east - west
    flux_x = flux_x.at[:, 1:-1].set((flux_x_faces[:, 1:] - flux_x_faces[:, :-1]) / dx)
    # Left boundary (i=0): insulated => mirror, net flux = k_east*(T_e - T_c) / dx^2
    flux_x = flux_x.at[:, 0].set(flux_x_faces[:, 0] / dx)
    # Right boundary (i=nx-1): insulated => mirror, net flux = -k_west*(T_c - T_w) / dx^2
    # which is k_west*(T_w - T_c) / dx^2
    flux_x = flux_x.at[:, -1].set(-flux_x_faces[:, -1] / dx)

    # y-direction flux: d/dy(k dT/dy)
    flux_y_faces = ky_north * (T[1:, :] - T[:-1, :]) / dy  # (ny-1, nx)
    flux_y = jnp.zeros_like(T)
    # Interior
    flux_y = flux_y.at[1:-1, :].set((flux_y_faces[1:, :] - flux_y_faces[:-1, :]) / dy)
    # Bottom (j=0): Dirichlet — will be overwritten, but set flux anyway
    flux_y = flux_y.at[0, :].set(flux_y_faces[0, :] / dy)
    # Top (j=ny-1): convection BC: -k dT/dn = h_conv*(T - T_inf)
    # Interior conduction from below + convective loss from above
    flux_y = flux_y.at[-1, :].set(
        -flux_y_faces[-1, :] / dy - h_conv * (T[-1, :] - T_inf) / dy
    )

    Q_2d = Q.reshape(ny, nx)
    T_new = T + dt / (rho * cp) * (flux_x + flux_y + Q_2d)

    # Enforce Dirichlet BC at bottom
    T_new = T_new.at[0, :].set(T_hot)

    return T_new


def _thermal_2d_solve(
    T_init_2d, nx, ny, n_steps, dx, dy, k0, k1, rho, cp, h_conv, T_inf, T_hot, Q, dt
):
    """Run n_steps of explicit Euler. Uses jax.lax.fori_loop for AD compatibility."""

    def body_fn(_, T):
        return _thermal_2d_step(
            T, nx, ny, dx, dy, k0, k1, rho, cp, h_conv, T_inf, T_hot, Q, dt
        )

    return jax.lax.fori_loop(0, n_steps, body_fn, T_init_2d)


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    nx = inputs["nx"]
    ny = inputs["ny"]
    n_steps = inputs["n_steps"]
    dx = inputs["Lx"] / (nx - 1)
    dy = inputs["Ly"] / (ny - 1)

    T_init_2d = inputs["T_init"].reshape(ny, nx)
    Q = inputs["Q"]

    T_final_2d = _thermal_2d_solve(
        T_init_2d,
        nx,
        ny,
        n_steps,
        dx,
        dy,
        inputs["k0"],
        inputs["k1"],
        inputs["rho"],
        inputs["cp"],
        inputs["h_conv"],
        inputs["T_inf"],
        inputs["T_hot"],
        Q,
        inputs["dt"],
    )

    return {"T_final": T_final_2d.reshape(-1)}


# -- Required endpoint ---------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the 2D thermal solver for n_steps explicit Euler steps."""
    out = apply_jit(inputs.model_dump())
    return {"T_final": np.asarray(out["T_final"])}


# -- JAX-handled gradient endpoints (no need to modify) ------------------------


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jvp_jit(
        inputs.model_dump(),
        tuple(jvp_inputs),
        tuple(jvp_outputs),
        tangent_vector,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return vjp_jit(
        inputs.model_dump(),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs):
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


# -- Helper functions ----------------------------------------------------------


@eqx.filter_jit
def jac_jit(inputs: dict, jac_inputs: tuple[str], jac_outputs: tuple[str]):
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )


@eqx.filter_jit
def jvp_jit(
    inputs: dict, jvp_inputs: tuple[str], jvp_outputs: tuple[str], tangent_vector: dict
):
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]
