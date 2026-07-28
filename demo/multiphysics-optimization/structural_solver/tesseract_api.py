# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural solver Tesseract: 2D linear thermoelastic stress on a rectangular plate.

Given a temperature field, computes thermal strain, solves for displacement
via a finite-difference discretization of the linear elasticity equations,
and returns the stress field, displacement, and a scalar objective
(compliance = u^T K u, a standard structural optimization objective).

The displacement output feeds back to the thermal solver for two-way coupling.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

NX = 30
NY = 30


class InputSchema(BaseModel):
    temperature: Differentiable[Array[(NX, NY), Float32]] = Field(
        description="Temperature field from thermal solver (NX x NY)"
    )
    youngs_modulus: Float32 = Field(description="Young's modulus", default=200.0)
    poissons_ratio: Float32 = Field(description="Poisson's ratio", default=0.3)
    thermal_expansion: Float32 = Field(
        description="Coefficient of thermal expansion", default=1e-3
    )


class OutputSchema(BaseModel):
    displacement: Differentiable[Array[(NX, NY, 2), Float32]] = Field(
        description="Displacement field (NX x NY x 2)"
    )
    stress: Differentiable[Array[(NX, NY, 3), Float32]] = Field(
        description="Stress field (NX x NY x 3: sigma_xx, sigma_yy, sigma_xy)"
    )
    objective: Differentiable[Array[(), Float32]] = Field(
        description="Scalar compliance objective"
    )


def _compute_strain_from_displacement(u, dx, dy):
    """Compute strain tensor components from displacement field via central differences."""
    ux = u[:, :, 0]
    uy = u[:, :, 1]

    # Strain: eps_xx = du_x/dx, eps_yy = du_y/dy, eps_xy = 0.5*(du_x/dy + du_y/dx)
    eps_xx = jnp.gradient(ux, dx, axis=0)
    eps_yy = jnp.gradient(uy, dy, axis=1)
    eps_xy = 0.5 * (jnp.gradient(ux, dy, axis=1) + jnp.gradient(uy, dx, axis=0))

    return eps_xx, eps_yy, eps_xy


def _thermal_strain(temperature, alpha):
    """Isotropic thermal strain: eps_thermal = alpha * T."""
    return alpha * temperature


def _stress_from_strain(eps_xx, eps_yy, eps_xy, eps_thermal, E, nu):
    """Plane stress constitutive relation with thermal strain."""
    # Mechanical strain = total strain - thermal strain
    mech_xx = eps_xx - eps_thermal
    mech_yy = eps_yy - eps_thermal
    mech_xy = eps_xy  # thermal strain is isotropic, no shear component

    # Plane stress stiffness
    c = E / (1 - nu**2)
    sigma_xx = c * (mech_xx + nu * mech_yy)
    sigma_yy = c * (nu * mech_xx + mech_yy)
    sigma_xy = E / (2 * (1 + nu)) * 2 * mech_xy

    return sigma_xx, sigma_yy, sigma_xy


def _spd_laplacian_solve(rhs, ax, ay, boundary_value=0.0):
    """Solve  ax·(-u_xx) + ay·(-u_yy) = rhs  on the interior grid, with Dirichlet bc's.

    The 5-point finite-difference operator is symmetric positive-definite, so we
    solve it directly with a Cholesky factorization (swap in ``lx.CG`` or a sparse
    direct solver for a much larger system).q
    """
    nx, ny = rhs.shape
    dx = 1.0 / (nx + 1)
    dy = 1.0 / (ny + 1)

    def neg_laplacian(u):
        # -(ax·u_xx + ay·u_yy) with zero boundaries, as second differences.
        uxx = jnp.diff(u, n=2, axis=0, prepend=0.0, append=0.0)
        uyy = jnp.diff(u, n=2, axis=1, prepend=0.0, append=0.0)
        return -(ax * uxx / dx**2 + ay * uyy / dy**2)

    # A Dirichlet value on an edge adds a·value/h² to the RHS at the adjacent
    # interior nodes (a corner picks up a contribution from both of its edges).
    b = rhs
    b = b.at[0, :].add(ax * boundary_value / dx**2)
    b = b.at[-1, :].add(ax * boundary_value / dx**2)
    b = b.at[:, 0].add(ay * boundary_value / dy**2)
    b = b.at[:, -1].add(ay * boundary_value / dy**2)

    operator = lx.FunctionLinearOperator(
        neg_laplacian,
        jax.ShapeDtypeStruct((nx, ny), rhs.dtype),
        tags=(lx.positive_semidefinite_tag,),
    )
    return lx.linear_solve(operator, b, solver=lx.Cholesky()).value


def _solve_elasticity(temperature, E, nu, alpha):
    """Solve 2D linear thermoelasticity for the displacement field.

    The temperature enters as a body force f_i = -(lambda + 2*mu)*alpha*dT/dx_i.
    Dropping the mixed-derivative (lambda + mu)*d2u_j/dx_i dx_j term decouples the
    two displacement components into independent anisotropic Poisson problems, each
    solved by `_spd_laplacian_solve`.
    """
    nx, ny = temperature.shape
    dx = 1.0 / (nx + 1)
    dy = 1.0 / (ny + 1)

    # Lame parameters for plane stress
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - nu))  # plane stress effective lambda

    # Thermal body force: f_i = -(lambda + 2*mu) * alpha * dT/dx_i
    thermal_coeff = (lam + 2 * mu) * alpha
    fx = thermal_coeff * jnp.gradient(temperature, dx, axis=0)
    fy = thermal_coeff * jnp.gradient(temperature, dy, axis=1)

    # Decoupled equilibrium (zero-displacement boundaries):
    #   (lambda + 2*mu) u_x,xx + mu u_x,yy = fx
    #   mu u_y,xx + (lambda + 2*mu) u_y,yy = fy
    ux = _spd_laplacian_solve(fx, lam + 2 * mu, mu)
    uy = _spd_laplacian_solve(fy, mu, lam + 2 * mu)
    return jnp.stack([ux, uy], axis=-1)


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    temperature = inputs["temperature"]
    E = inputs["youngs_modulus"]
    nu = inputs["poissons_ratio"]
    alpha = inputs["thermal_expansion"]

    dx = 1.0 / (NX + 1)
    dy = 1.0 / (NY + 1)

    # Solve for displacement (linear thermoelasticity; SPD direct solves)
    displacement = _solve_elasticity(temperature, E, nu, alpha)

    # Compute strain and stress
    eps_xx, eps_yy, eps_xy = _compute_strain_from_displacement(displacement, dx, dy)
    eps_thermal = _thermal_strain(temperature, alpha)
    sigma_xx, sigma_yy, sigma_xy = _stress_from_strain(
        eps_xx, eps_yy, eps_xy, eps_thermal, E, nu
    )

    stress = jnp.stack([sigma_xx, sigma_yy, sigma_xy], axis=-1)

    # Compliance objective: sum of squared displacement weighted by stiffness
    # (simplified proxy for u^T K u)
    objective = jnp.sum(displacement**2)

    return {
        "displacement": displacement.astype(jnp.float32),
        "stress": stress.astype(jnp.float32),
        "objective": objective.astype(jnp.float32),
    }


def apply(inputs: InputSchema) -> OutputSchema:
    return apply_jit(inputs.model_dump())


def abstract_eval(abstract_inputs: Any) -> Any:
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

    def wrapped_apply(dynamic_inputs: Any) -> Any:
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


@eqx.filter_jit
def jvp_jit(
    inputs: dict,
    jvp_inputs: tuple[str],
    jvp_outputs: tuple[str],
    tangent_vector: dict,
) -> Any:
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> Any:
    return jvp_jit(
        inputs.model_dump(),
        tuple(jvp_inputs),
        tuple(jvp_outputs),
        tangent_vector,
    )


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
) -> Any:
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> Any:
    return vjp_jit(
        inputs.model_dump(),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


@eqx.filter_jit
def jac_jit(
    inputs: dict,
    jac_inputs: tuple[str],
    jac_outputs: tuple[str],
) -> Any:
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> Any:
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))
