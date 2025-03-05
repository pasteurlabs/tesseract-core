# %% [markdown]
# # Check gradients against finite-differences
#
# This is an exploratory notebook where I'll experiment how to check Tesseract Jacobian, JVPs and VJPs against finite difference
# calculations.
#
# I'll do this in python, but the idea is to have a command in the runtime which does this check,
# which will be called directly from the sdk cli in a similar way to what we do for serve.
# The signature of this command will mostly be the same as vjp, jvp, and jacobian, with maybe some extra args
# to specify epsilon and the direction of the finite differences.
#
# Let's start by just performing the calculation of a Jacobian, using
# vectoradd_jax because it has a nontrivial structure of inputs and
# outputs:
# %%
import numpy as np

from tesseract_core import Tesseract

a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])

x0 = {"a": {"v": a, "s": 1}, "b": {"v": b, "s": 2}}

with Tesseract.from_image("vectoradd_jax") as t:
    result = t.apply(x0)
    jacobian = t.jacobian(
        x0,
        jac_inputs=["a.v", "b.s"],
        jac_outputs=["vector_min.result", "vector_add.normed_result"],
    )
    jvp = t.jacobian_vector_product(
        x0,
        jvp_inputs=["a.v", "b.s"],
        jvp_outputs=["vector_min.result", "vector_add.normed_result"],
        tangent_vector={"a.v": 0.5 * a, "b.s": 1.0},
    )
    vjp = t.vector_jacobian_product(
        x0,
        vjp_inputs=["a.v", "b.s"],
        vjp_outputs=["vector_min.result", "vector_add.normed_result"],
        cotangent_vector={
            "vector_min.result": 0.1 * np.ones(3),
            "vector_add.normed_result": np.array([1.0, 0.0, 0.0]),
        },
    )
result
# %%
jacobian
# %%
jvp
# %%
vjp
# %% [markdown]
# How would we implement a finite difference check for the jacobian? We could
# keep the same signature as jacobian, and component-wise add a "small" perturbation.
# A heuristic we could use is that for each component $x_i$, we add/subtract a perturbation of size $K x_i$,
# with $K \sim 10^{-3}$ or so. This has some shortcomings (what if some components are zero, and so on)
# which we could address later, but it seems to me like a good choice for a variety of reasons,
# including the fact that it automatically takes into consideration possibly different ranges
# for different components.
#
# Notice also that this does not calculate the full jacobian, but gives a result which can be
# compared with $J x$ -- otherwise we would need to run this calculation component-wise.
# %%
from copy import deepcopy

import tesseract_core.runtime.tree_transforms as tt


def fd_check_jacobian(inputs, jac_inputs, jac_outputs, K=1e-3):
    x0 = inputs
    x0_plus_eps = deepcopy(inputs)
    x0_minus_eps = deepcopy(inputs)
    epsilons = []

    # Here we build the perturbed inputs
    for in_ in jac_inputs:
        vec = tt.get_at_path(x0, in_)

        eps = K * vec
        epsilons.append(eps)

        x0_plus_eps = tt.set_at_path(x0_plus_eps, {in_: vec + eps})
        x0_minus_eps = tt.set_at_path(x0_minus_eps, {in_: vec - eps})

    # TODO: of course in the runtime this would be substituted by the
    #       actual apply and jacobian
    with Tesseract.from_image("vectoradd_jax") as t:
        f_x0_plus_eps = t.apply(x0_plus_eps)
        f_x0_minus_eps = t.apply(x0_minus_eps)
        jacobian = t.jacobian(
            x0,
            jac_inputs=["a.v", "b.s"],
            jac_outputs=["vector_min.result", "vector_add.normed_result"],
        )

    # Now we compute the finite-difference derivatives
    fd_result = deepcopy(jacobian)
    for out in jac_outputs:
        for in_, eps in zip(jac_inputs, epsilons):
            contrib_plus = tt.get_at_path(f_x0_plus_eps, out)
            contrib_minus = tt.get_at_path(f_x0_minus_eps, out)
            derivative = (contrib_plus - contrib_minus) / (2 * eps)
            fd_result[out][in_] = derivative

    # We then compare with J x and fd_result
    # TODO: return comparison rather than fd_result
    return fd_result


fd_check_jacobian(
    x0,
    jac_inputs=["a.v", "b.s"],
    jac_outputs=["vector_min.result", "vector_add.normed_result"],
)
# %%
with Tesseract.from_image("vectoradd_jax") as t:
    result = t.apply(x0)
    jacobian = t.jacobian(
        x0,
        jac_inputs=["a.v", "b.s"],
        jac_outputs=["vector_min.result", "vector_add.normed_result"],
    )
