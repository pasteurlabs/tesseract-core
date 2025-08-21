import os

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
from animate import make_animation

from tesseract_core import Tesseract

# In this example, we use tesseract to call and differentiate through the qpax solver.
# github: https://github.com/kevin-tracy/qpax/tree/main
# paper: https://arxiv.org/abs/2406.11749

# The task we're solving here is to tune a linear constraint in a QP,
# so that the solution of the QP matches a desired target value, i.e.

# min_G (x*(G) - x_true)^2
# s.t.  x*(G) \in argmin_x 0.5 * x^T Q x + q^T x
#                 s.t.  G x <= h,
# where x_true is the target value, and G are the coefficients we tune.
# The qpax tesseract is used to solve the inner QP and obtain gradients
# of x* w.r.t. G, which we then use to update G using gradient descent
# on the outer objective.


# we want to minimize the decision error
def decision_error(x_pred, x_true):
    return jnp.dot(x_pred - x_true, x_pred - x_true)


decision_grad_fn = jax.grad(decision_error, argnums=0)

# we want to tune a constraint so that this becomes the solution of our QP
x_true = jnp.array([-1.5, -1.5])

# QP definition
n = 2
Q = jnp.eye(n)
q = jnp.ones(n)
A = None
b = None
# we will adjust G to change the QP solution
G = jnp.array([-1.0, 1.0]).reshape((1, 2))  # inequality constraint Gx <= h
h = jnp.array([-1.0]).reshape((1,))  # inequality constraint h
input = {
    "Q": Q,
    "q": q,
    "A": A,
    "b": b,
    "G": G,
    "h": h,
    # this is a smoothing coeffient for the QP solver,
    # if this is too low, the gradients can become unstable.
    "target_kappa": 1e-1,
    "solver_tol": 1e-4,
}

max_iterations = 200
lrs = 0.1 * jnp.ones(max_iterations)
tol = 1e-3
save_every = 1

with Tesseract.from_image("qp_solve") as qp_solve:
    constraints = []
    losses = []
    sols = []
    for k in range(max_iterations):
        output = qp_solve.apply(input)

        x_pred = output["x"]
        loss = decision_error(x_pred, x_true)
        if k % save_every == 0:
            sols.append(x_pred)
            losses.append(loss)
            constraints.append((input["G"], input["h"]))
        print(f"Iteration {k + 1}, Loss: {loss:.2f}")
        if loss < tol:
            sols.append(x_pred)
            losses.append(loss)
            constraints.append((input["G"], input["h"]))
            print(f"Converged at iteration {k + 1} with loss {loss}")
            break

        tangent_vector = decision_grad_fn(x_pred, x_true)
        try:
            output = qp_solve.vector_jacobian_product(
                input,
                vjp_inputs=["G"],
                vjp_outputs=["x"],
                cotangent_vector={"x": tangent_vector},
            )
        except Exception as e:
            print(f"Error in vector_jacobian_product: {e}")
            break
        vjp_G = output["G"]
        # update G
        G = G - lrs[k] * vjp_G
        input["G"] = G


print("Final constraint:", constraints[-1])
print("Final loss:", losses[-1])
print("Final x*:", sols[-1])
writergif = animation.PillowWriter(fps=30)
anim = make_animation(constraints, sols, x_true, Q, q)
this_dir = os.path.dirname(os.path.abspath(__file__))
anim.save(f"{this_dir}/plots/qp_solve_animation.gif", writer=writergif)
