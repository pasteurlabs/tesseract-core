from typing import Optional

import juliacall
import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

# Initialize julia environment

jl = juliacall.newmodule("julia_surrogate")
jl.seval("using Pkg")
jl.seval('Pkg.activate("StressSurrogate")')
jl.seval("using StressSurrogate")

# warm up julia functions
jl.StressSurrogate.eval_forward(np.array([0.5, 0.5, 0.15, 45.0]))
jl.StressSurrogate.eval_gradient(np.array([0.5, 0.5, 0.15, 45.0]))

#
# Schemata
#


class InputSchema(BaseModel):
    xc: Differentiable[Float64] = Field(
        description="Ellipse center x coordinate.", default=0.5
    )
    yc: Differentiable[Float64] = Field(
        description="Ellipse center y coordinate.", default=0.5
    )
    axis_x: Differentiable[Float64] = Field(
        description="Axis skew in x direction.", default=0.15
    )
    theta: Differentiable[Float64] = Field(
        description="Ellipse angle with origin [degrees].", default=45.0
    )
    return_force_components: bool = Field(
        description="Whether to return force components for apply.",
        default=False,
    )
    return_stress_components: bool = Field(
        description="Whether to return von-mises stress for apply.",
        default=False,
    )


class OutputSchema(BaseModel):
    mean_stress: Differentiable[Float64] = Field(
        description="The maximum stress along the (x=1,y=1) boundaries."
    )
    fx: Optional[Array[(2601,), Float64]] = Field(description="Force x component.")
    fy: Optional[Array[(2601,), Float64]] = Field(description="Force y component.")
    s: Optional[Array[(2601,), Float64]] = Field(description="Von-mises stress.")


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    x = np.array([inputs.xc, inputs.yc, inputs.axis_x, inputs.theta])
    load_field = jl.StressSurrogate.generate_field(x)
    stress_field = jl.StressSurrogate.eval_surrogate(load_field)
    mean_stress = jl.StressSurrogate.calc_mean_stress_from_field(stress_field)

    if inputs.return_force_components:
        load_field_return = np.reshape(load_field, (2601, 2), order="F")
        fx = load_field_return[:, 0]
        fy = load_field_return[:, 1]
    else:
        fx = None
        fy = None

    if inputs.return_stress_components:
        s = stress_field
    else:
        s = None

    return OutputSchema(mean_stress=mean_stress, fx=fx, fy=fy, s=s)


def abstract_eval(abstract_inputs):
    return {"mean_stress": ShapeDType(shape=(), dtype="float64")}


#
# Optional endpoints
#


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    assert set(jac_inputs) == set(["xc", "yc", "axis_x", "theta"])
    x = np.array([inputs.xc, inputs.yc, inputs.axis_x, inputs.theta])
    grad = jl.StressSurrogate.eval_gradient(x)
    return {
        "xc": {"mean_stress": grad[0]},
        "yc": {"mean_stress": grad[1]},
        "axis_x": {"mean_stress": grad[2]},
        "theta": {"mean_stress": grad[3]},
    }
