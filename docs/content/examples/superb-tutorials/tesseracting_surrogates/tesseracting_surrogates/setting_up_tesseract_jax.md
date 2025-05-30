# Setting up the Tesseract Files for the JAX Surrogate

The procedure for generating the JAX tesseract is the same as used in the Julia pipeline. The `tesseract_api.py` function shows slight differences with respect to the one described above, these are mainly related to the functionality of the surrogate model and how the surrogate model is loaded into the environment.
We recall that the JAX surrogate provides a mapping from the ellipse parameters to the mean displacement magnitude on the free boundaries. We report it below.

<details>
  <summary><b>Click to expand the code</b></summary>

```Python
# Tesseract API module
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional
from package.mgn_surrogate import (
    jit_generate_field,
    jit_eval_surrogate,
    jit_calc_mean_displacement_from_field,
    jit_calc_mean_displacement_from_input,
    jit_grad_mean_displacement_from_input
)
from tesseract_runtime import Differentiable, Float64, ShapeDType, Array

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
    return_displacement_components: bool = Field(
        description="Whether to return displacement components for apply.",
        default=False,
    )

class OutputSchema(BaseModel):
    mean_displacement: Differentiable[Float64] = Field(
        description="The average displacement over (x=1, y=1) boundaries."
    )
    fx: Optional[Array[(2601,), " float32"]] = Field(description="Force x component.")
    fy: Optional[Array[(2601,), " float32"]] = Field(description="Force y component.")
    ux: Optional[Array[(2601,), " float32"]] = Field(description="Displacement x component.")
    uy: Optional[Array[(2601,), " float32"]] = Field(description="Displacement y component.")


#
# Required endpoints
#

def apply(inputs: InputSchema) -> OutputSchema:
    x = np.array([inputs.xc, inputs.yc, inputs.axis_x, inputs.theta])
    load_field = jit_generate_field(x)
    displacement_field = jit_eval_surrogate(load_field)
    mean_displacement = jit_calc_mean_displacement_from_field(displacement_field)

    if inputs.return_force_components:
        fx = load_field.nodes[:,0]
        fy = load_field.nodes[:,1]
    else:
        fx = None
        fy = None

    if inputs.return_displacement_components:
        ux = displacement_field.nodes[:,0]
        uy = displacement_field.nodes[:,1]
    else:
        ux = None
        uy = None

    return OutputSchema(
        mean_displacement=mean_displacement,
        fx=fx,
        fy=fy,
        ux=ux,
        uy=uy
    )

#
# Optional endpoints
#
def abstract_eval(abstract_inputs):
    return {"mean_displacement": ShapeDType(shape=(), dtype="float64")}

def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    assert set(jac_inputs) == set(["xc", "yc", "axis_x", "theta"])
    x = np.array([inputs.xc, inputs.yc, inputs.axis_x, inputs.theta])
    grad = np.array(jit_grad_mean_displacement_from_input(x))
    return {
        "xc": {"mean_displacement": grad[0]},
        "yc": {"mean_displacement": grad[1]},
        "axis_x": {"mean_displacement": grad[2]},
        "theta": {"mean_displacement": grad[3]},
    }
```
</details>

The JAX surrogate model with its associated functionalities are imported from the `mgn_surrogate` module within the `package` directory. The `InputSchema` now has an additional boolean parameter `return_displacement_components` to optionally return the resulting displacement field and, accordingly, the `OutputSchema` has specifications for the optional outputs `ux` and `uy`. The `apply` function returns `mean_displacement` at the free boundaries in addition to the optional output of `fx`, `fy`, `ux` and `uy`. The `jacobian` function now provides gradient of the mean displacement at free boundarieswith respect to the ellipse parameters.

The `tesseract_config.yaml` file in this case is quite simpler; it is reported below.

```yaml
name: "supersede_mgn_ellipse_displacement"
version: "1.0.0"

build_config:
  package_data:
    - [package, package]
```

Since the JAX surrogate model is built on top of the `Jraph` library, the corresponding dependencies are specified in the `tesseract_requirements.txt` file as follows.

```
jax[cpu]==0.4.30
jraph==0.0.6.dev0
numpy==2.0.0
absl-py==2.1.0
flatbuffers==24.3.25

```
