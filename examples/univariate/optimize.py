import matplotlib.pyplot as plt
import numpy as np
from animate import make_animation
from scipy.optimize import minimize

from tesseract_core import Tesseract

# Instantiate python api to the tesseract (read port from logs above)
tesseract = Tesseract(url="http://localhost:58354")


def rosenbrock(x: np.ndarray) -> float:
    """Wrap tesseract.apply to adhere to scipy's minimize interface."""
    output = tesseract.apply({"x": x[0], "y": x[1]})
    return output["result"].item()


def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    """Wrap tesseract.jacobian to adhere to scipy's minimize interface."""
    output = tesseract.jacobian(
        {"x": x[0], "y": x[1]}, jac_inputs=["x", "y"], jac_outputs=["result"]
    )
    return np.array([output["result"]["x"], output["result"]["y"]])


# Initial guess for the variables
x0 = np.array([-0.5, 1.0])
trajectory = [x0]

# kick off scipy's optimization using tesseract apply/jacobian
result = minimize(
    rosenbrock,
    x0,
    method="BFGS",
    jac=rosenbrock_gradient,
    options={"disp": True},
    callback=lambda xs: trajectory.append(xs.tolist()),
)

anim = make_animation(*list(zip(*trajectory)))
# anim.save("rosenbrock_optimization.gif", writer="pillow", fps=2, dpi=150)
plt.show()
