from tesseract_core import Tesseract

import numpy as np

print(type(np.float32(1)))

print(isinstance(np.float32(1), np.floating))

tesseract = Tesseract(url="http://localhost:8000")

# Test the function
out = tesseract.apply({"example": 1.0})


jac = tesseract.jacobian({"example": 1.0}, jac_inputs=("example",), jac_outputs=("example",))

vjp = tesseract.vector_jacobian_product(
    {"example": 1.0},
    vjp_inputs={"example"},
    vjp_outputs={"example"},
    cotangent_vector={"example": 1.0},
)

jvp = tesseract.jacobian_vector_product(
    {"example": 1.0},
    jvp_inputs={"example"},
    jvp_outputs={"example"},
    tangent_vector={"example": 1.0},
)

print(out)

