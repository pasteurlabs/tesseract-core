import numpy as np

from tesseract_core import Tesseract

print(type(np.float32(1)))

print(isinstance(np.float32(1), np.floating))

tesseract = Tesseract(url="http://localhost:8000")

# Test the function
out = tesseract.apply({"b": 2.0, "example": 1.0})


# jac = tesseract.jacobian({"b" : 2.0, "example": 1.0}, jac_inputs=("example", "b"), jac_outputs=("example", ))

# print(jac)

# vjp = tesseract.vector_jacobian_product(
#     {"b" : 2.0, "example": 1.0},
#     vjp_inputs=["example"],
#     vjp_outputs=["example"],
#     cotangent_vector={"example": 1.0},
# )

jvp = tesseract.jacobian_vector_product(
    {"b": 2.0, "example": 1.0},
    jvp_inputs=["example"],
    jvp_outputs=["example"],
    tangent_vector={"example": 1.0},
)

print(out)
