import numpy as np

from tesseract_core import Tesseract

tesseract = Tesseract(url="http://localhost:8000")

# Test the function
# out = tesseract.apply({"b": 2.0, "example": 1.0})

inputs = {
    "a": {"v": np.array([1.0, 2.0, 3.0])},
    "b": {"v": np.array([4.0, 5.0, 6.0]), "s": 3.0},
}

jvp_inputs = ["a.v"]

jvp_outputs = ["vector_add.result", "vector_min.normed_result"]

tangent_vector = {"a.v": np.array([0.0, 5.0, 2.0])}

cotangent_vector = {
    "vector_min.normed_result": np.array([0.1, 0.2, 0.3]),
    "vector_add.result": np.array([0.4, 0.5, 0.6]),
}
out = tesseract.apply(inputs)

print(out)


# jac = tesseract.jacobian({"b" : 2.0, "example": 1.0}, jac_inputs=("example", "b"), jac_outputs=("example", ))

# print(jac)

# vjp = tesseract.vector_jacobian_product(
#     {"b" : 2.0, "example": 1.0},
#     vjp_inputs=["example"],
#     vjp_outputs=["example"],
#     cotangent_vector={"example": 1.0},
# )

jvp = tesseract.jacobian_vector_product(
    inputs,
    jvp_inputs=jvp_inputs,
    jvp_outputs=jvp_outputs,
    tangent_vector=tangent_vector,
)

vjp = tesseract.vector_jacobian_product(
    inputs=inputs,
    vjp_inputs=jvp_inputs,
    vjp_outputs=jvp_outputs,
    cotangent_vector=cotangent_vector,
)

print(out)
