from tesseract_core import Tesseract

tesseract = Tesseract(url="http://localhost:8000")

inputs = {
    "a": {"v": [1.0, 2.0, 3.0]},
    "b": {"v": [4.0, 5.0, 6.0], "s": 3.0},
}

# with open("example_inputs.json", "w") as f:
#     json.dump({"inputs": inputs}, f, indent=4)


jvp_inputs = ["a.v"]

jvp_outputs = ["vector_add.result", "vector_min.normed_result"]

tangent_vector = {"a.v": [0.0, 5.0, 2.0]}

cotangent_vector = {
    "vector_min.normed_result": [0.4, 0.5, 0.6],
    "vector_add.result": [0.1, 0.2, 0.3],
}
out = tesseract.apply(inputs)


jac = tesseract.jacobian(
    inputs=inputs,
    jac_inputs=jvp_inputs,
    jac_outputs=jvp_outputs,
)

print(jac)

jvp_inputs = ["a.v", "a.s"]
jvp_outputs = ["vector_add.result", "vector_min.normed_result"]
tangent_vector = {"a.v": [0.0, 5.0, 2.0], "a.s": 1.0}
jvp = tesseract.jacobian_vector_product(
    inputs,
    jvp_inputs=jvp_inputs,
    jvp_outputs=jvp_outputs,
    tangent_vector=tangent_vector,
)

print(jvp)

vjp = tesseract.vector_jacobian_product(
    inputs=inputs,
    vjp_inputs=jvp_inputs,
    vjp_outputs=jvp_outputs,
    cotangent_vector=cotangent_vector,
)

print(vjp)
