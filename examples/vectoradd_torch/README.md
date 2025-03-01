# Vectoradd (with jax)

This tesseract is similar to `vectoradd`, but returns a nested dictionary
with an entry for the vectors added and one for the vectors added.
These both contain result and normed_result as outputs.
The tesseract uses jax behind the scenes. Also,
it exposes the `vector_jacobian_product` and `jacobian_vector_product` endpoints,
which likely are what we want to use in practice.
