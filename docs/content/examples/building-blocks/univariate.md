# Univariate Rosenbrock function

## Context

Example that wraps the univariate [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function), which is a common test problem for optimization algorithms. Defines a Tesseract that has all optional endpoints implemented, including `apply`, `abstract_eval`, `jacobian`, `jacobian_vector_product`, and `vector_jacobian_product`.

```{seealso}
This example (and using it to perform optimization) is also part of an [Expert Showcase](https://si-tesseract.discourse.group/t/jax-based-rosenbrock-function-minimization/48) in the Tesseract Community Forum.
```

## Example Tesseract (examples/univariate)

### Core functionality --- schemas and `apply` function

This example uses a pure-Python implementation of the Rosenbrock function as the basis for all endpoints:

```{literalinclude} ../../../../examples/univariate/tesseract_api.py
:pyobject: rosenbrock
:language: python
```

As such, the Tesseract has 2 differentiable scalar inputs (`x` and `y`) and a single output (the value of the Rosenbrock function at those inputs). The parameters `a` and `b` are treated as non-differentiable constants.

```{literalinclude} ../../../../examples/univariate/tesseract_api.py
:pyobject: InputSchema
:language: python
```

```{literalinclude} ../../../../examples/univariate/tesseract_api.py
:pyobject: OutputSchema
:language: python
```

This makes it straightforward to write the `apply` function, which simply unpacks the inputs and calls the `rosenbrock` function with them:

```{literalinclude} ../../../../examples/univariate/tesseract_api.py
:pyobject: apply
:language: python
```

### Jacobian endpoint

For the Jacobian, we exploit the fact that the `rosenbrock` function is traceable by [JAX](https://github.com/jax-ml/jax). We can therefore use [`jax.jacrev`](https://docs.jax.dev/en/latest/_autosummary/jax.jacrev.html) to compute the Jacobian of the function with respect to its inputs:

```{literalinclude} ../../../../examples/univariate/tesseract_api.py
:pyobject: jacobian
:language: python
```

### Other AD endpoints

We define the JVP (Jacobian-vector product) and VJP (vector-Jacobian product) endpoints by summing over rows / columns of the Jacobian matrix. That is, we call `jacobian` under the hood, then multiply the resulting Jacobian matrix by the (tangent / cotangent) vector input.

```{warning}
Defining JVP and VJP operations through sums over the full Jacobian matrix is inefficient and negates the benefits of using JVP / VJP. These endpoints are provided for completeness, but in practice, you would typically use JAX's built-in JVP and VJP functions directly.
```

```{literalinclude} ../../../../examples/univariate/tesseract_api.py
:pyobject: jacobian_vector_product
:language: python
```

```{literalinclude} ../../../../examples/univariate/tesseract_api.py
:pyobject: vector_jacobian_product
:language: python
```

### Abstract evaluation

Some Tesseract clients (like [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax)) require an abstract evaluation endpoint in order to pre-allocate memory for the inputs and outputs. This is a simple function that returns the shapes of the outputs based on the shapes of the inputs. In this case, the output is always a scalar, so we return an empty shape tuple:

```{literalinclude} ../../../../examples/univariate/tesseract_api.py
:pyobject: abstract_eval
:language: python
```
