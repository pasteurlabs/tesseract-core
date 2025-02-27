(tr-autodiff)=
# Differentiable Programming basics

[Differentiable Programming](https://en.wikipedia.org/wiki/Differentiable_programming) (DP) is a technique to compute the derivative of a (software) function with respect to its inputs. It is a key ingredient in many optimization algorithms, such as gradient descent, and is widely used in machine learning and scientific computing. Automatic differentiation (autodiff or AD) is a technique to compute the derivative of a function automatically, without the need to manually derive and implement the derivative.

Tesseracts natively support DP and autodiff as an optional feature – as long as at least one of the input or output arrays is marked as differentiable, and an [AD endpoint](#ad-endpoints) is implemented, the Tesseract can be differentiated with respect to its inputs.

## Autodiff flavors

There are several ways to compute the derivative of a pipeline of functions with respect to their inputs, and each has its own trade-offs. Tesseracts support both forward-mode and reverse-mode autodiff, as well as the computation of the full Jacobian and Hessian matrices.

```{figure} /img/autodiff-flavors.png
:alt: Autodiff flavors
:width: 500px

Autodiff flavors in a nutshell. From [Sapienza et al, 2024](https://doi.org/10.48550/arXiv.2406.09699).
```

```{seealso}
For a rigorous introduction to the current state of the art in AD methods, see [Sapienza et al, 2024](https://doi.org/10.48550/arXiv.2406.09699).
```

(ad-endpoints)=
## Tesseract AD endpoints

```{note}
Tesseracts are free to compute the derivative of their output with respect to their inputs in any way they see fit. The endpoints described here are just what is exposed to the outside world.
```

### Jacobian

The `jacobian` endpoint computes the Jacobian matrix $J$ of one or several outputs of the wrapped function $f$ with respect to one or several input variables $x$, at a point $X$.

$$ J_{ij} = \frac{\partial f_i}{\partial x_j} \bigg|_X $$

The Jacobian matrix has one additional axis compared to the input arrays. It is often used in optimization algorithms that require the gradient of the objective function.

```{warning}
Computing the Jacobian can be computationally expensive, especially when the output is a high-dimensional vector. In such cases, it may be more efficient to use Jacobian-vector products (JVPs) or vector-Jacobian products (VJPs) instead (see below).
```

In Tesseracts, both inputs and outputs can be arbitrarily nested tree-like objects, such as dicts of lists of arrays. The Jacobian endpoint supports computing the derivative of the entire output tree with respect to one or several input leaves.

#### Example usage

```python
# Assume a nested input schema with two arrays
>>> inputs = {
...     "x": {
...         "x1": np.array([1.0, 2.0]),
...         "x2": np.array([3.0, 4.0]),
...     }
... }

# Differentiate the output `y` with respect to both input arrays
>>> jacobian(inputs, jac_inputs=["x.x1", "x.x2"], jac_outputs=["y"])
{
    "y": {
        "x.x1": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "x.x2": np.array([[0.0, 0.0], [0.0, 0.0]])
    }
}
```

For more information, see the API reference for the {py:func}`Jacobian endpoint <tesseract_core.runtime.app_cli.jacobian>`.

### Jacobian-vector product (JVP) and vector-Jacobian product (VJP)

Jacobian-vector products (JVPs) and vector-Jacobian products (VJPs) are more efficient ways to compute the derivative of a function with respect to its inputs, especially when the input or output is a high-dimensional vector.

Instead of computing the full Jacobian matrix, JVPs and VJPs compute the product of the Jacobian matrix with a given vector when only the product of the Jacobian with a vector is needed. This is the case in classical forward-mode (JVP) and reverse-mode (VJP) AD.

In contrast to Jacobian, the JVP and VJP endpoints also require a tangent / cotangent vector to be passed as an additional input, which is multiplied with the Jacobian matrix before returning.

```{seealso}
For a practical introduction to JVPs and VJPs, see [the JAX documentation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#how-it-s-made-two-foundational-autodiff-functions).
```

#### Example usage

```python
# Assume a nested input schema with two arrays
>>> inputs = {
...     "x": {
...         "x1": np.array([1.0, 2.0]),
...         "x2": np.array([3.0, 4.0]),
...     }
... }

# Differentiate the output `y` with respect to both input arrays

# Tangent vector is a dict with keys given by jac_inputs
>>> jacobian_vector_product(inputs, jac_inputs=["x.x1", "x.x2"], jac_outputs=["y"], tangent_vector={"x.x1": np.array([1.0, 2.0]), "x.x2": np.array([3.0, 4.0])})
{
    "y": np.array([1.0, 2.0])
}

# Cotangent vector is a dict with keys given by jac_outputs
>>> vector_jacobian_product(inputs,jac_inputs=["x.x1", "x.x2"], jac_outputs=["y"], cotangent_vector={"y": np.array([1.0, 0.0])})
{
    "x.x1": np.array([1.0, 0.0]),
    "x.x2": np.array([0.0, 0.0]),
}
```

For more information, see the API reference for the {py:func}`Jacobian-vector product endpoint <tesseract_core.runtime.app_cli.jacobian_vector_product>` and the {py:func}`Vector-Jacobian product endpoint <tesseract_core.runtime.app_cli.vector_jacobian_product>`.

### Abstract Evaluation

In some scenarios it can be useful to know what the shapes of arrays in the output of a Tesseract will be,
without actually running the computation implemented in a Tesseract's `apply`. This can be particularly
important when performing automatic differentiation for efficient memory allocation and optimization
of computation graphs.

In order to do this, the {py:func}`abstract_eval <tesseract_core.runtime.app_cli.abstract_eval>` endpoint
can be implemented. This endpoint accepts the same inputs as the `apply` endpoint, except that
array arguments are replaced by their shape and dtype (see {py:class}`ShapeDType <tesseract_core.runtime.ShapeDType>`).
This makes it possible to infer output shapes from input shapes (and non-array arguments) before their actual data is known.


#### Example usage

```python
# No actual values are specified for "a" and "b"; only that they are 3D vectors
>>> inputs = {
...     "a": {"dtype": "float64", "shape": [3]},
...     "b": {"dtype": "float64", "shape": [3]},
...     "s": 1.0,
...     "normalize": False,
... }

# This tells us that the result is a 3D vector as well.
>>> abstract_eval(inputs)
{'result': {'dtype': 'float64', 'shape': [3]}}
```
