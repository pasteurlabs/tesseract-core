(tr-autodiff)=

# Differentiable Programming Basics

[Differentiable Programming](https://en.wikipedia.org/wiki/Differentiable_programming) (DP) is a technique to compute the derivative of a (software) function with respect to its inputs. It is a key ingredient in many optimization algorithms, such as gradient descent, and is widely used in machine learning and scientific computing. Automatic differentiation (autodiff or AD) is a technique to compute the derivative of a function automatically, without the need to manually derive and implement the derivative.

Tesseracts natively support DP and autodiff as an optional feature – as long as at least one of the input or output arrays is marked as differentiable, and a [gradient endpoint](#gradient-endpoints) is implemented, the Tesseract can be differentiated with respect to its inputs.

```{tip}
If you're using JAX, check out [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) — a companion package that makes it easy to create Tesseracts from JAX functions with autodiff endpoints generated automatically.
```

## Autodiff flavors

There are several ways to compute the derivative of a pipeline of functions with respect to their inputs, and each has its own trade-offs. Tesseracts support both forward-mode and reverse-mode autodiff, as well as the computation of the full Jacobian matrix.

```{figure} /img/autodiff-flavors.png
:alt: Autodiff flavors
:width: 500px

Autodiff flavors in a nutshell. From [Sapienza et al, 2024](https://doi.org/10.48550/arXiv.2406.09699).
```

```{seealso}
For a rigorous introduction to the current state of the art in AD methods, see [Sapienza et al, 2024](https://doi.org/10.48550/arXiv.2406.09699).
```

(gradient-endpoints)=

(ad-endpoints)=

## Tesseract gradient endpoints

```{note}
Tesseracts are free to compute the derivative of their output with respect to their inputs in any way they see fit. The endpoints described here are just what is exposed to the outside world.
```

### Jacobian

The `jacobian` endpoint computes the Jacobian matrix $J$ of one or several outputs of the wrapped function $f$ with respect to one or several input variables $x$, at a point $X$.

$$ J\_{ij} = \frac{\partial f_i}{\partial x_j} \bigg|\_X $$

The Jacobian matrix has one additional axis compared to the input arrays. It is often used in optimization algorithms that require the gradient of the objective function.

```{warning}
Computing the Jacobian can be computationally expensive, especially when the output is a high-dimensional vector. In such cases, it may be more efficient to use Jacobian-vector products (JVPs) or vector-Jacobian products (VJPs) instead (see below).
```

In Tesseracts, both inputs and outputs can be arbitrarily nested tree-like objects, such as dicts of lists of arrays. The Jacobian endpoint supports computing the derivative of the entire output tree with respect to one or several input leaves.

#### Path notation for nested inputs and outputs

The `jac_inputs` and `jac_outputs` arguments (and analogously `jvp_inputs`, `vjp_inputs`, etc.) use dot-separated path strings to refer to individual differentiable arrays within nested structures. The path syntax supports:

| Structure   | Syntax  | Example              |
| ----------- | ------- | -------------------- |
| Model field | `field` | `"x"`, `"params.z"`  |
| Dict key    | `{key}` | `"params.{my_key}"`  |
| List index  | `[i]`   | `"coefficients.[0]"` |

These can be combined freely: `"beta.gamma.{u}"` refers to the `"u"` key of the `gamma` dict inside the `beta` sub-model. `"delta.[0]"` refers to the first element of the `delta` list.

```{note}
Simple dict keys that are valid Python identifiers can also be written without braces (e.g., `"params.x"` instead of `"params.{x}"`). Use braces for keys that contain spaces or special characters (e.g., `"params.{my key}"`).
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
>>> jacobian(inputs, jac_inputs=["x.x1", "x.x2"], jac_outputs=["y"])
{
    "y": {
        "x.x1": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "x.x2": np.array([[0.0, 0.0], [0.0, 0.0]])
    }
}
```

#### Example with dicts and lists

```python
# Schema:
#   InputSchema:
#     alpha: dict[str, Differentiable[Array[(None,), Float32]]]   (keys: "x", "y")
#     delta: list[Differentiable[Array[(None,), Float32]]]        (2 elements)
#   OutputSchema:
#     result: Differentiable[Array[(3,), Float32]]
#     result_dict: dict[str, Differentiable[Array[(None,), Float32]]]  (keys: "a", "b")
#     result_list: list[Differentiable[Array[(None,), Float32]]]       (2 elements)

>>> jacobian(
...     inputs,
...     jac_inputs=["alpha.x", "delta.[0]"],
...     jac_outputs=["result", "result_dict.a", "result_list.[1]"],
... )
{
    "result": {
        "alpha.x": np.array([[...], ...]),       # shape (3, len(x))
        "delta.[0]": np.array([[...], ...]),      # shape (3, len(delta[0]))
    },
    "result_dict.a": {
        "alpha.x": np.array([[...], ...]),       # shape (len(a), len(x))
        "delta.[0]": np.array([[...], ...]),      # shape (len(a), len(delta[0]))
    },
    "result_list.[1]": {
        "alpha.x": np.array([[...], ...]),       # shape (len(list[1]), len(x))
        "delta.[0]": np.array([[...], ...]),      # shape (len(list[1]), len(delta[0]))
    },
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

# Tangent vector is a dict with keys given by jvp_inputs
>>> jacobian_vector_product(inputs, jvp_inputs=["x.x1", "x.x2"], jvp_outputs=["y"], tangent_vector={"x.x1": np.array([1.0, 2.0]), "x.x2": np.array([3.0, 4.0])})
{
    "y": np.array([1.0, 2.0])
}

# Cotangent vector is a dict with keys given by vjp_outputs
>>> vector_jacobian_product(inputs, vjp_inputs=["x.x1", "x.x2"], vjp_outputs=["y"], cotangent_vector={"y": np.array([1.0, 0.0])})
{
    "x.x1": np.array([1.0, 0.0]),
    "x.x2": np.array([0.0, 0.0]),
}
```

#### Tangent and cotangent vectors with dicts and lists

When inputs or outputs contain dicts or lists, the tangent and cotangent vectors use the same path notation as `jvp_inputs`/`vjp_outputs`:

```python
# Schema:
#   alpha: dict[str, Differentiable[Array[...]]]   (keys: "x", "y")
#   delta: list[Differentiable[Array[...]]]         (2 elements)
#   result_dict: dict[str, Differentiable[Array[...]]]  (keys: "a", "b")
#   result_list: list[Differentiable[Array[...]]]       (2 elements)

# JVP: tangent_vector keys must match jvp_inputs
>>> jacobian_vector_product(
...     inputs,
...     jvp_inputs=["alpha.x", "delta.[0]"],
...     jvp_outputs=["result_dict.a", "result_list.[1]"],
...     tangent_vector={
...         "alpha.x": np.array([1.0, 0.0, 0.0]),   # same shape as alpha["x"]
...         "delta.[0]": np.array([0.0, 1.0, ...]),  # same shape as delta[0]
...     },
... )
{
    "result_dict.a": np.array([...]),   # same shape as result_dict["a"]
    "result_list.[1]": np.array([...]), # same shape as result_list[1]
}

# VJP: cotangent_vector keys must match vjp_outputs
>>> vector_jacobian_product(
...     inputs,
...     vjp_inputs=["alpha.x", "delta.[0]"],
...     vjp_outputs=["result_dict.a", "result_list.[1]"],
...     cotangent_vector={
...         "result_dict.a": np.array([1.0, 0.0, 0.0]),  # same shape as result_dict["a"]
...         "result_list.[1]": np.array([0.0, 1.0, ...]), # same shape as result_list[1]
...     },
... )
{
    "alpha.x": np.array([...]),   # same shape as alpha["x"]
    "delta.[0]": np.array([...]), # same shape as delta[0]
}
```

```{important}
- **Tangent vectors** have keys matching `jvp_inputs` (input paths), and each value has the same shape as the corresponding input array.
- **Cotangent vectors** have keys matching `vjp_outputs` (output paths), and each value has the same shape as the corresponding output array.
- The JVP **result** has keys matching `jvp_outputs`, with each value having the same shape as the corresponding output.
- The VJP **result** has keys matching `vjp_inputs`, with each value having the same shape as the corresponding input.
```

For more information, see the API reference for the {py:func}`Jacobian-vector product endpoint <tesseract_core.runtime.app_cli.jacobian_vector_product>` and the {py:func}`Vector-Jacobian product endpoint <tesseract_core.runtime.app_cli.vector_jacobian_product>`.

### Gradient Endpoint Derivation Fallbacks (Experimental)

If you have already implemented one gradient endpoint but need the others, you can derive
them automatically using the experimental fallback helpers:

- Derive **JVP** or **VJP** from an existing `jacobian` with `jvp_from_jacobian` / `vjp_from_jacobian`
- Derive the full **Jacobian** from an existing JVP or VJP with `jacobian_from_jvp` / `jacobian_from_vjp`

```{seealso}
For a practical guide with full examples and cost trade-offs, see
{doc}`/content/examples/building-blocks/gradient-fallbacks`.
```

### Finite Difference Gradients (Experimental)

If implementing analytical gradients is too complex or time-consuming, you can use **finite differences**
to approximate gradients numerically. This is useful for prototyping, complex nested schemas, and
verification of analytical gradient implementations.

```{warning}
Numerical differentiation is less accurate and more
computationally expensive than analytical methods or automatic differentiation. Use with caution, especially for high-dimensional inputs.
```

```{seealso}
For a full guide on finite difference algorithms and a complete example, see {doc}`/content/examples/building-blocks/finitediff`.
```

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
