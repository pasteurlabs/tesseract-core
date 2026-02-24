# Finite Difference Gradients

This example demonstrates how to make any Tesseract differentiable using **finite differences**,
without implementing analytical gradient code.

```{warning}
This feature is **experimental** and available in `tesseract_core.runtime.experimental`.
The API may change in future releases.
```

## When to use finite differences

Finite differences are useful when:

- **Prototyping**: You want to quickly test gradient-based workflows before investing time in analytical derivatives
- **Complex nested schemas**: Your inputs have deeply nested structures (like the mesh data in this example) that would be tedious to differentiate manually
- **Legacy code**: You're wrapping existing code where deriving gradients would require significant refactoring
- **Verification**: You want to cross-check your analytical gradient implementation

## Trade-offs

| Approach               | Pros                                | Cons                                     |
| ---------------------- | ----------------------------------- | ---------------------------------------- |
| **Analytical**         | Accurate, efficient                 | Requires manual implementation           |
| **Finite Differences** | No manual work, works with any code | Less accurate, more function evaluations |

## The example: meshstats_finitediff

This example is a variant of the `meshstats` example that computes statistics on volumetric mesh data.
Instead of hand-written Jacobian implementations, it uses the finite difference helpers.

### Input schema

The input is a complex nested structure representing a volumetric mesh:

```python
class VolumetricMeshData(BaseModel):
    n_points: int
    n_cells: int
    points: Differentiable[Array[(None, 3), Float64]]
    num_points_per_cell: Array[(None,), Float64]
    cell_connectivity: Array[(None,), Int32]
    cell_data: dict[str, Array[(None, None), Float64]]
    point_data: dict[str, Array[(None, None), Float64]]
```

### Implementing gradients with finite differences

The key insight is that you can implement all three AD endpoints (`jacobian`, `jacobian_vector_product`,
`vector_jacobian_product`) with just a few lines of code:

```python
from tesseract_core.runtime.experimental import (
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
)

def jacobian(inputs, jac_inputs, jac_outputs):
    return finite_difference_jacobian(
        apply, inputs, jac_inputs, jac_outputs,
        algorithm="central",
        eps=1e-6,
    )

def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    return finite_difference_jvp(
        apply, inputs, jvp_inputs, jvp_outputs, tangent_vector,
        algorithm="central",
        eps=1e-6,
    )

def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return finite_difference_vjp(
        apply, inputs, vjp_inputs, vjp_outputs, cotangent_vector,
        algorithm="central",
        eps=1e-6,
    )
```

### Choosing an algorithm

The `algorithm` parameter controls the finite difference method:

| Algorithm      | Function Evaluations | Accuracy | Use case                                    |
| -------------- | -------------------- | -------- | ------------------------------------------- |
| `"central"`    | 2n                   | Highest  | Default choice                              |
| `"forward"`    | n + 1                | Medium   | When evaluations are expensive              |
| `"stochastic"` | O(√n)                | Lower    | High-dimensional inputs (1000s+ parameters) |

where `n` is the total number of input elements being differentiated.

### Making the algorithm configurable

The example also shows how to expose the algorithm choice as an input parameter:

```python
from typing import Literal

FDAlgorithm = Literal["central", "forward", "stochastic"]

class InputSchema(BaseModel):
    mesh: VolumetricMeshData
    fd_algorithm: FDAlgorithm = "central"

def jacobian(inputs, jac_inputs, jac_outputs):
    return finite_difference_jacobian(
        apply, inputs, jac_inputs, jac_outputs,
        algorithm=inputs.fd_algorithm,  # User can choose at runtime
        eps=1e-6,
    )
```

## Full source code

```{literalinclude} ../../../../examples/meshstats_finitediff/tesseract_api.py
:language: python
:caption: tesseract_api.py
```

```{literalinclude} ../../../../examples/meshstats_finitediff/tesseract_config.yaml
:language: yaml
:caption: tesseract_config.yaml
```

## See also

- {ref}`tr-autodiff` for background on differentiable programming in Tesseracts
- The [meshstats example](https://github.com/pasteurlabs/tesseract-core/tree/main/examples/meshstats) for the version with analytical gradients
