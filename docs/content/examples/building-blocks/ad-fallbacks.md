# AD Endpoint Derivation Fallbacks

This guide shows how to derive missing autodiff endpoints from ones you have already
implemented, using the experimental AD fallback helpers.

```{warning}
This feature is **experimental** and available in `tesseract_core.runtime.experimental`.
The API may change in future releases. All four helpers materialise the **full Jacobian
matrix** at some point, which can be expensive for high-dimensional inputs or outputs.
Use them when the derived endpoint is not on the hot path of your workflow.
```

## Overview

Tesseracts expose up to three AD endpoints — `jacobian`, `jacobian_vector_product`
(JVP), and `vector_jacobian_product` (VJP). The helpers `jvp_from_jacobian`,
`vjp_from_jacobian`, `jacobian_from_jvp`, and `jacobian_from_vjp` let you derive any
one of these endpoints from another you have already implemented, without writing
additional gradient code.

## Deriving JVP and VJP from the Jacobian

If you have a `jacobian` implementation (e.g. produced by an AD framework), you can
derive JVP and VJP from it:

```python
from tesseract_core.runtime.experimental import jvp_from_jacobian, vjp_from_jacobian

def jacobian(inputs, jac_inputs, jac_outputs):
    # Your existing Jacobian implementation
    ...

def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    return jvp_from_jacobian(
        jacobian, inputs, jvp_inputs, jvp_outputs, tangent_vector
    )

def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return vjp_from_jacobian(
        jacobian, inputs, vjp_inputs, vjp_outputs, cotangent_vector
    )
```

## Deriving the Jacobian from JVP or VJP

If you have a JVP or VJP but no explicit Jacobian, you can materialise the full
Jacobian matrix from either:

- **`jacobian_from_jvp`** — sweeps one-hot tangent vectors over each _input_ element.
  Costs **N** JVP calls (N = total input elements). Prefer this when outputs are
  high-dimensional.
- **`jacobian_from_vjp`** — sweeps one-hot cotangent vectors over each _output_ element.
  Costs **M** VJP calls (M = total output elements). Prefer this when inputs are
  high-dimensional.

### From JVP

```python
from tesseract_core.runtime.experimental import jacobian_from_jvp

def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    # Your existing JVP implementation
    ...

def jacobian(inputs, jac_inputs, jac_outputs):
    return jacobian_from_jvp(
        jacobian_vector_product, inputs, jac_inputs, jac_outputs
    )
```

### From VJP

`jacobian_from_vjp` needs to know the output shapes before probing, so it takes an
`eval_fn` argument — either `apply` or `abstract_eval`. `abstract_eval` is preferred
because it determines shapes without running the full forward computation.

```python
from tesseract_core.runtime.experimental import jacobian_from_vjp

def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    # Your existing VJP implementation
    ...

def abstract_eval(inputs):
    # Your existing abstract_eval implementation (preferred)
    ...

def jacobian(inputs, jac_inputs, jac_outputs):
    return jacobian_from_vjp(
        vector_jacobian_product, abstract_eval, inputs, jac_inputs, jac_outputs
    )
```

## Full source code

The complete example is the `univariate_adfallbacks` Tesseract — a variant of
[univariate](https://github.com/pasteurlabs/tesseract-core/tree/main/examples/univariate)
(Rosenbrock function) where the Jacobian is computed via JAX and JVP/VJP are derived
automatically using `jvp_from_jacobian` and `vjp_from_jacobian`.

```{literalinclude} ../../../../examples/univariate_adfallbacks/tesseract_api.py
:language: python
:caption: tesseract_api.py
```

```{literalinclude} ../../../../examples/univariate_adfallbacks/tesseract_config.yaml
:language: yaml
:caption: tesseract_config.yaml
```

## See also

- {ref}`tr-autodiff` for background on differentiable programming in Tesseracts
- {doc}`/content/examples/building-blocks/finitediff` for a gradient-free alternative
  when you have no AD endpoint at all
- The [univariate example](https://github.com/pasteurlabs/tesseract-core/tree/main/examples/univariate)
  for the version with manually implemented JVP and VJP
