# AD Endpoint Derivation Fallbacks

This guide shows how to derive missing autodiff endpoints from ones you have already
implemented, using the experimental AD fallback helpers.

```{warning}
This feature is **experimental** and available in `tesseract_core.runtime.experimental`.
The API may change in future releases.
```

## Overview

Tesseracts expose up to three AD endpoints — `jacobian`, `jacobian_vector_product`
(JVP), and `vector_jacobian_product` (VJP). In practice you may have only implemented
the Jacobian (e.g. via an AD framework), and need JVP or VJP for a particular workflow.

The helpers in this guide derive JVP and VJP from an existing Jacobian automatically,
without writing additional gradient code.

```{warning}
Deriving JVP or VJP from the Jacobian requires computing the **full Jacobian matrix**
first, which can be expensive for high-dimensional inputs or outputs. Use these helpers
when the derived endpoint is not on the hot path of your workflow.
```

## Usage

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

## Full source code

The complete example is the `univariate_adfallbacks` Tesseract — a variant of
[univariate](https://github.com/pasteurlabs/tesseract-core/tree/main/examples/univariate)
(Rosenbrock function) where the Jacobian is computed via JAX and JVP/VJP are derived
automatically using the fallback helpers.

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
