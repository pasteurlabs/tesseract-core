# Vector Add

## Context
Example using vector add with differentiable inputs and jacobians.

## Example Tesseract (examples/vectoradd)

In this example of vectoradd, we have the 2 vectors both as Differentiable Arrays.
Inputs must be Differentiable if we wish to write a Jacobian function for the Tesseract.

We also have an example of a basic apply function and how it utilizes the inputs.

```{literalinclude} ../../../examples/vectoradd/tesseract_api.py
:pyobject: InputSchema
:language: python
```

```{literalinclude} ../../../examples/vectoradd/tesseract_api.py
:pyobject: apply
:language: python
```

### Bonus: Jacobian

As previously mentioned, the inputs are marked as Differentiable in order for us to write
a Jacobian function.

In order for us to do that, we must also write the abstract eval function.

```{literalinclude} ../../../examples/vectoradd/tesseract_api.py
:pyobject: abstract_eval
:language: python
```

```{literalinclude} ../../../examples/vectoradd/tesseract_api.py
:pyobject: jacobian
:language: python
```
