---
orphan: true
---

# Building the Jax Solver Tesseract for Lorenz-96

This examples demonstrates how the Jax Solver Tesseract for the Lorenz-96 model is built for the purposes of the Lorenz-96 tutorial.

Below is the input and output schema definition for the Lorenz Tesseract.
```{literalinclude} ../../../../advanced_examples/lorenz/lorenz_tesseract/tesseract_api.py
:language: python
:pyobject: InputSchema
```

```{literalinclude} ../../../../advanced_examples/lorenz/lorenz_tesseract/tesseract_api.py
:language: python
:pyobject: OutputSchema
```

Below is the implementation of the apply function, which takes in an initial condition and returns a trajectory of physical states.
```{literalinclude} ../../../../advanced_examples/lorenz/lorenz_tesseract/tesseract_api.py
:language: python
:start-at: def lorenz96_step
:end-before:     return apply_jit(inputs.model_dump())
```
