# Building the JAX Solver Tesseract for Lorenz-96

This examples demonstrates how the JAX solver Tesseract for the Lorenz-96 model is built for the purposes of the data assimilation demo.

Below is the input and output schema definition for the Lorenz Tesseract.

```{literalinclude} ../../../demo/data-assimilation-4dvar/lorenz_tesseract/tesseract_api.py
:language: python
:pyobject: InputSchema
```

```{literalinclude} ../../../demo/data-assimilation-4dvar/lorenz_tesseract/tesseract_api.py
:language: python
:pyobject: OutputSchema
```

Below is the implementation of the apply function, which takes in an initial condition and returns a trajectory of physical states.

```{literalinclude} ../../../demo/data-assimilation-4dvar/lorenz_tesseract/tesseract_api.py
:language: python
:start-at: def lorenz96_step
:end-before:     return apply_jit(inputs.model_dump())
```
