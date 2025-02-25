# Optimizing a Rosenbrock function with Tesseracts and a PyTorch backend


## Description

This example shows how to use tesseracts with a PyTorch backend, specifically in optimizing a 2-dimensional [log-Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function). For those familiar with setting up optimizers in pytorch, this should be very familiar. The only caveat is understanding how to take gradients computed by the pytorch backend within a tesseract (i.e. the `jacobian` endpoint in `tesseract_api.py`), and then feeding those gradients to a process running a `torch.optim.Adam` optimizer locally (i.e. the `optimization.py` script).

## How to run

First build the image (called pytorch_optimization) and then serve the tesseract, which exposes an HTTP endoint:
```bash
tesseract build examples/sciml/pytorch_optimization
tesseract serve pytorch_optimization
```

Assuming the port is `8080`, we can run the Python optimization script from the command line as

```bash
python optimization.py --port 8080
```

which should return some plots of the optimization progress.

Additional parameters, such as learning rate, number of iterations, and seed, can be passed as follows:
```bash
python optimization.py --port 8080 --lr .001 --niter 500 --seed 99
```

## Requirements

Standard tesseract requirements and
* torch
* matplotlib
