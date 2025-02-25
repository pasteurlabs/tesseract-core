# Example of optimisation with Tesseracts

This example shows how to setup an optimisation process using Tesseract.

## Description
In this example Tesseract implements a parametrised [radial basis function](https://en.wikipedia.org/wiki/Radial_basis_function) (RBF). Aim of the optimisation process is to find the best parameters to fit this funcion to a noisy data. Implementation is located across the following files:
* tesseract_api.md, tesseract_config.yaml, tesseract_requirements.txt - implementation of the RBF Tesseract
* tesseract_client.py - a simple wrapper around HTTP requests made to the RBF Tesseract API
* optimisation_routine.py - implementation of the optimisation process

## How to run
First off, build the RBF tesseract and serve it, taking note of the port at which the HTTP endpoint is being served:

```bash
$ tesseract build examples/sciml/rbf_fitting
$ tesseract serve rbf_fitting
```

Let's assume the port is `8080`. To run the optimisation script:

```bash
$ python examples/sciml/rbf_fitting/optimization_routine.py --port 8080
```

If everything worked correctly, you will see the iteration progress output in the console, followed by several graphs illustrating the obtained fit.

### Requirements
In addition to standard Tesseract requirements, this example relies on following packages for optimisation and plotting:
* jax
* optax
* matplotlib
