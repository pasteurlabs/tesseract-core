# Julia Tesseract REST API

The RESTful API call for our Julia <span class="product">Tesseract</span> is described below. It can be launched via `tesseract serve julia_linear_elasticity_mlp_stress:latest`, i.e.,

```shell
$ tesseract serve julia_mlp_ellipse_stress:latest
 [i] Waiting for Tesseract containers to start ...
 [i] Container ID:
e30698825333a5100d12a5a9c0622db641c7cf1f9def42e8f8e6f040f4c67382
 [i] Name:
tesseract-m31472eu3pd1-julia_linear_elasticity_mlp_stress-onx60hqwqgv9-1
 [i] Entrypoint: ['tesseract-runtime', 'serve']
 [i] View Tesseract: http://localhost:49866/docs
 [i] Project ID, use it with 'tesseract teardown' command: tesseract-m31472eu3pd1
tesseract-m31472eu3pd1
```
We see this <span class="product">Tesseract</span> is running at `http://localhost:49866/`.

> **_NOTE:_** The port at which the <span class="product">Tesseract</span> serves is unique and random. The one recieved from `tesseract serve` should be saved and used for subsequent operations.

We can invoke the `apply` <span class="product">Tesseract</span> endpoint using `curl`.

```shell
$ curl http://localhost:49866/apply -H "Content-Type: application/json" -d '{"inputs": {"xc":0.5,"yc":0.5,"axis_x":0.15, "theta":45.0}}'
{"mean_stress":{"shape":[],"dtype":"float64","data":{"buffer":18597.497139362473,"encoding":"raw"}},"fx":null,"fy":null,"s":null}%
```

Similarly, for `jacobian` we have:

```shell
$ curl http://localhost:49866/jacobian -H "Content-Type: application/json" -d '{"inputs": {"xc":0.5,"yc":0.5,"axis_x":0.15, "theta":45.0}, "jac_inputs": ["xc", "yc", "axis_x", "theta"], "jac_outputs": ["mean_stress"]}'
{"xc":{"mean_stress":{"shape":[1],"dtype":"float64","data":{"buffer":[35826.470016987434],"encoding":"raw"}}},"yc":{"mean_stress":{"shape":[1],"dtype":"float64","data":{"buffer":[29297.982516374308],"encoding":"raw"}}},"axis_x":{"mean_stress":{"shape":[1],"dtype":"float64","data":{"buffer":[7468.335501146403],"encoding":"raw"}}},"theta":{"mean_stress":{"shape":[1],"dtype":"float64","data":{"buffer":[-1.8547801238481354],"encoding":"raw"}}}}%

```

In order to stop the running <span class="product">Tesseract</span>, one can use the command `tesseract teardown` with the corresponding project ID (generated after running `tesseract serve`):

```shell
$ tesseract teardown tesseract-m31472eu3pd1
 [i] Tesseracts are shutdown for Docker Compose project ID: tesseract-m31472eu3pd1
```

> **_NOTE:_** For downstream application development, one would more likely use HTTP and JSON libraries to manage calls to <span class="product">Tesseract</span> endpoints. The [optimization tutorial](../../optimization_with_surrogates/linear_ellipse_optimization.md) demonstrates how this is done using Julia.
