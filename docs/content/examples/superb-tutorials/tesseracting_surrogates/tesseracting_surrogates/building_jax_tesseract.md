# Building the Tesseract (JAX surrogate)

Once the three files (`tesseract_api.py`, `tesseract_config.yaml` and `tesseract_requirements.txt`) are configured, the <span class="product">Tesseract</span> can be built with the same requirements as above.

```shell
$ tesseract build examples/tesseracts/supersede_mgn_ellipse_displacement/
 [i] Building image ...
 [i] Built image sha256:fd0204db167e,
['tesseract_supersede_mgn_ellipse_displacement:latest']
```

## Call Tesseract via CLI

Similar to the Julia surrogate, we can invoke the `apply` and `jacobian` functions.

`apply` endpoint:

```shell
$ tesseract run supersede_mgn_ellipse_displacement:latest apply --inputs '{"xc":0.5,"yc":0.5,"axis_x":0.15, "theta":45.0}'

{"mean_displacement":{"shape":[],"dtype":"float64","data":{"buffer":0.13393983244895935,"encoding":"raw"}},"fx":null,"fy":null,"ux":null,"uy":null}compiling jit functions...
```

`jacobian` endpoint:

```shell
$ tesseract run supersede_mgn_ellipse_displacement:latest jacobian --inputs '{"xc":0.5,"yc":0.5,"axis_x":0.15, "theta":45.0}' --jac-inputs '["xc", "yc", "axis_x", "theta"]' --jac-outputs '["mean_displacement"]'
{"xc":{"mean_displacement":{"shape":[1],"dtype":"float64","data":{"buffer":[0.5412712693214417],"encoding":"raw"}}},"yc":{"mean_displacement":{"shape":[1],"dtype":"float64","data":{"buffer":[0.22620902955532074],"encoding":"raw"}}},"axis_x":{"mean_displacement":{"shape":[1],"dtype":"float64","data":{"buffer":[0.3307884931564331],"encoding":"raw"}}},"theta":{"mean_displacement":{"shape":[1],"dtype":"float64","data":{"buffer":[-0.000037455003621289507],"encoding":"raw"}}}}compiling jit functions...
```

## Push the Tesseract to a registry:

In order to share Tesseracts and keep track of them, you can push them to a Bibliotheca registry; make sure you configured
the `tesseract` CLI with the steps mentioned
<a href="../../../../../../index.html#tesseract-registry">here</a>;
then you can run the following command:
```bash
$ tesseract push supersede_mgn_ellipse_displacement
```
