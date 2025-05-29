# Building the Tesseract (Julia surrogate)

At this point, `tesseract_api.py`, `tesseract_config.yaml` and `tesseract_requirements.txt` are configured and we are ready to build the first <span class="product">Tesseract</span> as a Docker container. Thus, **always make sure that Docker is running.**

To build it, we run the following command:
```shell
$ tesseract build <directory>
```
`directory` is path to the directory where `tesseract_api.py`, `tesseract_config.yaml` and `tesseract_requirements.txt` are placed. More details on building a <span class="product">Tesseract</span> can be found <a href="../../../../../tesseract-docs/content/user-guide/create.html">here</a>. We provide an example of the building process below.

```shell
$ tesseract build examples/tesseracts/julia_mlp_ellipse_stress/
 [i] Building image ...
 [i] Built image sha256:b7a2a234738d, ['julia_linear_elasticity_mlp_stress:latest']
```

Now that everything is set to run the <span class="product">Tesseract</span>, we can interact with it via command line interface (CLI) or RESTful API.

## Call Tesseract via CLI

The core operation of <span class="product">Tesseract</span> is the `apply` function described above. The CLI call looks as follows.

```shell
$ tesseract run julia_mlp_ellipse_stress:latest apply --inputs '{"xc":0.5,"yc":0.5,"axis_x":0.15, "theta":45.0}'
```
Here `julia_linear_elasticity_mlp_stress:latest` is the name of Tesseracted surrogate (`julia_linear_elasticity_mlp_stres` is specified in the `tesseract_config.yaml` file). The user-defined ellipse parameters are specified in a JSON string (`{"xc":0.5,"yc":0.5,"axis_x":0.15, "theta":45.0}`). The latter must meet the type and structure defined by `InputSchema` where it is parsed. Then, the values are assigned to `input` and used inside the `apply` function.

The above command produces the following:

```shell
 Activating project at `/tesseract/StressSurrogate`
{"mean_stress":{"shape":[],"dtype":"float64","data":{"buffer":18597.497139362473,"encoding":"raw"}},"fx":null,"fy":null,"s":null}%
```

If required, users can optionally get loading field and/or the entire stress field.

```shell
$ tesseract run julia_mlp_ellipse_stress:latest apply --inputs '{"xc":0.5,"yc":0.5,"axis_x":0.15, "theta":45.0, "return_stress_components": "True"}'
  Activating project at `/tesseract/StressSurrogate`
{"mean_stress":{"shape":[],"dtype":"float64","data":{"buffer":18597.497139362473,"encoding":"raw"}},"fx":null,"fy":null,"s":{"shape":[2601],"dtype":"float64","data":{"buffer":[9152.746690386275,19983.31286143374,26281.071982564506, ....
```
As the returned array is very big, the whole output is not shown here.

Similar to `apply` endpoint, the optional `jacobian` endpoint, providing gradient of the quantity of interest with respect to the ellipse paramters, can also be invoked. It requires additional inputs of `--jac-inputs` and `--jac-outputs` specifying the inputs vector and output vector for a Jacobian evaluation. For our case, the output vector is a scalar and its gradient is evaluated with respect to all the input parameters.

```shell
$ tesseract run julia_mlp_ellipse_stress:latest jacobian --inputs '{"xc":0.5,"yc":0.5,"axis_x":0.15, "theta":45.0}' --jac-inputs '["xc", "yc", "axis_x", "theta"]' --jac-outputs '["mean_stress"]'
  Activating project at `/tesseract/StressSurrogate`
{"xc":{"mean_stress":{"shape":[1],"dtype":"float64","data":{"buffer":[35826.470016987434],"encoding":"raw"}}},"yc":{"mean_stress":{"shape":[1],"dtype":"float64","data":{"buffer":[29297.982516374308],"encoding":"raw"}}},"axis_x":{"mean_stress":{"shape":[1],"dtype":"float64","data":{"buffer":[7468.335501146403],"encoding":"raw"}}},"theta":{"mean_stress":{"shape":[1],"dtype":"float64","data":{"buffer":[-1.8547801238481354],"encoding":"raw"}}}}%
```

## Push the Tesseract to a registry:

In order to share Tesseracts and keep track of them, you can push them to a Bibliotheca registry; make sure you configured
the `tesseract` CLI with the steps mentioned
<a href="../../../../../../index.html#tesseract-registry">here</a>;
then you can run the following command:
```bash
$ tesseract push julia_mlp_ellipse_stress
```
