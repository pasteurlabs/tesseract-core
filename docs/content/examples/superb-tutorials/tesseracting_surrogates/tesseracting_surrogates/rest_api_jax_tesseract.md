# JAX Tesseract REST API

## Via REST API

As before, we first launch the <span class="product">Tesseract</span> via `tesseract serve`.

```shell
$ tesseract serve supersede_mgn_ellipse_displacement:latest
 [i] Waiting for Tesseract containers to start ...
 [i] Container ID: 30407f03138f381abc40fcfc94236e93e75d688873a3f6108e93a7265d62fab7
 [i] Name: tesseract-cv8n8eyn91ys-tesseract_supersede_mgn_ellipse_displacement-ni4iwsanq2yl-1
 [i] Entrypoint: ['tesseract-runtime', 'serve']
 [i] View Tesseract: http://localhost:51638/docs
 [i] Project ID, use it with 'tesseract teardown' command: tesseract-cv8n8eyn91ys
tesseract-cv8n8eyn91ys
```

This starts running at `http://localhost:51638/` port. The `apply` and `jacobian` functions can be invoked with curl.

For the `apply` function we have:
```shell
$ curl http://localhost:51638/apply -H "Content-Type: application/json" -d '{"inputs": {"xc":0.5,"yc":0.5,"axis_x":0.15, "theta":45.0}}'
{"mean_displacement":{"shape":[],"dtype":"float64","data":{"buffer":0.13393983244895935,"encoding":"raw"}},"fx":null,"fy":null,"ux":null,"uy":null}%
```

For the `jacobian` function we have:

```shell
$ curl http://localhost:51638/jacobian -H "Content-Type: application/json" -d '{"inputs": {"xc":0.5,"yc":0.5,"axis_x":0.15, "theta":45.0}, "jac_inputs": ["xc", "yc", "axis_x", "theta"], "jac_outputs": ["mean_displacement"]}'
{"xc":{"mean_displacement":{"shape":[1],"dtype":"float64","data":{"buffer":[0.5412712693214417],"encoding":"raw"}}},"yc":{"mean_displacement":{"shape":[1],"dtype":"float64","data":{"buffer":[0.22620902955532074],"encoding":"raw"}}},"axis_x":{"mean_displacement":{"shape":[1],"dtype":"float64","data":{"buffer":[0.3307884931564331],"encoding":"raw"}}},"theta":{"mean_displacement":{"shape":[1],"dtype":"float64","data":{"buffer":[-0.000037455003621289507],"encoding":"raw"}}}}%
```

We can stop the <span class="product">Tesseract</span> using `tesseract teardown` with the generated project ID.
```shell
$ tesseract teardown tesseract-cv8n8eyn91ys
tesseract-cv8n8eyn91ys
 [i] Tesseracts are shutdown for Docker Compose project ID: tesseract-cv8n8eyn91ys
```
