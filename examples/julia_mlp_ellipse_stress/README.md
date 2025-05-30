# Running the `julia_mlp_ellipse_stress` Tesseract

## Example Command Line Queries

Assuming the tesseract is named `julia_mlp_ellipse_stress`.
```
tesseract run julia_mlp_ellipse_stress apply --inputs '{"xc":0.5,"yc":0.5,"axis_x":0.15, "theta":45.0}'
```

## Example HTTP Queries
Assuming the tesseract is served locally on port 8000.

Call apply endpoint
```
curl -X POST -H "Content-Type: application/json" -d '{"inputs":{ "xc":0.5, "yc":0.5, "axis_x":0.15, "theta":45.0}}' http://127.0.0.1:8000/apply
```

Call jacobian endpoint
```
curl -X POST -H "Content-Type: application/json" -d '{"inputs":{ "xc":0.5, "yc":0.5, "axis_x":0.15, "theta":45.0}, "jac_inputs":["xc","yc","axis_x","theta"], "jac_outputs":["mean_stress"]}' http://127.0.0.1:8000/jacobian
```

## Running Locally with the Tesseract Runtime
If trying to run locally with the `tesseract_runtime` environment, you will need to install and instantiate the Julia project environement in `StressSurrogate`.

To install julia, checkout https://julialang.org/downloads/. Once installed, you will need to instantiate the necessary Julia environment. From the command line go to the `StressSurrogate` folder, launch julia (i.e. by typing `julia` at the command line) and run the following commands:

```julia
using Pkg
Pkg.instantiate()
```
This will download the necessary packages listed in the `Project.toml`. Now exit julia and navigate back up one directory (so `StressSurrogate` is now in your path). Launch the tesseract runtime as normally done (i.e. export `TESSERACT_API_PATH` to point to the `tesseract_api.py` and run `tesseract_runtime serve`.)

Note you will also need to install the python packages in `tesseract_requirements.txt`.
