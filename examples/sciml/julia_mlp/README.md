# Building the Julia MLP Tesseract Example Locally

## Install tesseract
First, install the tesseract package to your local python environment. From the
source tesseract repository simply run:
```
pip install .
```
If you want to test the endpoints with `run_tesseract_api.py` you will need to install the tesseract runtime. From the source tesseract directory run `pip install .[runtime]`

## Build the tesseract
```
tesseract build examples/sciml/julia_mlp
```

## Run the Command Line Interface
```
tesseract run julia_flux_mlp apply '{"inputs": {"n_epochs" : 100}}'
```

## Serve the tesseract
```
tesseract serve julia_flux_mlp
```
Alternatively, you can use docker directly and specify the port: `docker run -p 8000:8000 julia_flux_mlp serve`

## Make an example HTTP request
Assuming our Tesseract is served on port 8000, we can call the `apply` endpoint like following:
```
curl -H "Content-Type: application/json" -d '{"inputs":{ "n_epochs":100 }}' http://127.0.0.1:8000/apply
```
