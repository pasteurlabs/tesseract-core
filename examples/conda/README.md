# Build a Tesseract from a Conda environment

To build a Tesseract from a
[Anaconda/miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)
environment you can export your environment (with the `--no-builds` flag):
```bash
conda env export --no-builds > tesseract_environment.yaml
```

Additionally you have to set the correct `base_image` and requirements `provider` as shown
in the `tesseract_config.yaml`.

Finally, you can build and use the Tesseract as usual:
```bash
$ tesseract build examples/conda
$ tesseract run helloworld-conda apply '{"inputs": {"message": "Hey!"}}'
```
