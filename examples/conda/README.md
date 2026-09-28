# Build a Tesseract from a Conda environment

To build a Tesseract from a [conda](https://www.anaconda.com/docs/getting-started/miniconda/main)
environment, first export your environment (with the `--no-builds` flag):

```bash
conda env export --no-builds > tesseract_environment.yaml
```

Then, set the requirements `provider` as shown in
[`tesseract_config.yaml`](tesseract_config.yaml).

If the base image does not ship with conda, Tesseract installs
[Miniforge](https://github.com/conda-forge/miniforge) to create the environment.
conda itself stays out of the final image, which only contains the environment.
To use your own conda installation instead, set `base_image` to an image
that provides `conda` on its `PATH` (the final image is then based on that
image too, including its conda installation).

Finally, you can build and use the Tesseract as usual:

```bash
$ tesseract build examples/conda
$ tesseract run helloworld-conda apply '{"inputs": {"message": "Hey!"}}'
```
