# Install a Tesseract's dependencies from a Conda environment

To install a Tesseract's dependencies from a [conda](https://www.anaconda.com/docs/getting-started/miniconda/main)
environment, first export your environment (with the `--no-builds` flag):

```bash
conda env export --no-builds > tesseract_environment.yaml
```

Then, set the requirements `provider` as shown in
[`tesseract_config.yaml`](tesseract_config.yaml).

Now you can use the Tesseract as usual. `Tesseract.from_source` creates the
environment in `.tesseract-venv` next to `tesseract_api.py`, using the `conda`
on your `PATH` (or the one named by `CONDA_EXE` or `TESSERACT_CONDA_EXECUTABLE`):

```python
from tesseract_core import Tesseract

with Tesseract.from_source("examples/conda/tesseract_api.py") as t:
    print(t.apply({"message": "Hey!"})["cowsays"])
```

To package the Tesseract as a container image, build it with `tesseract build`:

```bash
$ tesseract build examples/conda
$ tesseract run helloworld-conda apply '{"inputs": {"message": "Hey!"}}'
```

If the base image does not ship with conda, the build installs
[Miniforge](https://github.com/conda-forge/miniforge) to create the environment,
but leaves it out of the final image. To use your own conda installation instead,
set `base_image` to an image that provides `conda` on its `PATH`. The final image
also derives from `base_image`, so it then includes that conda installation.
