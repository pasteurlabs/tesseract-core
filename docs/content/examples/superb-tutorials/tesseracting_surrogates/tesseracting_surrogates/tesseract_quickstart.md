# Tesseract Quickstart
A detailed documentation on <span class="product">Tesseracts</span> lives <a href="../../../../../tesseract-docs/index.html">here</a>. In this tutorial, the basic instructions to create a <span class="product">Tesseract</span> and build it from trained Julia and JAX surrogate models are provided assuming that the <a href="../../../../../../index.html#tesseract-cli">Tesseract CLI is installed</a> in the Python enviroment.

In order to start creating a <span class="product">Tesseract</span> in the current directory (to be otherwise specified with ` --target-dir [DIRECTORY] `), the following command must be run.

```bash
$ tesseract init
```
This creates three file templates (explained below) that serve as a starting point; then, users can modify these templates according to their requirements. The command above also asks for the <span class="product">Tesseract</span> name during the execution while creating these file.
- `tesseract_api.py`, a python module where all the core computations (such as `eval_forward`, `eval_gradient`, `calc_mean_mean_displacement_from_input`, etc.) should be implemented.
- `tesseract_config.yaml`, a `yaml` file where you can specify metadata, such as the <span class="product">Tesseract</span> name and version, various build options, which base Docker image to use, definitions of custom steps in building the <span class="product">Tesseract</span>, access to external data, and so on.
- `tesseract_requirements.txt`, a `txt` file where you can specify the (Python) dependencies of the <span class="product">Tesseract</span>. This file should be in the [requirements file format](https://pip.pypa.io/en/stable/reference/requirements-file-format/).
