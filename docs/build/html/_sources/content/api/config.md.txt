# Configuration via `tesseract_config.yaml`

The `tesseract_config.yaml` file contains a Tesseract's metadata, such as its name, description, version, and build configuration.

## Example file

```{literalinclude} ../../../examples/helloworld/tesseract_config.yaml
```

## Schema

The `TesseractConfig` class is used to define the schema for the `tesseract_config.yaml` file. It contains the following fields:

```{eval-rst}
.. autopydantic_model:: tesseract_core.sdk.api_parse.TesseractConfig
    :member-order: bysource
    :model-show-config-summary: False


.. autopydantic_model:: tesseract_core.sdk.api_parse.TesseractBuildConfig
    :member-order: bysource
    :model-show-config-summary: False
```
