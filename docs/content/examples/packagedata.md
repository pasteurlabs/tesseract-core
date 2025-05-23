# Passing in local data into a Tesseract

## Context
You may need to include local files into your Tesseract, e.g. if we want to load in pretrained model weights, or artifacts that we may need access to. This guide will demonstrate how.

(package-data)=
## Example Tesseract (examples/package_data)

In this example, we want to include and process `parameters.json` file in our Tesseract.

In order to achieve this, we must modify the `tesseract_config.yaml` file with the source (local path) and the target destinations (location in the Tesseract container).

```{literalinclude} ../../../examples/package_data/tesseract_config.yaml
:language: yaml
:start-at: build_config:
```

Now that it is included in the build config, we can use it in our Tesseracts apply function as normal.

```{literalinclude} ../../../examples/package_data/tesseract_api.py
:language: python
:start-at: def apply(inputs: InputSchema) -> OutputSchema:
```
