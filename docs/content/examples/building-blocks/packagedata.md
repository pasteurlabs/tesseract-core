# Packaging local files into a Tesseract

{gh-tree}`View on GitHub <examples/package_data>`

## Context

You may need to include local files into your Tesseract, e.g. if we want to load in pretrained model weights, or artifacts that we may need access to. This guide will demonstrate how.

(package-data)=

## Example Tesseract (examples/package_data)

In this example, we want to include and process `parameters.json` file in our Tesseract.

`Tesseract.from_source` runs `tesseract_api.py` where it is, so `apply` can read `parameters.json` from the same directory without further setup. A container image only contains the files you copy into it. To include the file in images built with `tesseract build`, we add it to `package_data` in `tesseract_config.yaml`, with the source (a local path) and the destination (a path in the container, relative to the directory that holds `tesseract_api.py`).

```{literalinclude} ../../../../examples/package_data/tesseract_config.yaml
:language: yaml
:start-at: build_config:
```

Since the target path matches the source path, the file sits next to `tesseract_api.py` in both cases, and the `apply` function can locate it relative to its own module:

```{literalinclude} ../../../../examples/package_data/tesseract_api.py
:language: python
:start-at: def apply(inputs: InputSchema) -> OutputSchema:
```
