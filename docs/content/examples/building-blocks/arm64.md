# Custom build steps: PyVista on ARM64

## Context

Tesseracts use by default an official Python docker image as the base image. Although this covers many useful cases, some system dependencies sometimes require a custom image and extra build steps.

## Example yaml

Via `tesseract_config.yaml`, it is possible to somewhat flexibly alter the build process to accomodate different needs. As a concrete example, here's what we had to do internally in order to to build an arm64 Tesseract with PyVista installed as a dependency:

```{literalinclude} ../../../../examples/pyvista-arm64/tesseract_config.yaml
:language: yaml
```

Using the `custom_build_steps` field, we can run arbitrary commands on the image as if they were in a Dockerfile. We run commands directly on the shell via `RUN` commands. All these steps specified in `custom_build_steps` are executed at the very end of the build process, followed only by a last execution of `tesseract-runtime check` that checks that the runtime can be launched and the user-defined `tesseract_api` module can be imported.
