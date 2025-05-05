# Custom build steps: PyVista on ARM64 example

## Context

Tesseracts use by default an official Python docker image as the base image. Although this covers many useful cases, some system dependencies sometimes require a custom image and extra build steps.

## Example yaml
Via `tesseract_config.yaml`, it is possible to somewhat flexibly alter the build process to accomodate different needs. As a concrete example, here's what we had to do internally in order to to build an arm64 Tesseract with PyVista installed as a dependency:

```{literalinclude} ../../../examples/pyvista-arm64/tesseract_config.yaml
:language: yaml
```


Here is what the various configurations set above do to the build process:
  *  We started from a base image that had VTK 9.2 installed (`debian:trixie`), which was specified via `base_image`.
  *  The `target_platform` is set to `linux/arm64`, which will build the resulting image for an ARM64 architecture.
  *  We installed python3-vtk9 and python3-venv to get the VTK Python bindings and `venv` by specifying them in `extra_packages`. You can think of `extra_packages` as a list of packages which are `apt-get install`ed right on the base image just immediately after some other dependencies (like `git`, `ssh`, and `build-essential`) are installed.
  *  We can then run arbitrary commands on the image which is being built via `custom_build_steps`. This list of commands need to be specified as if they were in a Dockerfile. In particular, we start here by temporarily setting the user to `root`, as the default user in the Tesseract build process is `tesseractor` -- which does not have root privileges -- and then switch back to the `tesseractor` user at the very end. We then     run commands directly on the shell via `RUN` commands. All these steps specified in `custom_build_steps` are executed at the very end of the build process, followed only by a last execution of `tesseract-runtime check` that checks that the runtime can be launched and the user-defined `tesseract_api` module can be imported.
