# Use (private) dependencies in a tesseract via tarball

> [!NOTE]
> TLDR: Check `./build.sh` for an example of how to build this tesseract.

While there is a simple way to install (private) dependencies in a tesseract via
the `--forward-ssh-agent` flag from `tesseract build` in some cases - e.g. when
building tesseracts on remote machines - it might not be desirable/possible to
install via `git+ssh`. This example shows how to use a tarball instead:

1. Download (private) tarball
2. Add local path to `tesseract_requirements.txt` starting with `./` (no need to
   add it to `tesseract_config.package_data`)
3. `tesseract build`
