The contents of this folder are copied to the build context and installed during `tesseract build`. Providing a separate `pyproject.toml` allows us to keep the runtime dependencies separate from the SDK dependencies.

We assume the following folder structure during the build process:

```
__tesseract_runtime__/
    <contents of this folder>
    tesseract_core/
        runtime/
            <contents of tesseract_core/runtime>
```
