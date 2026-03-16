# Tips for Defining Tesseract APIs

## Advanced Pydantic features

```{warning}
Pydantic V2 metadata and transformations like `AfterValidator`, `Field`, `model_validator`, and `field_validator` are generally supported for all inputs named `inputs` (first argument of various endpoints), and outputs of `apply`. They are silently stripped in all other cases (except in [`abstract_eval`](#abstract-eval-pydantic)).
```

Tesseract uses [Pydantic](https://docs.pydantic.dev/latest/) to define and validate endpoint signatures. Pydantic is a powerful library that allows for complex type definition and validation, but not all of its features are supported by Tesseract.

One core feature of Tesseract is that only the input and output schema for `apply` is user-specified, while all other endpoint schemas are inferred from them, which cannot preserve all features of the original schema.

Tesseract supports almost all Pydantic features for endpoint inputs named `inputs` (that is, the first argument to `apply`, `jacobian`, `jacobian_vector_product`, `vector_jacobian_product`):

```python
class InputSchema(BaseModel):
    # ✅ Field metadata + validators
    field: int = Field(..., description="Field description", ge=0, le=10)

    # ✅ Nested models
    nested: NestedModel

    # ✅ Default values
    default: int = 10

    # ✅ Union types
    union: Union[int, str]
    another_union: int | str

    # ✅ Generic containers
    list_of_ints: List[int]
    dict_of_strs: Dict[str, str]

    # ✅ Field validators
    validated_field: Annotated[int, AfterValidator(my_validator)]

    # ✅ Model validators
    @model_validator
    def check_something(self):
        if self.field > 10:
            raise ValueError("Field must be less than 10")
        return self

    # ❌ Recursive models, will raise a build error
    itsame: "InputSchema"

    # ❌ Custom types with __get_pydantic_core_schema__, will raise runtime errors
    custom: CustomType

```

```{note}
In case you run into issues with Pydantic features not listed here, please [open an issue](https://github.com/pasteurlabs/tesseract-core/issues/new/choose).
```

(x86-vs-arm)=

### 🔪 Sharp edge: x86 vs ARM architecture on Apple Silicon

If you're using a Mac, your system uses the ARM64 processor architecture, while many Docker images and Python packages are built for x86_64 (also known as AMD64). This can lead to architecture incompatibilities when building or running Tesseracts.

**Common symptoms:**

- Build failures with errors mentioning "platform mismatch" or "exec format error"
- Runtime errors like `exec /tesseract/entrypoint.sh: exec format error`
- Slow performance due to Rosetta 2 emulation
- Package installation failures in `tesseract_requirements.txt`
- A Python package fails to install because it doesn't provide a pre-built Linux ARM64 wheel

**Solutions:**

1. **Build x86_64 images for sharing or compatibility (recommended):** If you intend to share Tesseracts with others, deploy to x86_64 servers, or are running into difficulties with missing ARM64 wheels, build for x86_64. Edit your `tesseract_config.yaml`:

   ```yaml
   # tesseract_config.yaml
   build_config:
     target_platform: linux/amd64 # Build for x86_64
   ```

   Note this uses QEMU emulation and will be slower to build, but produces images that work everywhere.

2. **Build for your native architecture (for local development):** By default, Tesseract builds for your native platform. If you only need to run locally, this is faster. You can explicitly set it in `tesseract_config.yaml`:

   ```yaml
   # tesseract_config.yaml
   build_config:
     target_platform: linux/arm64 # Explicitly set for Apple Silicon
   ```

3. **Use ARM-compatible base images:** Some base images don't have ARM64 variants. Check that your base image supports ARM64 (e.g., `python:3.11-slim` supports both architectures).

4. **Handle packages without Linux ARM64 wheels:** Some Python packages don't provide pre-built wheels for Linux ARM64. Note that a macOS ARM64 wheel is not sufficient here, since Tesseracts run in Linux containers.

   One solution is to include the system packages required to build the wheel from source during the `tesseract build` step by specifying the `extra_packages` build option. Common required packages may include `build-essential`, `gcc`, or `nvidia-cuda-toolkit`:

   ```yaml
   # tesseract_config.yaml
   build_config:
     extra_packages:
       - build-essential
       - gcc
   ```

   Other options include using conda (`venv_backend: conda`) or pinning to a version that has ARM64 support. Alternatively, build for x86_64 as described above.

To verify the architecture of a built Tesseract image: `docker inspect --format='{{.Architecture}}' my_tesseract:latest`

(abstract-eval-pydantic)=

### 🔪 Sharp edge: `abstract_eval` and field validators

A special case are the inputs and outputs to `abstract_eval`, which also keep the full Pydantic schema, albeit with some limitations. In particular, all `Array` types will be replaced by a special object that only keeps the shape and dtype of the array, but not the actual data. Therefore, validators that depend on arrays **must** check for this special object and pass it through:

```python
class InputSchema(BaseModel):
    myarray: Array[(None,), Float64]

    @field_validator("myarray", mode="after")
    @classmethod
    def check_array(cls, v) -> np.ndarray:
        # Pass through non-arrays
        # ⚠️ Without this, abstract_eval breaks ⚠️
        if not isinstance(v, np.ndarray):
            return v

        # This is the actual validator that's used for other endpoints
        return v + 1
```

## Building Tesseracts with private dependencies

In case you have some dependencies in `tesseract_requirements.txt` for which you need to
ssh into a server (e.g., private repositories which you specify via "git+ssh://..."),
you can make your ssh agent available to `tesseract build` with the option
`--forward-ssh-agent`. Alternatively you can use `pip download` to download a dependency
to the machine that builds the Tesseract.

## Customizing the build process

There are several steps in the process of building a Tesseract image
which can be configured via the `tesseract_config.yaml` file, in particular the `build_config` section.
For example:

- By default the base image is `debian:bookworm-slim`.
  Depending on your specific needs (different python version,
  preinstalled dependencies, ...), it might be beneficial to
  specify a different one in `base_image`.
  There is however the constraint that
  whatever other image you specify, it must be Ubuntu- or
  Debian-based.
- The default target architecture is "native" (same as the host platform).
  If you need to build for a specific platform, use e.g. `target_platform: "linux/arm64"`.
- As `tesseract_requirements.txt` only allows you to specify Python
  dependencies, if there are system ones you need to install inside
  the Tesseract you can do so via the `extra_packages` list. All
  packages you specify will be installed via `apt-get`.
- You can copy data inside a Tesseract via the `package_data` list.
  The data will be then part of the Tesseract image. This is a
  good choice for some static artifacts you need to have available
  for computation, such as the weights of a machine learning model.
  Source paths may reference files outside the Tesseract directory (e.g., `../data/weights.bin`),
  but external entries must have unique target paths.
- If you want to further customize the way the image is built,
  you can add arbitrary commands to the Dockerfile specifying
  the build process via the `custom_build_steps` list. Use
  the same syntax you would use in a Dockerfile. To see where your
  commands would be added in the build process, have a look at
  the [Dockerfile template](https://github.com/pasteurlabs/tesseract-core/blob/main/tesseract/templates/Dockerfile.base)
  `tesseract build` uses by default.

## Creating a Tesseract from a Python package

Sometimes it is useful to create a Tesseract from an already-existing
Python package. In order to do so, you can run `tesseract init` in the root folder of
your package (i.e., where `setup.py` and `requirements.txt` would be). Import your package
as needed in `tesseract_api.py`, and specify the dependencies you need at runtime in
`tesseract_requirements.py`.
