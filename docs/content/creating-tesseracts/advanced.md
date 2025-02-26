# Tips for defining Tesseract APIs

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

## Debugging build failures

There are also several options you can provide to `tesseract build` which can be helpful in
various circumstances:
- The output of the various steps which happen under-the-hood while doing a build will
 only be printed if something fails; this means that your shell might appear unresponsive
 during this process. If you want more detailed information on what's going on during your
 build, and see updates about it in real-time, use `--loglevel debug`.
- By default `tesseract build` does not cache the various steps of building a Tesseract[^1].  This is a good choice once your Tesseract is mature enough (i.e., once you are not re-building frequently). While you are still implementing new features in the Tesseract and rebuilding it
 often, we recommend to use `--keep-build-cache` flag to keep your build steps in the cache,
 so that the next `tesseract build` runs faster.
 ```{note}
 A single `tesseract build` command without
 the `--keep-build-cache` option will invalidate the cache.
 ```
- `--config-override` can be used to manually override options specified in the `tesseract_config.yaml`,
 for example: `--config-override build_config.target_platform=linux/arm64`
- `tesseract build` relies on a `docker build` command to create the Tesseract image. By
 default, the build context is a temporary folder to which all necessary files to build a Tesseract
 are copied to. The option `--docker-build-dir <directory>` allows you to specify a different
 directory where to do this operations. This might be useful to debug issues which
 arise while building a Tesseract, as in `directory` you will see all the context available to
 `docker build` and nothing else.

## Building Tesseracts with private dependencies

- In case you have some dependencies in `tesseract_requirements.txt` for which you need to
 ssh into a server (e.g., private repositories which you specify via "git+ssh://..."),
 you can make your ssh agent available to `tesseract build` with the option
 `--forward-ssh-agent`. Alternatively you can use `pip download` to download a dependency
 to the machine that builds the Tesseract.

## Customizing the build process

There are several steps in the process of building a Tesseract image
which can be configured via the `tesseract_config.yaml` file, in particular the `build_config` section.
For example:
  - By default the base image is `python:3.12-slim-bookworm`.
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
  - If you want to further customize the way the image is built,
    you can add arbitrary commands to the Dockerfile specifying
    the build process via the `custom_build_steps` list. Use
    the same syntax you would use in a Dockerfile. To see where your
    commands would be added in the build process, have a look at
    the [Dockerfile template](https://github.com/pasteurlabs/tesseract-core/blob/main/tesseract/templates/Dockerfile.base)
    `tesseract build` uses by default.

(tr-without-docker)=
## Tesseracts without containerization

While developing a Tesseract, the process of building and rebuilding the
tesseract image for quick local tests can be very time-consuming. Using
`--keep-build-cache` ameliorates this issue, but the fastest and most
convenient way to speed this up is to just run the code you are developing
in your virtual environment.

In order to do so, you should:
  - Make sure you have a development installation of Tesseract (see <project:#installation-dev>).
    In particular, calling `which tesseract-runtime` in the Terminal should return a path in your
    virtual environment.
  - Install your Tesseract's dependencies via `pip install -r tesseract_requirements.txt`.
  - Point to the runtime where it can find the `tesseract_api.py` of the Tesseract you are working on.
    This is done by setting the `TESSERACT_API_PATH` environment variable via
    `export TESSERACT_API_PATH=/path/to/your/tesseract_api.py`.

After that is done, you will be able to use the `tesseract_runtime` command in your shell.
This is the exact same command that is launched inside Tesseract containers to run their
various endpoints, and its syntax mirrors the one of `tesseract run`.

For instance, to call the `apply` function, rather than first building a `helloworld` image
and then running

```bash
$ tesseract run helloworld apply '{"inputs": {"name": "Tessie"}}'
```

you can just call in your environment the following:
```
tesseract-runtime apply '{"inputs": {"name": "Tessie"}}'
```

More info on usage is contained in `tesseract-runtime --help` (and in its subcommands,
like `tesseract-runtime apply --help`).


## Creating a Tesseract from a Python package

Sometimes it is useful to create a Tesseract from an already-existing
Python package. In order to do so, you can run `tesseract init` in the root folder of
your package (i.e., where `setup.py` and `requirements.txt` would be). Import your package
as needed in `tesseract_api.py`, and specify the dependencies you need at runtime in
`tesseract_requirements.py`.

[^1]: There are several reasons for this. We wanted to make Tesseracts friendly to
people unfamiliar with Docker, and this default reduces the chance of them accidentally filling their
hard drives with data. Also, in a typical SciML setting, cache gets
invalidated quite fast, so a lot of the data which would be saved to the disk
would not really help with build times.
