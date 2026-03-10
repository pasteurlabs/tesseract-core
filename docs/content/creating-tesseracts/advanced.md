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

The `build_config` section of `tesseract_config.yaml` controls how the Tesseract image is built. See also the [full config schema](../api/config.md).

### `base_image`

The default base image is `debian:bookworm-slim`. You can specify a different one if you need a specific Python version or preinstalled dependencies. The image must be Ubuntu- or Debian-based.

```yaml
build_config:
  base_image: "python:3.11-slim-bookworm"
```

### `target_platform`

The default target architecture is "native" (same as the host). To cross-compile:

```yaml
build_config:
  target_platform: "linux/arm64"
```

### `extra_packages`

System dependencies not covered by `tesseract_requirements.txt` can be installed via `apt-get`:

```yaml
build_config:
  extra_packages:
    - libgomp1
    - gfortran
```

### `package_data`

Copy static files (model weights, lookup tables, etc.) into the Tesseract image:

```yaml
build_config:
  package_data:
    - "weights/*.pt"
    - "data/config.json"
```

### `custom_build_steps`

Arbitrary Dockerfile commands for anything the above options don't cover. See the [Dockerfile template](https://github.com/pasteurlabs/tesseract-core/blob/main/tesseract_core/sdk/templates/Dockerfile.base) for where these are injected.

```yaml
build_config:
  custom_build_steps:
    - "RUN apt-get update && apt-get install -y cmake"
```

## Creating a Tesseract from a Python package

Sometimes it is useful to create a Tesseract from an already-existing
Python package. In order to do so, you can run `tesseract init` in the root folder of
your package (i.e., where `setup.py` and `requirements.txt` would be). Import your package
as needed in `tesseract_api.py`, and specify the dependencies you need at runtime in
`tesseract_requirements.txt`.
