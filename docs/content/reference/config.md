# Configuration (`tesseract_config.yaml`)

The `tesseract_config.yaml` file contains a Tesseract's metadata, such as its name, description, version, and build configuration.

## Example file

```{literalinclude} ../../../examples/helloworld/tesseract_config.yaml

```

## Editor validation

A [JSON Schema](https://json-schema.org/) for `tesseract_config.yaml` is published at
[`https://docs.pasteurlabs.ai/projects/tesseract-core/stable/schema.json`](https://docs.pasteurlabs.ai/projects/tesseract-core/stable/schema.json)
(`stable` tracks the latest release). Editors that understand it can validate the
file as you type and offer inline completion and help for every field.

`tesseract_config.yaml` is registered with [SchemaStore](https://www.schemastore.org/),
the registry that most editors (VS Code, JetBrains, and any editor backed by the
[YAML Language Server](https://github.com/redhat-developer/yaml-language-server))
consult to decide which schema applies to a file. Because of that, validation and
autocompletion work automatically for any file named `tesseract_config.yaml`, with
no configuration.

To point an editor at a specific version of the schema, or if you renamed the file,
add a `# yaml-language-server:` comment at the top:

```yaml
# yaml-language-server: $schema=https://docs.pasteurlabs.ai/projects/tesseract-core/stable/schema.json
name: "my-tesseract"
```

The reference is a comment, not a config field, so it is ignored when the Tesseract
is built. (A top-level `$schema:` _key_ would be rejected, since unknown fields are
not allowed.)

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
