(tr-api)=

# Python API (`tesseract_core`)

```{eval-rst}
.. automodule:: tesseract_core
   :members:
```

## Schema introspection (`tesseract_core.sdk.schema_tree`)

Clients only receive a Tesseract's OpenAPI schema, not the pydantic models
behind it. These helpers turn that schema into a tree of typed nodes, so a
client can e.g. find where a payload's arrays are and what shape and dtype they
should have.

```{eval-rst}
.. automodule:: tesseract_core.sdk.schema_tree
   :members:
```
