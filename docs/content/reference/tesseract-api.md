(tr-api)=

# Python API (`tesseract_core`)

```{eval-rst}
.. automodule:: tesseract_core
   :members:
   :exclude-members: parse_schema, schema_map, schema_map_with_path
```

## Schema introspection (`tesseract_core.sdk.schema_tree`)

Clients only receive a Tesseract's OpenAPI schema, not the pydantic models
behind it. These helpers turn that schema into a tree of typed nodes, so a
client can find where a payload's arrays are and what shape and dtype they
should have. `parse_schema`, `schema_map` and `schema_map_with_path` can also
be imported from `tesseract_core` directly.

```{eval-rst}
.. automodule:: tesseract_core.sdk.schema_tree
   :members:
```
