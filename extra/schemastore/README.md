# SchemaStore registration for `tesseract_config.yaml`

[SchemaStore](https://www.schemastore.org/) is the registry that most editors
(VS Code, JetBrains, and any editor using the YAML Language Server) consult to
decide which JSON Schema applies to a given file. Registering
`tesseract_config.yaml` there means users get validation and autocompletion with
**zero configuration** — no `# yaml-language-server:` comment required.

This directory stages the artifacts for a one-time pull request to
[`SchemaStore/schemastore`](https://github.com/SchemaStore/schemastore). It is
not shipped with the package.

## What to submit

SchemaStore can point at a schema we host ourselves, so we only register a
catalog entry — the schema content stays served from our docs site at
`https://docs.pasteurlabs.ai/projects/tesseract-core/stable/schema.json`
(regenerated from `TesseractConfig` on every docs build).

### 1. Catalog entry

Add the object in [`catalog-entry.json`](catalog-entry.json) to the `schemas`
array in `src/api/json/catalog.json` (keep the array alphabetically sorted by
`name`).

The `url` points at our self-hosted schema; `fileMatch` uses the exact filename,
which is distinctive enough to avoid the false positives SchemaStore warns about
for generic names like `config.yaml`.

### 2. Test fixtures (recommended, not required)

SchemaStore encourages a positive and a negative test file:

- Copy [`positive.yaml`](positive.yaml) to `src/test/tesseract-config/positive.yaml`.
- Copy [`negative.yaml`](negative.yaml) to `src/negative_test/tesseract-config/negative.yaml`.

The negative fixture reproduces the example from
[issue #459](https://github.com/pasteurlabs/tesseract-core/issues/459): an unknown
top-level key that a correct schema must reject.

## Keeping it in sync

Nothing needs re-submitting per release: the catalog `url` points at the `stable`
docs alias, which always serves the schema for the latest tagged release.
