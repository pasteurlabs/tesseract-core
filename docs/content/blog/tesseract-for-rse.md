---
html_class: blog-page
og:title: Tesseract for Research Software Engineers
og:description: "A standard contract for packaging, sharing, and composing scientific software across research groups — typed interfaces, reproducibility, and multi-group solver coupling."
---

# Tesseract for Research Software Engineers

_Mar 31, 2026_ · The Tesseract Team

Tesseract gives RSEs a standard contract for packaging, sharing, and composing scientific software across research groups.

## Why Tesseract?

- **Typed interface contracts.** Every Tesseract self-documents its inputs, outputs, and schemas. Users can inspect the interface without reading source code.
- **Reproducibility by default.** Containers pin all dependencies. The same Tesseract produces the same results on any machine.
- **Multi-group solver coupling.** Connect components from different teams into a single pipeline via CLI, REST, or Python SDK -- no glue code wars.
- **Lower maintenance burden.** Ship a Tesseract instead of a README with 30 setup steps. Users run `tesseract serve` and call the API.

## Get started

- [Installation](../introduction/installation.md)
- [Creating Tesseracts](../creating-tesseracts/create.md)
- [Design patterns](../creating-tesseracts/design-patterns.md)
- [Deployment guide](../creating-tesseracts/deploy.md)
