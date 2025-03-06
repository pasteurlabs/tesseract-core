<img src="docs/static/logo-transparent.png" width="128" align="right">

### Tesseract Core

Autodiff-native, self-documenting software components for Simulation Intelligence. :package:

[Read the docs](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/) |
[Report an issue](https://github.com/pasteurlabs/tesseract-core/issues) |
[Talk to the community](https://si-tesseract.discourse.group/) |
[Contribute](CONTRIBUTING.md)

---

**Tesseract Core** bundles:

1. Tools to define, create, and run Tesseracts, via the `tesseract` CLI and `tesseract_core` Python API.
2. The Tesseract Runtime, a lightweight, high-performance execution environment for Tesseracts.

## What is a Tesseract?

Tesseracts are components that expose experimental, research-grade software to the world. They are self-contained, self-documenting, and self-executing, via command line and HTTP. They are designed to be easy to create, easy to use, and easy to share, including in a production environment. This repository contains all you need to define your own and execute them.

Tesseracts provide built-in support for [differentiable programming](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/differentiable-programming.html) by propagating gradient information at the level of individual components, making it easy to build complex, diverse software pipelines that can be optimized end-to-end.

## Quick start

> [!NOTE]
> Before proceeding, make sure you have a [working installation of Docker](https://docs.docker.com/engine/install/) and a modern Python installation (Python 3.10+).

1. Install Tesseract Core:

   ```bash
   $ pip install tesseract-core
   ```

2. Build an example Tesseract:

   ```bash
   $ tesseract build examples/vectoradd --tag 0.1.0
   ```

3. Display its API documentation:

   ```bash
   $ tesseract apidoc vectoradd:0.1.0
   ```

<p align="center">
<img src="docs/img/apidoc-screenshot.png" width="600">
</p>

4. Run the Tesseract:

   ```bash
   $ tesseract run vectoradd:0.1.0 apply '{"inputs": {"a": [1], "b": [2]}}'
   {"result":{"object_type":"array","shape":[1],"dtype":"float64","data":{"buffer":[3.0],"encoding":"json"}}}⏎
   ```

> [!TIP]
> Now you're ready to dive into the [documentation](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/) for more information on
[installation](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/installation.html),
[creating Tesseracts](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/create.html), and
[invoking them](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/using-tesseracts/use.html).

## License

Tesseract Core is licensed under the [Apache License 2.0](LICENSE) and is free to use, modify, and distribute (under the terms of the license).

Tesseract is a registered trademark of Pasteur Labs, Inc. and may not be used without permission.
