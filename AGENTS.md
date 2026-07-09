# Notes for AI Agents

This file contains counter-intuitive rules and aspects of the Tesseract codebase that AI agents should be aware of.

## General principles

- **Read the actual code before proposing an architecture or approach.** Do not speculate about capabilities or patterns.
- **Reproduce the problem before trying to solve it.** If a user reports a bug, try to reproduce it locally before proposing a fix. If you have a hypothesis regarding a root cause, verify it. If that's not possible, communicate it clearly.

## Environment setup

- **Use `uv` for dependency management.** Install with `uv pip install -e ".[dev]"`. This is faster and more reliable than plain pip.
- **Run `pre-commit install`** after cloning to set up git hooks.

## Testing

- **Prefer end-to-end tests over unit tests.** Tests that build and run real Tesseracts catch more bugs than mocked unit tests.
- **Avoid mocks.** If you need complex mocking, write an end-to-end test instead.
- **Don't test implementation details.** Tests should verify behavior, not internal structure.
- **Be mindful of slow tests.** End-to-end tests are slow. Check if an existing test can be extended before adding a new one, especially if adding new Docker images to build.
- **Fast tests vs end-to-end tests.** Tests in `tests/endtoend_tests/` require Docker and are skipped automatically when Docker isn't available (or via `--skip-endtoend`). Other tests in `tests/` are fast and don't require Docker. Use fast tests for quick iteration during development.
- **Don't add mocks for Docker.** Tests that need Docker should be marked as end-to-end tests and skipped in fast test runs.
- **Rarely test exceptions.** Only test exception handling when control flow is complex or the error message is critical for UX. Don't write tests that just verify an exception is raised.
- **Never skip or disable tests without asking.** If a test is failing and you want to skip it, ask the user first. Don't add `@pytest.skip`, `@pytest.mark.xfail`, or comment out tests without explicit approval.
- **Always run appropriate tests and verify code you touched works end-to-end before presenting it as complete.** Do not wait for the user to ask 'did you test this?'

## Code style

- **Follow existing patterns.** Look at similar code in the codebase and match its style. Don't introduce new patterns without good reason.
- **Use pre-commit.** Run `pre-commit run --all-files` before committing. Hooks include Ruff for linting/formatting. If linting is failing, run via `pre-commit run --all-files`, not `ruff --fix` or similar, to ensure all hooks run.
- **Follow conventional commits.** PR titles must follow the format: `type[(scope)]: description` (e.g., `feat(sdk): add new feature`).

## Demos

- **Each demo lives in `demo/<name>/` with a `demo.ipynb` notebook** plus its Tesseract source, `requirements.txt`, and any companion images. The notebook is the published artifact; the rest supports it.
- **The docs build flattens demos.** `docs/conf.py` copies `demo/<name>/demo.ipynb` to `docs/content/demo/<name>.ipynb` and copies companion images (`.png/.gif/.jpg/.svg`) flat into `docs/content/demo/`. So inside a notebook, reference images and sibling demos by bare filename (`illustration.png`, `cfd-optimization.ipynb`), **not** by relative path (`../cfd-optimization/demo.ipynb`) — the latter works on disk but breaks in the rendered docs.
- **Register new demos in the toctree.** Add the flattened name (`<name>.ipynb`) to the `{toctree}` and a grid card in `docs/content/demo/demo.md`.
- **Notebooks are not executed at build time** (`nb_execution_mode = "off"`). Outputs are taken as committed, so run the notebook and save its outputs before committing — stale or missing outputs ship as-is.
- **Follow the shared demo template.** Notebooks open with a `# Title`, an "In this tutorial, you will learn how to:" numbered list, and a `## Context` section; progress through numbered `## Step N` sections; and close with `## Takeaways` (numbered, bolded lessons), a `### What's next` list, and a link to the [community forum](https://si-tesseract.discourse.group/). Keep the tone instructional and technical — confident/marketing framing belongs in the accompanying blog or forum showcase post, not the demo.

## Architecture

- **The runtime is separate from the CLI.** `tesseract_core.runtime` runs inside containers; `tesseract_core.sdk` and CLI run on the host. They don't share code or dependencies.

## Ecosystem

Each of these is a separate repository/Python package.

- **Tesseract Core** is the main codebase that defines the Tesseract specification, the Python SDK for defining and building Tesseracts, and the runtime for executing Tesseracts in containers.
- **Tesseract-JAX** is a mature package that supports full integration of Tesseract calls into JAX programs, including JIT compilation and automatic differentiation of code that mixes Tesseract calls and JAX operations.
- **Tesseract-Torch** is the PyTorch counterpart to Tesseract-JAX: it embeds Tesseract calls as PyTorch operators so that `torch.autograd` flows through code that mixes Tesseract calls and PyTorch operations.
- **Tesseract-Streamlit** provides tools to auto-generate Streamlit apps from (externally running / locally built) Tesseracts. It can be used to quickly create interactive demos for Tesseracts and custom visualization without writing any Streamlit code, but is limited to forward application (`apply`).
