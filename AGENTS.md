# Notes for AI Agents

This file contains counter-intuitive aspects of the Tesseract codebase that AI agents should know.

## General Principles

- **Read the codebase before proposing an architecture or approach.** Do not speculate about capabilities or patterns - look at the actual code.

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

## Code Quality

- **Always run tests and verify code works end-to-end before presenting it as complete.** Do not wait for the user to ask 'did you test this?'

## Code style

- **Follow existing patterns.** Look at similar code in the codebase and match its style. Don't introduce new patterns without good reason.
- **Use pre-commit.** Run `pre-commit run --all-files` before committing. Hooks include Ruff for linting/formatting.
- **Follow conventional commits.** PR titles must follow the format: `type[(scope)]: description` (e.g., `feat(sdk): add new feature`).

## Architecture

- **The runtime is separate from the CLI.** `tesseract_core.runtime` runs inside containers; `tesseract_core.sdk` and CLI run on the host.
