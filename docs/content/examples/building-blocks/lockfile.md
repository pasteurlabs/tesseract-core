# Installing dependencies from a lockfile

{gh-tree}`View on GitHub <examples/pylock>`

## Context

Instead of a flat `tesseract_requirements.txt`, a Tesseract can install its
dependencies from a [PEP 751](https://packaging.python.org/en/latest/specifications/pylock-toml/)
lockfile (`pylock.toml`). A lockfile pins every dependency's exact version, source
index, and artifact hashes, so installing it performs no dependency resolution.
The lockfile is used both for the environment `Tesseract.from_source` builds and
for container images built with `tesseract build`.

Point the Tesseract at a lockfile with `requirements_file` in `tesseract_config.yaml`:

```{literalinclude} ../../../../examples/pylock/tesseract_config.yaml
:language: yaml
```

Any PEP 751 filename is accepted, either `pylock.toml` or a named variant like
`pylock.prod.toml`. The format is inferred from the name.

If you already have a `pyproject.toml` (or a `uv.lock`), export a lockfile with
[`uv export`](https://docs.astral.sh/uv/reference/cli/#uv-export):

```bash
$ uv export --format pylock.toml -o pylock.toml
```

## Example Tesseract

The example computes a sum with `torch` and passes it through `numpy`, so both
packages from the committed lockfile must be installed and importable for `apply`
to succeed:

```{literalinclude} ../../../../examples/pylock/tesseract_api.py
:language: python
```

Then run it as usual:

```python
>>> from tesseract_core import Tesseract
>>> with Tesseract.from_source("examples/pylock/tesseract_api.py") as t:
...     print(t.apply({"a": 2.0, "b": 3.0}))
{'result': 5.0}
```

Or package it as a container image:

```bash
$ tesseract build examples/pylock
$ tesseract run pylock apply '{"inputs": {"a": 2.0, "b": 3.0}}'
{"result":5.0}
```

## The Tesseract runtime is installed on top

```{warning}
The lockfile is installed first, then the Tesseract runtime and its
dependencies are installed into the same environment. If a runtime dependency
requires a version outside the range your lockfile pins, that package is adjusted
to satisfy the runtime, so a handful of packages in the final environment may
differ from the lockfile. Everything else is installed exactly as pinned. Keep
this in mind if you rely on the environment or image matching the lockfile
byte-for-byte.
```

## Private and multi-index lockfiles

A lockfile records each package's index location and hashes, but never
credentials. To install from an authenticated index (a private PyPI, an Azure
Artifacts feed, etc.) in a container build, declare the host under
`build_config.host_credentials` and supply the token at build time with
`tesseract build --secret`. `Tesseract.from_source` ignores
`host_credentials` and installs with uv using your own credentials (for example
`~/.netrc`, a keyring, or `UV_INDEX_*` environment variables).
