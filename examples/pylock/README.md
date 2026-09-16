# Build a Tesseract from a lockfile

This Tesseract installs its dependencies from a [PEP 751](https://packaging.python.org/en/latest/specifications/pylock-toml/)
lockfile (`pylock.toml`) instead of a flat `tesseract_requirements.txt`. A
lockfile pins every dependency's exact version, source index, and artifact
hashes, so the build performs no resolution.

Point the build at a lockfile with `requirements_file` (see
[`tesseract_config.yaml`](tesseract_config.yaml)):

```yaml
build_config:
  requirements:
    provider: uv-pip
    requirements_file: pylock.toml
```

Any PEP 751 filename is accepted, either `pylock.toml` or a named variant like
`pylock.prod.toml`. The format is inferred from the name. Then build and run as
usual:

```bash
$ tesseract build examples/pylock
$ tesseract run pylock apply '{"inputs": {"a": 2.0, "b": 3.0}}'
```

## Where the lockfile comes from

The committed `pylock.toml` spans two indexes, PyPI (for `numpy`) and the
PyTorch CPU index (for `torch`), with per-package platform markers.

The dependency set is defined in [`_pyproject.toml`](_pyproject.toml). To
regenerate the lockfile after editing it, run:

```bash
$ ./examples/pylock/regenerate_lockfile.sh
```

The script resolves a lockfile from `_pyproject.toml` and exports it to
`pylock.toml` with `uv export --no-emit-project`, the same command you would use
to turn a [uv workspace lock](https://docs.astral.sh/uv/concepts/projects/sync/#exporting-the-lockfile)
into a standalone `pylock.toml`. Here the root project is an empty stub, so this
drops it and keeps only the third-party dependencies.

## Learn more

See the [Lockfile building block](../../docs/content/examples/building-blocks/lockfile.md)
in the documentation for details, including how the Tesseract runtime is installed
on top of the lockfile and how to install from an authenticated (private) index.
