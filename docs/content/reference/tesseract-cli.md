# CLI Reference (`tesseract`)

```{eval-rst}
.. typer:: tesseract_core.sdk.cli:app
   :prog: tesseract
   :make-sections:
   :show-nested:
   :preferred: svg
   :theme: dark
   :width: 100
```

## Running a Tesseract from source

`tesseract run` and `tesseract serve` take either an image name or the path to a Tesseract's
source: a directory containing `tesseract_api.py`, or the Python file itself. A source path needs
no image and no Docker. The Tesseract runs in a local process, in the environment that
`Tesseract.from_source()` uses: `.tesseract-venv` next to `tesseract_api.py`, built from
`tesseract_config.yaml` on first use and reused while its requirements are unchanged.

```bash
$ tesseract run examples/helloworld apply '{"inputs": {"name": "Osborne"}}'
{"greeting":"Hello Osborne!"}

$ cd examples/vectoradd
$ tesseract run . check-gradients @example_inputs.json
✅ Gradient check for jacobian passed ✅ (0 failures / 857 checks)

$ tesseract serve examples/helloworld --port 8000
```

- An argument that exists on disk and is a directory containing `tesseract_api.py`, or a Python
  file, is a source path. Anything else is an image name. If a directory and a local image share a
  name, the directory is used and a warning says so. Pass `./helloworld` or `helloworld:latest` to
  say which one you mean.
- `tesseract serve` runs a source path in the foreground until you press Ctrl+C. There is no
  container to `tesseract teardown`, and `tesseract ps` does not list it.
- `--python-executable` runs the Tesseract on an interpreter you choose instead of the managed
  environment. That interpreter must have `tesseract-core[runtime]` and the Tesseract's
  requirements installed.
- Options that only apply to containers are refused: `--volume`, `--gpus`, `--network`,
  `--network-alias`, `--user`, `--memory`, `--docker-args` (`tesseract run`) and `--runtime-args`
  (`tesseract serve`, where it passes arguments to Docker). Everything else works as it does for an
  image, including `--env`, `--input-path`, `--output-path`, `--output-format`, `--port` and
  `--debug`. The process inherits your environment, working directory and user.
