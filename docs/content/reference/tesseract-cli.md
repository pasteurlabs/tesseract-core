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
`tesseract_config.yaml` on first use and reused while its requirements are unchanged. Building it
needs `uv`, or `conda` for a Tesseract with `requirements.provider: conda`.

```bash
$ tesseract run examples/helloworld apply '{"inputs": {"name": "Osborne"}}'
{"greeting":"Hello Osborne!"}
```

`check-gradients` logs a summary of the checks and exits with 1 if any of them fails:

```bash
$ tesseract run examples/vectoradd check-gradients @examples/vectoradd/example_inputs.json
```

`tesseract serve` prints the URL it serves on, and stays in the foreground until you press Ctrl+C:

```bash
$ tesseract serve examples/helloworld --port 8000
http://127.0.0.1:8000
```

- An argument that is a directory containing `tesseract_api.py`, or a Python file, is a source
  path. Anything else is an image name, except that an argument starting with `.` or `/`, or
  ending with `/`, is always a path, so a mistyped one is reported as such. A source path wins over
  an image of the same name, and the log says which source runs. Pass `helloworld:latest` (or any
  `name:tag`) to run the image instead.
- There is no container to `tesseract teardown` for a source path, and `tesseract ps` does not
  list it.
- `--python-executable` runs the Tesseract on an interpreter you choose instead of the managed
  environment. That interpreter must have `tesseract-core[runtime]` and the Tesseract's
  requirements installed.
- Options that only apply to containers are refused, and say so in `--help`: `--volume`, `--gpus`,
  `--network`, `--user` and `--memory`, plus `--docker-args` for `tesseract run`, and
  `--network-alias` and `--runtime-args` (which passes arguments to Docker there) for
  `tesseract serve`. Everything else works as it does for an image, including `--env`,
  `--input-path`, `--output-path`, `--output-format`, `--port` and `--debug`. The process inherits
  your environment, working directory and user.
