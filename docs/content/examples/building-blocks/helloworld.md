# HelloWorld

{gh-tree}`View on GitHub <examples/helloworld>`

## Context

This is a basic Hello World Tesseract. See [getting-started]
for more details on this Tesseract.

## Example Tesseract (`examples/helloworld`)

```{literalinclude} ../../../../examples/helloworld/tesseract_api.py
:language: python
```

## Running it without an image

`tesseract run` and `tesseract serve` accept the source directory directly. The first call builds
an environment for the Tesseract in `examples/helloworld/.tesseract-venv` with `uv`, and later calls
reuse it. No Docker is needed:

```bash
$ tesseract run examples/helloworld apply '{"inputs": {"name": "Osborne"}}'
{"greeting":"Hello Osborne!"}
```

`tesseract serve examples/helloworld` serves it over HTTP in the foreground until you press Ctrl+C.
To share or deploy it, build an image with `tesseract build examples/helloworld` and pass the image
name to the same commands. See {doc}`/content/reference/tesseract-cli` for the details.
