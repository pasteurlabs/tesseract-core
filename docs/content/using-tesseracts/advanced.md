# Advanced usage

## File system I/O

The `tesseract` command can take care of
passing data from local disk
(or any [fsspec-compatible](https://filesystem-spec.readthedocs.io/en/latest/) resource,
like HTTP, FTP, S3 Buckets, and so on) to a Tesseract via the `@` syntax.

If you want to write the output of a Tesseract to a file,
you can use the `--output-path` parameter, which also supports any
[fsspec-compatible](https://filesystem-spec.readthedocs.io/en/latest/)
target path:

```bash
$ tesseract run vectoradd apply --output-path /tmp/output @inputs.json
```

## Using GPUs

To leverage GPU support in your Tesseract environment, you can specify which NVIDIA GPU(s) to make available
using the `--gpus` argument when running a Tesseract command. This allows you to select specific GPUs or
enable all available GPUs for a task.

To run Tesseract on a specific GPU, provide its index:
```bash
$ tesseract run --gpus 0 helloworld apply '{"inputs": {"name": "Osborne"}}'
```

To make all available GPUs accessible, use the `--gpus all` option:
```bash
$ tesseract run --gpus all helloworld apply '{"inputs": {"name": "Osborne"}}'
```

You can also specify multiple GPUs individually:
```bash
$ tesseract run --gpus 0 --gpus 1 helloworld apply '{"inputs": {"name": "Osborne"}}'
```

The GPUs are indexed starting at zero with the same convention as `nvidia-smi`.
