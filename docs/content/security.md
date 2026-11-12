# Security considerations

## Trust model

- Tesseract code is trusted. Treat a Tesseract like any other code you choose to run.
- Building an untrusted Tesseract is comparable to running `docker build` on an untrusted Dockerfile.
- Running an untrusted image gives the container the host input/output directories, optional GPUs, and `--ipc=host` when a GPU transport is enabled.
- `Tesseract.from_tesseract_api` imports `tesseract_api.py` into the calling process and must be as trusted as any other imported Python code.
- Inputs from clients are untrusted: validate file references under `--input-path`, and bound payload size and nesting.

## Serving over HTTP

`tesseract serve` binds to `127.0.0.1` by default. Setting `--host-ip 0.0.0.0` exposes the API to the network.
The runtime has no authentication, and every client can call every endpoint. Only serve on trusted networks,
or put the API behind an authenticating reverse proxy that you control.

## Reporting a vulnerability

See [SECURITY.md](https://github.com/pasteurlabs/tesseract-core/blob/main/SECURITY.md).
