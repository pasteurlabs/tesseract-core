# Array Encodings Reference

Tesseract supports three encoding formats for array data. The encoding determines how numeric arrays are represented in the JSON payload exchanged between client and server, for both inputs and outputs.

## Available formats

````{tab-set}
:sync-group: encoding-format

```{tab-item} json
:sync: json

Arrays are serialized as nested JSON lists. Human-readable but slow and memory-intensive for large arrays.

    {
      "object_type": "array",
      "shape": [3],
      "dtype": "float64",
      "data": [1.0, 2.0, 3.0]
    }

```

```{tab-item} base64
:sync: base64

Binary array data is base64-encoded and embedded in JSON. Good balance of efficiency and portability.

    {
      "object_type": "array",
      "shape": [3],
      "dtype": "float64",
      "data": {
        "buffer": "AAAAAAAA8D8AAAAAAAAAQAAAAAAAAAhA",
        "encoding": "base64"
      }
    }

```

```{tab-item} binref
:sync: binref

Array data is stored in separate binary files, with JSON containing only references. Most efficient for large data.

    {
      "object_type": "array",
      "shape": [1000000],
      "dtype": "float64",
      "data": {
        "buffer": "arrays/output_0.bin:0",
        "encoding": "binref"
      }
    }

```
````

## Which format should I use?

| Format     | Description                                   | Best For                                                                                         |
| ---------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **json**   | Arrays encoded as nested JSON lists           | Debugging, human-readable output. Avoid for large arrays                                         |
| **base64** | Binary data encoded as base64 strings in JSON | General-purpose default for HTTP transport                                                       |
| **binref** | References to binary files on disk            | Large arrays (>10MB), when disk I/O is preferable over HTTP, when data is written to disk anyway |

The chart below shows how encoding format affects serialization and transfer overhead as array size grows. For more on overall Tesseract performance trade-offs, see {doc}`/content/concepts/performance`.

```{figure} /img/benchmark_encoding.png
:alt: Encoding performance comparison
:width: 80%

Serialization and transfer overhead by encoding format and array size.
```

## Usage

The default format is `json`. To use a different encoding, set the format flag. The CLI examples below call `tesseract-runtime` with `TESSERACT_API_PATH` pointing at the Tesseract:

```bash
$ export TESSERACT_API_PATH=examples/vectoradd/tesseract_api.py
```

`tesseract run <image-name>` accepts the same options for a Tesseract packaged as a container image (`-f` is short for `--output-format`, `-o` for `--output-path`).

### base64

::::{tab-set}
:::{tab-item} CLI
:sync: cli

```bash
$ tesseract-runtime --output-format json+base64 apply @examples/vectoradd/example_inputs_b64.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":"AAAAAAAALEAAAAAAAAA2QAAAAAAAAD5A","encoding":"base64"}}}
```

:::
:::{tab-item} REST API
:sync: http

```bash
$ curl \
  -H "Accept: application/json+base64" \
  -H "Content-Type: application/json" \
  -d @examples/vectoradd/example_inputs_b64.json \
  http://<tesseract-address>:<port>/apply
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":"AAAAAAAALEAAAAAAAAA2QAAAAAAAAD5A","encoding":"base64"}}}
```

:::
::::

### binref

The `json+binref` format stores array data in separate `.bin` files and puts only references in the JSON. This enables lazy loading via [LazySequence](#tesseract_core.runtime.experimental.LazySequence). See the [`Array` docstring](#tesseract_core.runtime.Array) for more details.

The `.bin` files are written to the output path set with `--output-path`, and the references in the JSON are relative to it. `tesseract-runtime` uses the current working directory if no output path is set. For a containerized Tesseract, `tesseract run` and `tesseract serve` reject `json+binref` unless `--output-path` is set, since the `.bin` files would otherwise be written inside the container and lost on teardown.

::::{tab-set}
:::{tab-item} CLI
:sync: cli

```bash
$ tesseract-runtime --output-format json+binref --output-path /tmp/output apply @examples/vectoradd/example_inputs.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":"7796fb36-849a-42ce-8288-a07426111f0c.bin:0","encoding":"binref"}}}

$ ls /tmp/output
7796fb36-849a-42ce-8288-a07426111f0c.bin logs
```

:::
:::{tab-item} REST API
:sync: http

```bash
$ tesseract-runtime --output-path /tmp/output serve
$ curl \
  -H "Accept: application/json+binref" \
  -H "Content-Type: application/json" \
  -d @examples/vectoradd/example_inputs.json \
  http://<tesseract-address>:<port>/apply
```

The `.bin` file references are relative to the `--output-path`. When serving a container image with `tesseract serve`, pass `--output-path` so the `.bin` files are accessible on the host.
:::
::::

```{tip}
When the client and a served Tesseract share a machine, placing the binref
input/output directories on a shared-memory `tmpfs` (`/dev/shm`) lets large
arrays move between them without base64-encoding into the HTTP body. See
{doc}`/content/how-to/fast-local-runs`.
```

### lz4 compression

Set the `TESSERACT_COMPRESSION=lz4` config variable to compress arrays in the output by default (served Tesseracts can also [compress per request](#choosing-the-encoding-per-request)). This applies to both `json+binref` and `json+base64` formats. For binref, each array is compressed individually, preserving offset-based random access, with the compressed size embedded in the buffer path (`<file>:<offset>:<compressed_size>`).

`TESSERACT_COMPRESSION` only affects how a Tesseract encodes its _output_; inputs are always decoded according to their own per-array `compression` field, so no configuration is needed to read compressed data.

```bash
$ TESSERACT_COMPRESSION=lz4 tesseract-runtime --output-format json+binref --output-path /tmp/output apply @examples/vectoradd/example_inputs.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":"....bin:0:44","encoding":"binref","compression":"lz4"}}}
```

With `tesseract run`, forward the variable into the container with `-e` (a plain shell prefix would stay on the host):

```bash
$ tesseract run -e TESSERACT_COMPRESSION=lz4 <image-name> apply -f json+binref -o /tmp/output @examples/vectoradd/example_inputs.json
```

(choosing-the-encoding-per-request)=

## Choosing the encoding per request

For a served Tesseract, `--output-format` and `TESSERACT_COMPRESSION` only set defaults. A client can request a different encoding for each call through the `Accept` header, where the media type selects the output format and the `compression` parameter selects the compression.

```bash
$ curl \
  -H "Accept: application/json+base64; compression=lz4" \
  -H "Content-Type: application/json" \
  -d @examples/vectoradd/example_inputs.json \
  http://<tesseract-address>:<port>/apply
```

Anything the header leaves out falls back to the server's defaults. If the server cannot produce any encoding the header accepts, it responds with `406 Not Acceptable` without running the endpoint and lists what it supports. The same values are advertised in the server's OpenAPI schema (`/openapi.json`) as `x-supported-output-formats`, `x-supported-compressions` and `x-supported-gpu-transports`.

The experimental GPU transports (the `gpu_transport` parameter) must also be enabled on the server, and are only used for requests that ask for them.

In the Python SDK, `Tesseract.server_capabilities` lists the output formats, compressions and GPU transports the server accepts, and `Tesseract.with_encoding` returns a view of the Tesseract that requests a different encoding for every call made through it. To query the encoding requested by a given Tesseract object or view, use `Tesseract.current_encoding`:

```python
from tesseract_core import Tesseract

with Tesseract.from_source("examples/vectoradd/tesseract_api.py") as tess:
    print(tess.server_capabilities.compressions)  # ('none', 'lz4')
    print(tess.current_encoding)
    # RequestedEncoding(output_format='json+base64', gpu_transport=None, compression=None)
    view = tess.with_encoding(compression="lz4")
    print(view.current_encoding)
    # RequestedEncoding(output_format='json+base64', gpu_transport=None, compression='lz4')
    result = view.apply(inputs)
```

The SDK checks each call against what the server advertises before sending it. Runtimes older than 1.13 advertise nothing and cannot parse `Accept` parameters, so requesting `compression` or a GPU transport other than `none` from them raises an error. Runtimes 1.13 and 1.14 identify GPUs in a format the SDK no longer reads, so requesting a GPU transport other than `none` from them raises an error too.

A GPU transport the server offers may still not work from a given client. `cuda_ipc`, for example, needs both processes on one host with the same GPU visible to both (they may number it differently, such as when a container is given only one of the host's GPUs). `Tesseract.resolve_gpu_transport` checks whether a transport works by exchanging a small GPU array with the server, once per connection, and returns the transport to use for GPU arrays. If the Tesseract was created or viewed with a specific `gpu_transport`, that transport is returned if it works, and an error is raised if it does not. Otherwise, it returns the first transport the server offers that works, or `none` with a warning if none of them do. Tesseract-JAX and Tesseract-Torch use this to keep GPU arrays on the device whenever possible, and custom code handling GPU arrays can do the same:

```python
transport = tess.resolve_gpu_transport()
result = tess.with_encoding(gpu_transport=transport).apply(gpu_inputs)
```

With a GPU transport other than `none`, a response hands the client references to the server's GPU memory, which the server keeps until the client has copied the arrays out. The server names each such response in a `Tesseract-Exports` response header, and the Python SDK names the responses it has decoded in a `Tesseract-Exports-Done` header on its next request (or when it closes), so any number of clients and threads can share one server. Each request from a raw HTTP client that never sends `Tesseract-Exports-Done` releases the GPU memory of earlier responses to all such clients, which is only safe for a single such client making one request at a time. Sending the header, empty until there is something to name, avoids that. Memory no client names is released after 5 minutes.
