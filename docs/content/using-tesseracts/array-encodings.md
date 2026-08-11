# Array Encodings

Tesseract supports four encoding formats for array data. The encoding determines how numeric arrays are represented in the JSON payload exchanged between client and server, for both inputs and outputs.

The first three (`json`, `base64`, `binref`) move the array *bytes* through the payload. The fourth, `cuda_ipc`, keeps GPU arrays **on the device** and moves only a small handle to their GPU memory, so no array data crosses the CPU. It is experimental and has additional requirements (see [cuda_ipc](#cuda-ipc) below).

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

```{tab-item} cuda_ipc
:sync: cuda_ipc

The array stays in GPU memory; the payload carries only a CUDA IPC handle to it (plus the byte offset/size within the backing allocation). No array bytes touch the CPU. Experimental; requires a shared host GPU (see below).

    {
      "object_type": "array",
      "shape": [1000000],
      "dtype": "float32",
      "data": {
        "handle": "<base64-encoded 64-byte cudaIpcMemHandle_t>",
        "device": 0,
        "storage_offset": 0,
        "storage_size": 4194304,
        "encoding": "cuda_ipc"
      }
    }

```
````

## Which format should I use?

| Format       | Description                                   | Best For                                                                                         |
| ------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **json**     | Arrays encoded as nested JSON lists           | Debugging, human-readable output. Avoid for large arrays                                         |
| **base64**   | Binary data encoded as base64 strings in JSON | General-purpose default for HTTP transport                                                       |
| **binref**   | References to binary files on disk            | Large arrays (>10MB), when disk I/O is preferable over HTTP, when data is written to disk anyway |
| **cuda_ipc** | CUDA IPC handle to on-GPU memory (experimental) | GPU→GPU transfer between a client and a Tesseract that share a physical GPU on the same host, e.g. optimization/MCMC loops that would otherwise bounce arrays through the CPU |

The chart below shows how encoding format affects serialization and transfer overhead as array size grows. For more on overall Tesseract performance trade-offs, see {doc}`/content/misc/performance`.

```{figure} /img/benchmark_encoding.png
:alt: Encoding performance comparison
:width: 80%

Serialization and transfer overhead by encoding format and array size.
```

## Usage

The default format is `json`. To use a different encoding, set the format flag:

### base64

::::{tab-set}
:::{tab-item} CLI
:sync: cli

```bash
$ tesseract run vectoradd apply -f "json+base64" @examples/vectoradd/example_inputs_b64.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":"AAAAAAAALEAAAAAAAAA2QAAAAAAAAD5A","encoding":"base64"}}}
```

:::
:::{tab-item} REST API
:sync: http

```bash
$ curl \
  -H "Accept: application/json+base64" \
  -H "Content-Type: application/json" \
  -d @examples/vectoradd/example_inputs.json \
  http://<tesseract-address>:<port>/apply
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":"AAAAAAAALEAAAAAAAAA2QAAAAAAAAD5A","encoding":"base64"}}}
```

:::
::::

### binref

The `json+binref` format stores array data in separate `.bin` files and puts only references in the JSON. This enables lazy loading via [LazySequence](#tesseract_core.runtime.experimental.LazySequence). See the [`Array` docstring](#tesseract_core.runtime.Array) for more details.

::::{tab-set}
:::{tab-item} CLI
:sync: cli

```bash
$ tesseract run vectoradd apply -f "json+binref" -o /tmp/output @examples/vectoradd/example_inputs.json

$ ls /tmp/output
7796fb36-849a-42ce-8288-a07426111f0c.bin results.json

$ cat /tmp/output/results.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":"7796fb36-849a-42ce-8288-a07426111f0c.bin:0","encoding":"binref"}}}
```

:::
:::{tab-item} REST API
:sync: http

Specify `--output-path` when serving so the `.bin` files are accessible on the host. Otherwise they're only available inside the container (under `/tesseract/output_path`).

```bash
$ tesseract serve <tesseract-name> --output-path /tmp/output
$ curl \
  -H "Accept: application/json+binref" \
  -H "Content-Type: application/json" \
  -d @examples/vectoradd/example_inputs.json \
  http://<tesseract-address>:<port>/apply
```

The `.bin` file references are relative to the `--output-path`.
:::
::::

(cuda-ipc)=
### cuda_ipc

```{warning}
`json+cuda_ipc` is **experimental**. It keeps GPU arrays on the device and passes only a CUDA IPC handle, avoiding any CPU round-trip. This is dramatically faster for large GPU arrays (the transfer cost is flat regardless of array size), but only works under the specific conditions below.
```

**Requirements**

- The client and the Tesseract must run on the **same physical host** and see the **same GPU**. IPC handles are meaningless on another host or a different device, so this format is unusable for remote Tesseracts or serialize-to-disk workflows.
- Both processes must share an IPC namespace. When serving via the SDK with `json+cuda_ipc`, the container is automatically started with `--ipc=host`; you must additionally grant GPU access (e.g. `gpus=["all"]`).
- **Encoding** works with any array exposing `__cuda_array_interface__` (CuPy, PyTorch, JAX, Numba). **Decoding** requires [CuPy](https://cupy.dev/) and returns a `cupy.ndarray` (convertible to PyTorch/JAX via DLPack or `__cuda_array_interface__`). CPU (NumPy) arrays in a `cuda_ipc` payload transparently fall back to base64.

**Lifetime model (important)**

Because CUDA memory is not garbage-collected across processes, `cuda_ipc` relies on a simple, explicit ownership contract. It is correct **only** under these assumptions:

1. **Serial requests.** A client must not have more than one `cuda_ipc` request in flight at a time. The server keeps each request's exported GPU buffers alive only until the *next* request arrives (a buffer "ring" of depth 1), then releases them. A serial client has always finished consuming request *N* before it issues request *N+1*, so this is safe; concurrent requests are not supported.
2. **Copy on decode.** Decoding copies the array into client-owned GPU memory and closes the IPC mapping before returning. The returned `cupy.ndarray` is therefore fully owned by the client and stays valid even after the server reuses or frees the original buffer. This costs one on-GPU copy, still far cheaper than a host round-trip.

You do not need to manage any of this manually; the SDK client and the runtime server implement both halves of the contract. Just keep requests serial.

`cuda_ipc` is available through the **Python SDK only**, since decoding a handle requires GPU-aware client code (CuPy). It is listed as a recognized format by `tesseract run`/`tesseract serve`, but passing it on the command line raises an error directing you to the SDK.

```python
import numpy as np
import cupy
from tesseract_core import Tesseract

with Tesseract.from_image(
    "my_gpu_tesseract",
    gpus=["all"],                    # GPU access is required
    output_format="json+cuda_ipc",   # --ipc=host is added automatically
) as t:
    result = t.apply({"x": np.arange(1_000_000, dtype=np.float32)})
    y = result["y"]                  # a cupy.ndarray, still on the GPU
    assert isinstance(y, cupy.ndarray)
```
