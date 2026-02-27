# Performance Trade-offs & Optimization

This page helps you understand the performance characteristics of Tesseract and determine whether it's a good fit for your workload.

## Overview

Tesseract adds overhead to your computations through:

1. **Framework overhead** - Pydantic validation, schema processing (~0.5ms)
2. **HTTP communication** - Request/response handling (~6ms when containerized)
3. **Data serialization** - Encoding arrays for transport (varies with data size and encoding method)
4. **Container startup** - One-time cost when starting a containerized Tesseract (~2s)

For many scientific computing workloads where computations take seconds, minutes, or hours, this overhead is negligible.

```{note}
**Tesseract is not a high-performance RPC framework.** If your workload requires microsecond latency or millions of calls per second, Tesseract is not the right tool. However, for workloads with significant computation time or I/O, Tesseract's overhead is typically a small fraction of total runtime, and the benefits of isolation, reproducibility, and flexibility typically outweigh the costs.
```

## Is Tesseract right for my workload?

<br>

```{figure} /img/benchmark_guidance.png
:alt: Tesseract overhead guidance chart
:width: 100%

Guidance for three common scenarios, depending on the size of input/output data and the computation time of your Tesseract. Note that non-containerized execution assumes that data is already in-memory, HTTP-based execution includes serialization overhead on server and client, and containerized execution includes container overhead and disk I/O. This is comparing apples to oranges to bananas - trade-offs typically depend on the specific use case, so we recommend running benchmarks with your actual workload to make an informed decision.
```

<br>

### Rules of thumb by use case

| Scenario                                       | Recommendation                                                                                                                                                                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Second-scale workloads on medium-size data** | The sweet spot for containerized HTTP execution, with low overhead execution benefitting from most Tesseract features.                                                                                                   |
| **Long-running operations on large datasets**  | Use CLI with `json+binref` encoding. The ~2.5s container overhead is negligible for multi-minute runs, and binref allows large arrays to be passed without copies.                                                       |
| **Tight loops on in-memory data**              | Use `from_tesseract_api()` to bypass all network/container overhead. At ~0.5ms per call, you can run thousands of iterations per second. Requires all dependencies to be available in the same local Python environment. |
| **Cheap operations on small data via HTTP**    | HTTP overhead can dominate when computation is fast. Batch multiple inputs into a single request.                                                                                                                        |
| **Development and debugging**                  | Use `from_tesseract_api()` for fast iteration, then switch to containerized execution via `from_image()` for final testing.                                                                                              |
| **Shell scripts and one-off runs**             | CLI is convenient but has ~2s overhead per invocation. For multiple calls, keep a container running.                                                                                                                     |
| **Cheap operations on huge datasets**          | Tesseract may not be the right tool for you. Consider whether you can break your workload into more compute-intensive components, or if a more traditional RPC framework is a better fit.                                |

```{seealso}
See also: {doc}`/content/using-tesseracts/use` for detailed usage examples.
```

## Interaction modes

### Which interaction modes exist?

````{tab-set}
:sync-group: interaction-mode

```{tab-item} from_tesseract_api
:sync: api

The fastest path - bypasses Docker and HTTP entirely, using direct Python calls. Requires all dependencies to be available locally.

    from tesseract_core.sdk.tesseract import Tesseract

    tesseract = Tesseract.from_tesseract_api("path/to/tesseract_api.py")
    result = tesseract.apply({"data": my_array})
```

```{tab-item} from_image (HTTP)
:sync: http

Full containerized execution with HTTP communication. Uses base64 encoding by default.

    from tesseract_core.sdk.tesseract import Tesseract

    with Tesseract.from_image("my-tesseract:latest") as tesseract:
        result = tesseract.apply({"data": my_array})
```

```{tab-item} CLI
:sync: cli

Command-line interface for containerized execution. Supports binref encoding for efficient large data transfer.

    tesseract run my-tesseract apply @payload.json \
        --input-path ./data \
        --output-path ./output \
        --output-format json+binref
```
````

<br>

```{figure} /img/benchmark_overhead.png
:alt: Tesseract overhead by interaction mode
:width: 80%

Overhead comparison across interaction modes for different array sizes. Uses a no-op Tesseract that does nothing but decode and encode data, isolating framework overhead.
```

<br>

### Which one should I use?

| Mode                      | Overhead | Best For                                             | Example                                          |
| ------------------------- | -------- | ---------------------------------------------------- | ------------------------------------------------ |
| **`from_tesseract_api`**  | ~0.5ms   | Development, tight loops, performance-critical paths | `Tesseract.from_tesseract_api("path/to/api.py")` |
| **`from_image` (HTTP)**   | ~6-7ms   | Production, CI/CD, multi-language environments       | `Tesseract.from_image("my-tesseract:latest")`    |
| **CLI (`tesseract run`)** | ~2.5s    | Shell scripts, one-off runs, large data with binref  | `tesseract run my-tesseract apply @input.json`   |

```{seealso}
See also: {doc}`/content/using-tesseracts/use` for detailed usage examples, {doc}`/content/api/tesseract-cli` for CLI reference.
```

## Encoding formats

### What encoding formats are available?

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

### Which encoding format should I use?

Tesseract supports three encoding formats for array data, each with different trade-offs:

| Format     | Description                                   | Overhead                     | Best For                                                                                         |
| ---------- | --------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------ |
| **json**   | Arrays encoded as nested JSON lists           | High (5-10x slower)          | Debugging, human-readable output                                                                 |
| **base64** | Binary data encoded as base64 strings in JSON | Medium                       | HTTP transport, general-purpose default                                                          |
| **binref** | References to binary files on disk            | Low (fastest for large data) | Large arrays (>10MB), when disk I/O is preferable over HTTP, when data is written to disk anyway |

<br>

```{figure} /img/benchmark_encoding.png
:alt: Tesseract overhead by encoding mode
:width: 80%

Overhead comparison across encoding modes for different array sizes.
```

<br>

See also: {doc}`/content/using-tesseracts/array-encodings` for detailed encoding documentation.

## Optimizing performance

### 1. Choose the right interaction mode

Select the interaction mode that best fits your use case:

```python
# Development - fastest iteration
tesseract = Tesseract.from_tesseract_api(my_api_module)

# Production - full isolation
with Tesseract.from_image("my-tesseract:latest") as tesseract:
    result = tesseract.apply(inputs)
```

### 2. Batch small operations

If you have many small operations, batch them into a single request:

```python
# Instead of many small calls
# for item in items:
#     result = tesseract.apply({"data": item})

# Batch into one call
results = tesseract.apply({"data": np.stack(items)})
```

### 3. Reuse Tesseract instances

Container startup is expensive. Reuse instances across calls:

```python
# Good - reuse the context
with Tesseract.from_image("my-tesseract") as tesseract:
    for batch in batches:
        result = tesseract.apply(batch)

# Bad - new container per call
for batch in batches:
    with Tesseract.from_image("my-tesseract") as tesseract:
        result = tesseract.apply(batch)
```

### 4. Use appropriate encoding

For large arrays via CLI, consider binref encoding:

```bash
tesseract run my-tesseract apply @payload.json \
    --input-path ./data \
    --output-path ./output \
    --output-format json+binref
```

See {doc}`/content/using-tesseracts/array-encodings` for more details on encoding options.

### 5. Enable profiling for analysis

Enable profiling to understand where time is spent:

```python
# When using from_tesseract_api
tesseract = Tesseract.from_tesseract_api(
    "path/to/tesseract_api.py",
    runtime_config={"profiling": True}
)

# When using from_image
with Tesseract.from_image("my-tesseract:latest", runtime_config={"profiling": True}) as tesseract:
    result = tesseract.apply(inputs)
```

```bash
# When using CLI
$ tesseract run my-tesseract apply @payload.json \
    --input-path ./data \
    --output-path ./output \
    --output-format json+binref \
    --profiling

# Or via environment variable
TESSERACT_PROFILING=true tesseract-runtime serve
```
