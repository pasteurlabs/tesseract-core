# Performance Trade-offs & Optimization

This page helps you understand the performance characteristics of Tesseract and determine whether it's a good fit for your workload.

## Overview

Using Tesseracts adds overhead to your computations through:

1. **Container startup** - One-time cost when starting a containerized Tesseract (~2s)
2. **Data transfer** - Moving data between client and server (depends on data size and network bandwidth)
3. **Data serialization** - Encoding arrays for transport (depends on data size and encoding format)
4. **HTTP communication** - Request/response handling (~6ms when served locally)
5. **Framework overhead** - Pydantic validation, schema processing (~0.5ms)

Which of these dominates depends on your workload. For small to medium arrays (up to ~100K elements), serialization is sub-millisecond and the interaction mode (HTTP roundtrip, container startup) is the main cost. For large arrays, encoding format becomes critical: it determines both serialization time and the volume of data transferred. In general, data transfer costs more than serialization — choosing a compact encoding (like base64 or binref over JSON) helps primarily by reducing the amount of data sent over the wire.

For many scientific computing workloads where computations take seconds, minutes, or hours, the total overhead is negligible.

```{note}
**Tesseract is not a high-performance RPC framework.** If your workload requires microsecond latency or millions of calls per second, Tesseract is not the right tool. However, for workloads with significant computation time or I/O, Tesseract's overhead is typically a small fraction of total runtime, and the benefits of isolation, reproducibility, and flexibility typically outweigh the costs.
```

## Is Tesseract right for my workload?

```{warning}
Performance depends heavily on your specific workload, data sizes, hardware, and interaction patterns. Benchmarking with representative inputs is the best way to understand the trade-offs for your use case. The guidance in this section is based on general principles and typical scenarios, but your mileage may vary.
```

<br>

```{figure} /img/benchmark_guidance.png
:alt: Tesseract overhead guidance chart
:width: 100%

Overhead as percentage of computation time, depending on interaction mode and I/O data size.
```

<br>

The guidance plot shows overhead curves for three interaction modes at three representative I/O sizes (1kB, 1MB, 1GB). Each combination represents a typical usage pattern:

| Mode (color)                                   | What it measures                                                                                                            | Typical use case                                       |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **Non-containerized, in-memory** (green)       | Direct Python calls via `Tesseract.from_tesseract_api`. Passes data as in-memory Python objects without Docker or HTTP.     | Development, tight loops, performance-critical paths   |
| **Containerized, json+base64 via HTTP** (blue) | Full Docker + HTTP stack, served via HTTP (e.g. `tesseract serve`). Includes serialization and network transfer.            | Production, CI/CD, multi-language environments         |
| **Containerized, json+binref via CLI** (red)   | CLI invocation via `tesseract run`. Includes container startup and disk I/O, but avoids transferring data over the network. | Shell scripts, one-off runs, pipelines with large data |

The three I/O sizes (dotted = 1kB, dashed = 1MB, solid = 1GB) show how data volume shifts the overhead curve. For small data, the fixed costs (HTTP roundtrip, container startup) dominate. For large data, transfer and serialization take over.

```{figure} /img/benchmark_overhead.png
:alt: Tesseract overhead by interaction mode
:width: 80%

Overhead comparison across interaction modes for different array sizes. Uses a no-op Tesseract that does nothing but decode and encode data, isolating framework overhead.
```

### Rules of thumb by use case

| Scenario                                       | Recommendation                                                                                                                                                                                                                                                       |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tight loops on in-memory data**              | Use {doc}`non-containerized execution </content/using-tesseracts/use>` to bypass all network/container overhead. At ~0.5ms per call, you can run thousands of iterations per second. Requires all dependencies to be available in the same local Python environment. |
| **Second-scale workloads on medium-size data** | The sweet spot for containerized HTTP execution, with low overhead execution benefitting from most Tesseract features.                                                                                                                                               |
| **Cheap operations on small data via HTTP**    | HTTP overhead can dominate when computation is fast. Batch multiple inputs into a single request.                                                                                                                                                                    |
| **Development and debugging**                  | Use non-containerized execution or `tesseract-runtime serve` for fast iteration, then switch to serving via HTTP for final testing.                                                                                                                                  |
| **Shell scripts and one-off runs**             | CLI is convenient but has ~2s overhead per invocation. For multiple calls, keep a container running.                                                                                                                                                                 |
| **Long-running operations on large datasets**  | Use CLI with `json+binref` encoding. The ~2.5s container overhead is negligible for multi-minute runs, and binref allows large arrays to be passed between Tesseracts without expensive copies.                                                                      |
| **Cheap operations on huge datasets**          | Tesseract may not be the right tool for you. Consider whether you can partition your workload into more compute-intensive components, or if a more traditional RPC framework is a better fit.                                                                        |

## Optimizing performance

### 1. Choose the right encoding format

Encoding format affects both serialization time and — more importantly — the volume of data transferred. A 10M-element float64 array is ~80MB as raw binary, ~107MB as base64, and can be several hundred MB as JSON text. Since data transfer typically dominates over serialization, choosing a compact format is the most effective way to reduce overhead for large arrays.

| Format     | Description                                   | Data Size vs Raw Binary | Best For                                                                                         |
| ---------- | --------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------ |
| **base64** | Binary data encoded as base64 strings in JSON | ~1.33x                  | General-purpose default for HTTP transport                                                       |
| **binref** | References to binary files on disk            | 1x (raw binary)         | Large arrays (>10MB), when disk I/O is preferable over HTTP, when data is written to disk anyway |
| **json**   | Arrays encoded as nested JSON lists           | ~3-10x                  | Debugging, human-readable output. Avoid for large arrays                                         |

See also: {doc}`/content/using-tesseracts/array-encodings` for detailed encoding documentation.

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

If you're running a script multiple times against the same Tesseract, consider keeping a container running and connecting via `Tesseract.from_url()`:

```python
# Start once: tesseract serve my-tesseract
tesseract = Tesseract.from_url("http://localhost:8100")
result = tesseract.apply(inputs)
```

### 4. Enable profiling for analysis

Enable profiling to understand where time is spent:

```python
# Non-containerized usage
tesseract = Tesseract.from_tesseract_api(
    "path/to/tesseract_api.py",
    runtime_config={"profiling": True}
)

# Served via HTTP
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
