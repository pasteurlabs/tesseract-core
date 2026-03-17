# Performance Trade-offs & Optimization

## Overview

Using Tesseracts adds overhead to your computations through:

1. **Container startup** (~2s) — One-time cost when starting a containerized Tesseract.
2. **HTTP communication** (~2.5ms) — Request/response handling per call.
3. **Data transfer** — Moving data between client and server (depends on data size and network bandwidth).
4. **Data serialization** — Encoding arrays for transport (depends on data size and encoding format).
5. **Framework overhead** (~0.5ms) — Internal machinery, schema processing. Present even in non-containerized mode.

For workloads where computations take seconds or longer, total overhead is typically negligible. See the [rules of thumb](#rules-of-thumb-by-use-case) for guidance on which overhead sources dominate for different workloads.

```{note}
Tesseract is not a high-performance RPC framework. If your workload requires microsecond latency or millions of calls per second, consider a more traditional RPC framework.
```

## Example scenario: Locally hosted Tesseract

### Benchmarking scenario

The numbers and figures on this page are based on benchmarks run under a specific scenario. This scenario represents a best-case baseline for Tesseract overhead: it minimizes network latency and container virtualization costs, so the numbers isolate framework overhead rather than infrastructure overhead.

- **Bare-metal Linux Docker** (no Docker Desktop virtualization)
- **Loopback networking** (Tesseract running on the same machine as the client)
- **Local SSD** for binref disk I/O
- All arrays are **float64**

Your numbers will differ depending on your setup. In particular:

- **Docker Desktop (macOS/Windows)** adds a virtualization layer, increasing container startup time and HTTP latency, and decreasing raw performance.
- **Remote Tesseracts** make network latency and bandwidth the dominant cost for HTTP mode — compact encodings (base64, binref) matter even more.
- **Network-attached storage** for binref can be significantly slower than local SSD.

Benchmark with representative inputs to understand the trade-offs for your use case.

### The right interaction mode depends on your workload

```{warning}
Advice in this section is specific to the benchmarking scenario described above. Your mileage will vary based on your setup and workload.
```

<br>

```{figure} /img/benchmark_guidance.png
:alt: Tesseract overhead guidance chart
:width: 100%

Overhead as percentage of computation time, depending on interaction mode and I/O data size, for the benchmark scenario (local Tesseract with fast network and disk).
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

```{warning}
Advice in this section is specific to the benchmarking scenario described above. Your mileage will vary based on your setup and workload.
```

| Scenario                                       | Recommendation                                                                                                                                                                                                                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tight loops on in-memory data**              | Consider {doc}`non-containerized execution </content/using-tesseracts/use>` to bypass all network/container overhead. At ~0.5ms per call, you can run thousands of iterations per second. Requires all dependencies to be available in the same local Python environment. |
| **Second-scale workloads on medium-size data** | The sweet spot for containerized HTTP execution, with low overhead benefitting from most Tesseract features.                                                                                                                                                              |
| **Cheap operations on small data via HTTP**    | HTTP overhead (~2.5ms) can dominate when computation is fast. Batch multiple inputs into a single request.                                                                                                                                                                |
| **Development and debugging**                  | Use {doc}`non-containerized execution </content/using-tesseracts/use>` or {doc}`tesseract-runtime serve </content/creating-tesseracts/deploy>` for fast iteration, then switch to containerized HTTP for final testing.                                                   |
| **Shell scripts and one-off runs**             | CLI is convenient but has ~2s overhead per invocation from container startup. For multiple calls, keep a container running.                                                                                                                                               |
| **Long-running operations on large datasets**  | Use CLI with `json+binref` encoding. The ~2s container overhead is negligible for multi-minute runs, and binref allows large arrays to be passed between Tesseracts without expensive copies.                                                                             |
| **Cheap operations on huge datasets**          | Serialization and transfer will dominate. Try partitioning your workload so each Tesseract call does more compute per byte of I/O, or use binref to avoid redundant data copies between pipeline stages.                                                                  |

## Optimizing performance

### 1. Choose the right encoding format

Encoding format affects both serialization time and the volume of data transferred. A 10M-element float64 array is ~76MB as raw binary, ~100MB as base64, and ~230-760MB as JSON. If I/O is slow, data transfer dominates over serialization, and choosing a compact format is the most effective optimization.

In short: use **base64** (default) for HTTP transport, **binref** for large arrays or disk-based pipelines, and **json** only when you need human-readable output. See {doc}`/content/using-tesseracts/array-encodings` for format details and usage examples.

### 2. Batch small operations

If you have many small operations, batch them into a single request:

```python
# ❌ Avoid: Many small calls
for item in items:
    result = tesseract.apply({"data": item})

# ✅ Prefer: Batch into one call
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

### 4. Profile to find bottlenecks

Enable profiling to understand where time is spent. See {doc}`/content/debugging` for how to enable and interpret profiling output.
