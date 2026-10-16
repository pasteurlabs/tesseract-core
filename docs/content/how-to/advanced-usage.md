# Advanced Usage

## File aliasing

Payloads can refer to files by path instead of containing all data inline. A Tesseract reads the files its inputs refer to from its input path, and writes output files to its output path. With `Tesseract.from_source`, set them with `input_path` and `output_path`:

```python
from tesseract_core import Tesseract

with Tesseract.from_source(
    "file_io/tesseract_api.py", input_path="./testdata", output_path="./output"
) as tess:
    tess.apply({"paths": ["sample_2.json", "sample_3.json"]})
```

`tesseract-runtime` takes the same settings as `--input-path` and `--output-path`:

```bash
$ export TESSERACT_API_PATH=file_io/tesseract_api.py
$ tesseract-runtime --input-path ./testdata --output-path ./output \
    apply '{"inputs": {"paths": ["sample_2.json", "sample_3.json"]}}'
```

`tesseract run` mounts the input path into the container (read-only) and the output path for results:

```bash
$ tesseract run file_io apply \
    --input-path ./testdata \
    --output-path ./output \
    '{"inputs": {"paths": ["sample_2.json", "sample_3.json"]}}'
```

See the [`file_io` example](../examples/building-blocks/file_io) for a complete walkthrough.

On the command line, prefix a payload with `@` to read it from a file. This works for local paths and any [fsspec-compatible](https://filesystem-spec.readthedocs.io/en/latest/) resource (HTTP, FTP, S3, etc.). The `--output-path` of `tesseract run` also accepts fsspec-compatible targets:

```bash
$ tesseract run vectoradd apply --output-path /tmp/output @inputs.json
```

```{seealso}
For handling large datasets that don't fit in memory, see the [out-of-core dataloading tutorial](https://si-tesseract.discourse.group/t/out-of-core-dataloading/52) which demonstrates streaming data through Tesseracts using file references and volume mounts.
```

## Logging metrics and artifacts

Tesseracts can log metrics and artifacts (e.g., iteration numbers, VTK files) as shown in the `metrics` example:

```{literalinclude} ../../../examples/metrics/tesseract_api.py

```

By default, metrics, parameters, and artifacts are logged to a `logs` directory in the Tesseract's output path. (When running in a container, this directory lives inside the container.)

To log to an [MLflow](https://mlflow.org/) server instead, set `TESSERACT_MLFLOW_TRACKING_URI` to the server's URL. `Tesseract.from_source` and `tesseract-runtime` read it from the environment you start them from, and `Tesseract.from_source` also accepts it as runtime configuration:

```python
from tesseract_core import Tesseract

with Tesseract.from_source(
    "metrics/tesseract_api.py",
    runtime_config={"mlflow_tracking_uri": "http://localhost:8080"},
) as tess:
    tess.apply({})
```

For local development, `uvx mlflow server --port 8080` starts a server at that address.

If your MLflow server uses basic auth, set `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` in the same environment.

To pass additional parameters to the MLflow run (tags, run name, description), use `TESSERACT_MLFLOW_RUN_EXTRA_ARGS`, or `mlflow_run_extra_args` in `runtime_config`. It accepts a dictionary that is passed directly to [`mlflow.start_run()`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.start_run):

```python
runtime_config = {
    "mlflow_tracking_uri": "http://localhost:8080",
    "mlflow_run_extra_args": {"run_name": "my_experiment", "tags": {"env": "production"}},
}
```

### MLflow from a container

A container does not inherit your environment, so pass these variables with `--env`. This works with both `tesseract serve` and `tesseract run`:

```bash
$ tesseract serve --env=TESSERACT_MLFLOW_TRACKING_URI="..." \
    --env=MLFLOW_TRACKING_USERNAME="..." --env=MLFLOW_TRACKING_PASSWORD="..." \
    --env=TESSERACT_MLFLOW_RUN_EXTRA_ARGS='{"run_name": "my_experiment", "tags": {"env": "production"}}' \
    metrics
```

To run an MLflow server in Docker next to a Tesseract container, use the provided Docker Compose file:

```bash
docker-compose -f extra/mlflow/docker-compose-mlflow.yml up
```

Then launch the `metrics` Tesseract with the server's network, volume mount, and tracking URI:

```bash
tesseract serve --network=tesseract-mlflow-server --env=TESSERACT_MLFLOW_TRACKING_URI=http://mlflow-server:5000 --volume mlflow-data:/mlflow-data:rw metrics
```

(passing-environment-variables)=

## Passing environment variables

A Tesseract served with `Tesseract.from_source` inherits the environment of the Python process that starts it, plus the top-level `env` from its `tesseract_config.yaml` (see {ref}`setting-environment-variables`). `tesseract-runtime` inherits the environment of the shell that runs it, and does not read `tesseract_config.yaml`.

A container does not inherit your environment. Pass variables to it with `--env`, which works with both `serve` and `run`, or with `environment` in `Tesseract.from_image`:

```bash
$ tesseract serve --env=MY_ENV_VARIABLE="some value" helloworld
$ tesseract run --env=MY_ENV_VARIABLE="some value" helloworld apply '{"inputs": {"name": "Osborne"}}'
```

## Parallelism and worker processes

By default, a served Tesseract runs a single worker process. To handle concurrent requests, increase the worker count with `--num-workers` (for `tesseract-runtime serve` and `tesseract serve`) or the `num_workers` parameter of `Tesseract.from_image`. `Tesseract.from_source` always serves a single worker. Multiple workers are not available for `tesseract run` and one-shot `tesseract-runtime` commands, which process a single request and exit.

Each worker runs as a separate process, so they are not affected by the GIL but do not share in-process state.

### When to use multiple workers

Multiple workers are useful when:

- **Handling concurrent requests** — If multiple clients will call your Tesseract simultaneously, each worker can handle one request at a time. With a single worker, requests are processed sequentially.
- **CPU-bound computations** — If your Tesseract performs CPU-intensive work and you have multiple cores available, multiple workers can process requests in parallel.
- **Batch processing** — When processing many independent inputs, you can submit them concurrently and let workers handle them in parallel.

### When NOT to use multiple workers

Stick with a single worker when:

- **GPU-bound computations** — GPUs typically can't run multiple processes efficiently. If your Tesseract uses a GPU, multiple workers will compete for GPU resources and may cause out-of-memory errors or slowdowns.
- **GPU transports** — Serving a GPU transport other than `none` (such as `cuda_ipc`) requires a single worker.
- **High memory usage** — Each worker loads its own copy of the model/data into memory. If your Tesseract uses 4GB of RAM, 4 workers will use 16GB total.
- **Stateful operations** — Workers don't share state. If your computation requires shared state between requests, multiple workers won't work correctly.

### Serving with multiple workers

Start the Tesseract with `tesseract-runtime serve`, and connect to it with `Tesseract.from_url`:

```bash
$ export TESSERACT_API_PATH=/path/to/tesseract_api.py
$ tesseract-runtime serve --num-workers 4 --port 8000
```

```python
from concurrent.futures import ThreadPoolExecutor
from tesseract_core import Tesseract

t = Tesseract.from_url("http://localhost:8000")

# Process requests concurrently using threads
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(t.apply, batch))
```

For a Tesseract image, pass the worker count to `tesseract serve` or `Tesseract.from_image`:

```bash
$ tesseract serve --num-workers 4 my-tesseract
```

```python
with Tesseract.from_image("my-tesseract", num_workers=4) as t:
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(t.apply, batch))
```

### Choosing the right number of workers

As a starting point:

- **CPU-bound**: `num_workers` = number of CPU cores
- **I/O-bound** (e.g., calling external APIs): `num_workers` = 2 x number of CPU cores
- **GPU-bound**: `num_workers` = 1 (or match the number of GPUs if using `--gpus`)

Monitor memory usage and adjust. More workers isn't always better — context switching overhead can reduce throughput.

## Using GPUs

`Tesseract.from_source` and `tesseract-runtime` run as ordinary processes on your machine, so they see the same GPUs as the process that starts them. To restrict which GPUs a Tesseract uses, set `CUDA_VISIBLE_DEVICES` in that environment (see {ref}`passing-environment-variables`).

A container sees no GPUs unless you pass them in. Use the `--gpus` argument of `tesseract run` and `tesseract serve`, or `gpus` in `Tesseract.from_image`, to make NVIDIA GPUs available to it.

To use a specific GPU:

```bash
$ tesseract run --gpus 0 helloworld apply '{"inputs": {"name": "Osborne"}}'
```

To use all available GPUs:

```bash
$ tesseract run --gpus all helloworld apply '{"inputs": {"name": "Osborne"}}'
```

To specify multiple GPUs:

```bash
$ tesseract run --gpus 0 --gpus 1 helloworld apply '{"inputs": {"name": "Osborne"}}'
```

GPUs are indexed starting at zero, matching `nvidia-smi` conventions.

## Container options

The options in this section apply to Tesseracts running in containers, started with `tesseract serve`, `tesseract run`, or `Tesseract.from_image`.

### Networking and interactions between Tesseracts

Tesseract containers can be connected to shared Docker networks, allowing them to communicate with each other and with other services at runtime. A common use case is composing multiple Tesseracts into a pipeline, where one Tesseract calls another via its REST API using `Tesseract.from_url()`. A working end-to-end example of this pattern is available in {gh-tree}`examples/_multi-tesseract/multi-helloworld <examples/_multi-tesseract/multi-helloworld>`.

You can attach Tesseracts to a named Docker network using the `--network` flag. The network is created automatically if it does not already exist.

```bash
$ tesseract serve "my-tesseract:latest" --network my_network --network-alias my-tesseract --port 8000
```

The `serve` command prints container metadata to `stdout`, including the Tesseract's IP address and port on each connected network. The `--network-alias` argument assigns a hostname so other containers on the same network can reach it by name (e.g. `http://my-tesseract:8000`) rather than by IP. Note that the port is assigned randomly if left unspecified.

**A note on `--network=host`:**

Setting `--network=host` makes a container share the host's network stack, bypassing network isolation. Whether it works as expected depends on how Docker is running on your system.

Docker Engine on Linux runs the daemon directly on the host OS, so `--network=host` gives the container access to the host's network namespace, including `localhost`. This is the only platform that supports sharing the full network stack.

Docker Desktop runs the Docker daemon inside a lightweight Linux VM. Here `--network=host` refers to the network stack of the VM, not your actual machine. As a result services on your machine's `localhost` are not reachable from inside the container. Docker Desktop >=4.34 added opt-in support for host networking (Settings > Resources > Network > "Enable host networking"), but only works at the TCP/UDP level and does not give access to the host's network namespace. For more details check out the [Docker documentation](https://docs.docker.com/engine/network/drivers/host/).

### Volume mounts and user permissions

Permission handling for mounted volumes varies between Docker Desktop, Docker Engine, and Podman. By default, Tesseract maps the container user's UID and GID to match the host user running the `tesseract` command.

If this doesn't work for your setup, override it with the `--user` argument to set a specific UID/GID for the container.

```{warning}
If the container user is neither `root` nor the file owner, you may encounter permission errors on mounted volumes. Fix this by setting the correct UID/GID with `--user`, or by making the files readable by all users.
```

## Tesseracts on HPC clusters

Common HPC use cases for Tesseracts include:

- Deploying a long-running pipeline component on a GPU node
- Running an optimization workflow on a dedicated compute node
- Distributing parameter scans across many cores

This works without a container engine, using `tesseract-runtime serve` directly (see {ref}`running-without-containers`). See our [HPC tutorial](https://si-tesseract.discourse.group/t/deploying-and-interacting-with-tesseracts-on-hpc-clusters-using-tesseract-runtime-serve/104) for a SLURM-based walkthrough covering both batch and interactive use.
