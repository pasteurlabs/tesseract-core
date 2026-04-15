# Advanced Usage

## File aliasing

The `tesseract` CLI can load data from local disk or any [fsspec-compatible](https://filesystem-spec.readthedocs.io/en/latest/) resource (HTTP, FTP, S3, etc.) using the `@` syntax.

Use `--input-path` to mount a folder into the Tesseract (read-only). Paths in the payload must be relative to `--input-path`:

```bash
tesseract run pathreference apply \
    --input-path ./testdata \
    --output-path ./output \
    '{"inputs": {"paths": ["sample_2.json", "sample_3.json"]}}'
```

See the [`pathreference` example](../examples/building-blocks/pathreference) for a complete walkthrough.

To write output to a file, use `--output-path` (also supports fsspec-compatible targets):

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

By default, metrics, parameters, and artifacts are logged to a `logs` directory in the Tesseract's `--output-path`. (When running in a container, this directory lives inside the container.)

To log to an [MLflow](https://mlflow.org/) server instead, set the `TESSERACT_MLFLOW_TRACKING_URI` environment variable. For local development, spin up an MLflow server using the provided Docker Compose file:

```bash
docker-compose -f extra/mlflow/docker-compose-mlflow.yml up
```

Then launch the `metrics` Tesseract with the appropriate volume mount, network, and tracking URI:

```bash
tesseract serve --network=tesseract-mlflow-server --env=TESSERACT_MLFLOW_TRACKING_URI=http://mlflow-server:5000 --volume mlflow-data:/mlflow-data:rw metrics
```

The same options work with `tesseract run`.

To connect to a custom MLflow server instead:

```bash
$ tesseract serve --env=TESSERACT_MLFLOW_TRACKING_URI="..."  metrics
```

If your MLflow server uses basic auth, pass the credentials as environment variables:

```bash
$ tesseract serve --env=TESSERACT_MLFLOW_TRACKING_URI="..." \
    --env=MLFLOW_TRACKING_USERNAME="..." --env=MLFLOW_TRACKING_PASSWORD="..." \
    metrics
```

To pass additional parameters to the MLflow run (tags, run name, description), use `TESSERACT_MLFLOW_RUN_EXTRA_ARGS`. This accepts a Python dictionary string passed directly to [`mlflow.start_run()`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.start_run).

**Example: Setting tags only**

```bash
$ tesseract serve --env=TESSERACT_MLFLOW_TRACKING_URI="..." \
    --env=TESSERACT_MLFLOW_RUN_EXTRA_ARGS='{"tags": {"key1": "value1", "key2": "value2"}}' \
    metrics
```

**Example: Setting run name and tags**

```bash
$ tesseract serve --env=TESSERACT_MLFLOW_TRACKING_URI="..." \
    --env=TESSERACT_MLFLOW_RUN_EXTRA_ARGS='{"run_name": "my_experiment", "tags": {"env": "production"}}' \
    metrics
```

**Example: Multiple parameters**

```bash
$ tesseract serve --env=TESSERACT_MLFLOW_TRACKING_URI="..." \
    --env=TESSERACT_MLFLOW_RUN_EXTRA_ARGS='{"run_name": "test_run", "description": "Testing new feature", "tags": {"version": "1.0"}}' \
    metrics
```

## Volume mounts and user permissions

Permission handling for mounted volumes varies between Docker Desktop, Docker Engine, and Podman. By default, Tesseract maps the container user's UID and GID to match the host user running the `tesseract` command.

If this doesn't work for your setup, override it with the `--user` argument to set a specific UID/GID for the container.

```{warning}
If the container user is neither `root` nor the file owner, you may encounter permission errors on mounted volumes. Fix this by setting the correct UID/GID with `--user`, or by making the files readable by all users.
```

## Passing environment variables

Use `--env` to pass environment variables to Tesseract containers. This works with both `serve` and `run`:

```bash
$ tesseract serve --env=MY_ENV_VARIABLE="some value" helloworld
$ tesseract run --env=MY_ENV_VARIABLE="some value" helloworld apply '{"inputs": {"name": "Osborne"}}'
```

## Parallelism and worker processes

By default, Tesseracts run with a single worker process. To handle concurrent requests, increase the worker count with `--num-workers` (for `tesseract serve`) or the `num_workers` parameter (in the Python SDK). This is not available for `tesseract run`, which processes a single request and exits.

Each worker runs as a separate process, so they are not affected by the GIL but do not share in-process state.

### When to use multiple workers

Multiple workers are useful when:

- **Handling concurrent requests** — If multiple clients will call your Tesseract simultaneously, each worker can handle one request at a time. With a single worker, requests are processed sequentially.
- **CPU-bound computations** — If your Tesseract performs CPU-intensive work and you have multiple cores available, multiple workers can process requests in parallel.
- **Batch processing** — When processing many independent inputs, you can submit them concurrently and let workers handle them in parallel.

### When NOT to use multiple workers

Stick with a single worker when:

- **GPU-bound computations** — GPUs typically can't run multiple processes efficiently. If your Tesseract uses a GPU, multiple workers will compete for GPU resources and may cause out-of-memory errors or slowdowns.
- **High memory usage** — Each worker loads its own copy of the model/data into memory. If your Tesseract uses 4GB of RAM, 4 workers will use 16GB total.
- **Stateful operations** — Workers don't share state. If your computation requires shared state between requests, multiple workers won't work correctly.

### CLI usage

```bash
# Serve with 4 worker processes
$ tesseract serve --num-workers 4 my-tesseract
```

### Python SDK usage

```python
from concurrent.futures import ThreadPoolExecutor
from tesseract_core import Tesseract

# Serve with multiple workers
with Tesseract.from_image("my-tesseract", num_workers=4) as t:
    # Process requests concurrently using threads
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

Use the `--gpus` argument to make NVIDIA GPUs available to a Tesseract.

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

## Tesseracts on HPC clusters

Common HPC use cases for Tesseracts include:

- Deploying a long-running pipeline component on a GPU node
- Running an optimization workflow on a dedicated compute node
- Distributing parameter scans across many cores

This works even without containerization, using `tesseract-runtime serve` directly. See our [HPC tutorial](https://si-tesseract.discourse.group/t/deploying-and-interacting-with-tesseracts-on-hpc-clusters-using-tesseract-runtime-serve/104) for a SLURM-based walkthrough covering both batch and interactive use.

(running-without-containers)=

## Running Tesseracts without containers

When containerization is unavailable or undesirable, run Tesseracts directly using the `tesseract-runtime` CLI (the same command that runs inside Tesseract containers).

Setup:

1. [Install tesseract-core](installation-runtime) with the runtime extra.
2. Install the Tesseract's dependencies: `pip install -r tesseract_requirements.txt`
3. Set `TESSERACT_API_PATH` to point to your `tesseract_api.py`.

Then use `tesseract-runtime` instead of `tesseract run`:

```bash
# Instead of:
$ tesseract run helloworld apply '{"inputs": {"name": "Tessie"}}'

# Use:
$ export TESSERACT_API_PATH=/path/to/tesseract_api.py
$ tesseract-runtime apply '{"inputs": {"name": "Tessie"}}'
```

`tesseract-runtime` supports the same endpoints and options as containerized Tesseracts. Run `tesseract-runtime --help` for details.

```{tip}
Running without containers is also useful for [debugging and development](project:#running-tesseracts-without-containerization).
```
