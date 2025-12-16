# Advanced Usage

## File aliasing

The `tesseract` command can take care of
passing data from local disk
(or any [fsspec-compatible](https://filesystem-spec.readthedocs.io/en/latest/) resource,
like HTTP, FTP, S3 Buckets, and so on) to a Tesseract via the `@` syntax.

You can mount a folder into a Tesseract with `--input-path`. A The input path is mounted with read-only permissions so a Tesseract will never mutate files located at the input path.
Paths in a Tesseract's payload have to be relative to `--input-path`:
```bash
tesseract run filereference apply \
    --input-path ./testdata \
    --output-path ./output \
    '{"inputs": {"data": ["sample_2.json", "sample_3.json"]}}'
```
See [`examples/filereference`](../examples/building-blocks/filereference)


If you want to write the output of a Tesseract to a file,
you can use the `--output-path` parameter, which also supports any
[fsspec-compatible](https://filesystem-spec.readthedocs.io/en/latest/)
target path:

```bash
$ tesseract run vectoradd apply --output-path /tmp/output @inputs.json
```

## Logging metrics and artifacts

Tesseracts may log metrics and artifacts (e.g. iteration numbers, VTK files, ...) as demonstrated in the `metrics` example Tesseract.

```{literalinclude} ../../../examples/metrics/tesseract_api.py
```

By default, Tesseracts log metrics, parameters and artifacts to a directory `logs` in the Tesseract's `--output-path`. (Note that, when running Tesseracts in a container, the log directory is placed inside the container.)

Alternatively, you can log metrics and artifacts to an MLflow server by setting the `TESSERACT_MLFLOW_TRACKING_URI` environment variable. For local development, you can spin up an MLflow server (ready to use with Tesseract) through the provided docker-compose file:

```bash
docker-compose -f extra/mlflow/docker-compose-mlflow.yml up
```

This MLflow server shows logged items in the MLflow GUI at `http://localhost:5000`.

Launch the `metrics` example Tesseract with the the following volume mount and `TESSERACT_MLFLOW_TRACKING_URI` to ensure that it connects to that MLflow server. Note that your Tesseract now needs to declare `mlflow` as a required package in its `requirements.txt` file.

MacOS and Windows with Docker Desktop requires the `host.docker.internal` hostname to access services running on the host machine:

```bash
tesseract serve --env=TESSERACT_MLFLOW_TRACKING_URI=http://host.docker.internal:5000 --volume mlflow-data:/mlflow-data:rw metrics
```

For Linux, set your Tesseracts to use the host network and access the MLflow server at `localhost`:

```bash
tesseract serve --env=TESSERACT_MLFLOW_TRACKING_URI=http://localhost:5000 --network=host --volume mlflow-data:/mlflow-data:rw metrics
```

The same options apply when executing Tesseracts through `tesseract run`.

As an alternative to the MLflow setup we provide, you can point your Tesseract to a custom MLflow server:

```bash
$ tesseract serve --env=TESSERACT_MLFLOW_TRACKING_URI="..."  metrics
````

Note that if your MLFlow server uses basic auth, you need to populate the `TESSERACT_MLFLOW_TRACKING_USERNAME` and 
`TESSERACT_MLFLOW_TRACKING_PASSWORD` for the Tesseract to be able to authenticate to it.

```bash
$ tesseract serve --env=TESSERACT_MLFLOW_TRACKING_URI="..." \
    --env=MLFLOW_TRACKING_USERNAME="..." --env=MLFLOW_TRACKING_PASSWORD="..." \
    metrics
````

## Volume mounts and user permissions

When mounting a volume into a Tesseract container, default behavior depends on the Docker engine being used. Specifically, Docker Desktop, Docker Engine, and Podman have different ways of handling user permissions for mounted volumes.

Tesseract tries to ensure that the container user has the same permissions as the host user running the `tesseract` command. This is done by setting the user ID and group ID of the container user to match those of the host user.

In cases where this fails or is not desired, you can explicitly set the user ID and group ID of the container user using the `--user` argument. This allows you to specify a different user or group for the container, which can be useful for ensuring proper permissions when accessing mounted volumes.

```{warning}
In cases where the Tesseract user is neither `root` nor the local user / file owner, you may encounter permission issues when accessing files in mounted volumes. To resolve this, ensure that the user ID and group ID are set correctly using the `--user` argument, or modify the permissions of files to be readable by any user.
```

## Passing environment variables to Tesseract containers

Through the optional `--env` argument, you can pass environment variables to Tesseracts.
This works both for serving a Tesseract and running a single execution:

```bash
$ tesseract serve --env=MY_ENV_VARIABLE="some value" helloworld
$ tesseract run --env=MY_ENV_VARIABLE="some value" helloworld apply '{"inputs": {"name": "Osborne"}}'
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

## Deploying and interacting with Tesseracts on HPC clusters

Running Tesseracts on high-performance computing clusters can have many use cases including:

* Deployment of a single long-running component of a pipeline on a state-of-the-art GPU.
* Running an entire optimization workflow on a dedicated compute node
* Large parameter scans distributed in parallel over a multitude of cores.

All of this is possible even in scenarios where containerisation options are either unavailable or incompatible by directly using `tesseract-runtime` (which includes a `serve` feature). For more details, please see our [tutorial](https://si-tesseract.discourse.group/t/deploying-and-interacting-with-tesseracts-on-hpc-clusters-using-tesseract-runtime-serve/104), which demonstrates how to launch uncontainerised Tesseracts using SLURM, either as a batch job or for interactive use.

## Debug mode

`tesseract serve` supports a `--debug` flag; this has two effects:
  *  Tracebacks from execution are returned in the response body, instead of a generic 500 error.
     This is useful for debugging and testing, but unsafe for production environments.
  *  Aside from listening to the usual Tesseract requests, a debugpy server is also started in
     the container, and the port it's listening to is forwarded to some free port on the host which
     is displayed in the cli when spinning up a tesseract via `tesseract serve`. This allows you to perform
     remote debugging sessions.

In particular, if you are using VScode, here is a sample launch config to attach to a running Tesseract in
debug mode:
```json
        {
            "name": "Tesseract: Remote debugger",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": "PORT_NUMBER_HERE"
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}/examples/helloworld",
                    "remoteRoot": "/tesseract"
                }
            ],
        },
```
(make sure to fill in with the actual port number). After inserting this into the `configurations`
field of your `launch.json` file, you should be able to attach to the Tesseract being served by clicking on the
green "play" button at the top left corner of the "Run and Debug" tab.

![Starting remote debug session in VScode](./remote_debug.png)

For more information on the VSCode debugger, see [this guide](https://code.visualstudio.com/docs/debugtest/debugging).
