# Debugging guide

This page covers strategies for interactive development and debugging of Tesseracts.

(running-tesseracts-without-containerization)=

## Running Tesseracts without containerization

While developing a Tesseract, the process of building and rebuilding the
tesseract image for quick local tests can be very time-consuming. The fastest and most
convenient way to speed this up is to run the code directly in your local Python environment using `tesseract-runtime`.

```{seealso}
Running without containers is also useful as a deployment option in environments where Docker is unavailable. See <project:#running-without-containers>.
```

To set up local development:

1. Make sure you have a development installation of Tesseract (see <project:#installation-dev>).
2. Install your Tesseract's dependencies: `pip install -r tesseract_requirements.txt`
3. Set the `TESSERACT_API_PATH` environment variable:
   ```bash
   $ export TESSERACT_API_PATH=/path/to/your/tesseract_api.py
   ```

Now you can use `tesseract-runtime` directly:

```bash
# Instead of building and running:
$ tesseract run helloworld apply '{"inputs": {"name": "Tessie"}}'

# Just run directly:
$ tesseract-runtime apply '{"inputs": {"name": "Tessie"}}'
```

This enables fast iteration cycles—edit your code, run, and see results immediately without rebuilding containers.

## Using the Python SDK for local development

Another approach for rapid iteration is using the Python SDK's `Tesseract.from_tesseract_api()` method, which loads your Tesseract API directly without containerization:

```python
from tesseract_core import Tesseract
import numpy as np

# Load directly from the tesseract_api.py file
tess = Tesseract.from_tesseract_api("/path/to/your/tesseract_api.py")

# Call endpoints directly
result = tess.apply(inputs={"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])})
```

This approach is particularly useful for:

- Interactive development in Jupyter notebooks
- Running unit tests against your Tesseract
- Debugging with Python debuggers like `pdb` or IDE debuggers

## Debug mode

`tesseract serve` supports a `--debug` flag; this has two effects:

- Tracebacks from execution are returned in the response body, instead of a generic 500 error.
  This is useful for debugging and testing, but unsafe for production environments.
- Aside from listening to the usual Tesseract requests, a debugpy server is also started in
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

![Starting remote debug session in VScode](./using-tesseracts/remote_debug.png)

For more information on the VSCode debugger, see [this guide](https://code.visualstudio.com/docs/debugtest/debugging).

## Profiling

Tesseract includes built-in profiling support to help identify performance bottlenecks. When profiling is enabled, the runtime collects CPU profiling statistics using Python's `cProfile` module and reports them after each endpoint execution.

### Enabling profiling

There are several ways to enable profiling:

**Via `tesseract run` CLI flag:**

```bash
$ tesseract run myimage apply '{"inputs": {...}}' --profiling
```

**Via environment variable (for `tesseract-runtime`):**

```bash
$ TESSERACT_PROFILING=1 tesseract-runtime apply '{"inputs": {...}}'
```

**Via Python SDK:**

```python
tess = Tesseract.from_tesseract_api(
    "/path/to/tesseract_api.py",
    runtime_config={"profiling": True}
)
```

### Understanding profiling output

When profiling is enabled, you'll see output like this after each endpoint call:

```
--- Profiling Statistics ---
=== By Cumulative Time (includes sub-calls) ===
         1234 function calls in 0.456 seconds

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       10    0.100    0.010    0.400    0.040 mymodule.py:42(expensive_function)
      100    0.050    0.001    0.200    0.002 numpy/core/fromnumeric.py:70(sum)
      ...

=== By Total Time (excluding sub-calls) ===
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       10    0.100    0.010    0.400    0.040 mymodule.py:42(expensive_function)
      ...
```

The output includes two sorted views:

- **By Cumulative Time**: Shows functions that took the most total time including all sub-calls. Useful for finding high-level bottlenecks.
- **By Total Time**: Shows functions that spent the most time in their own code (excluding sub-calls). Useful for finding low-level hotspots.

```{note}
Framework internals (FastAPI, Starlette, uvicorn, etc.) are automatically filtered out to focus on your code.
```

## Tracing

Tracing provides detailed debug-level logging of Tesseract operations. This is useful for understanding the flow of data through your Tesseract and diagnosing issues.

### Enabling tracing

**Via `tesseract run` CLI flag:**

```bash
$ tesseract run myimage apply '{"inputs": {...}}' --tracing
```

**Via environment variable (for `tesseract-runtime`):**

```bash
$ TESSERACT_TRACING=1 tesseract-runtime apply '{"inputs": {...}}'
```

**Via Python SDK:**

```python
tess = Tesseract.from_tesseract_api(
    "/path/to/tesseract_api.py",
    runtime_config={"tracing": True}
)
```

### Tracing output

When tracing is enabled, you'll see DEBUG-level log messages with timestamps:

```
2024-01-15 10:30:45,123 - tesseract_runtime - DEBUG - Processing apply request
2024-01-15 10:30:45,124 - tesseract_runtime - DEBUG - Input validation complete
2024-01-15 10:30:45,456 - tesseract_runtime - DEBUG - Apply execution complete
```

## Combining profiling and tracing

You can enable both profiling and tracing simultaneously for comprehensive debugging:

```bash
$ tesseract run myimage apply '{"inputs": {...}}' --profiling --tracing
```

Or via the Python SDK:

```python
tess = Tesseract.from_tesseract_api(
    "/path/to/tesseract_api.py",
    runtime_config={"profiling": True, "tracing": True}
)
```

## Debugging build failures

There are several options you can provide to `tesseract build` which can be helpful in
various circumstances:

- The output of the various steps which happen under-the-hood while doing a build will
  only be printed if something fails; this means that your shell might appear unresponsive
  during this process. If you want more detailed information on what's going on during your
  build, and see updates about it in real-time, use `--loglevel debug`.
- `--config-override` can be used to manually override options specified in the `tesseract_config.yaml`,
  for example: `--config-override build_config.target_platform=linux/arm64`
- `tesseract build` relies on a `docker build` command to create the Tesseract image. By
  default, the build context is a temporary folder to which all necessary files to build a Tesseract
  are copied to. The option `--build-dir <directory>` allows you to specify a different
  directory where to do this operations. This might be useful to debug issues which
  arise while building a Tesseract, as in `directory` you will see all the context available to
  `docker build` and nothing else.

## Debugging tips

### Common issues

1. **Import errors**: If your Tesseract fails to load, check that all dependencies are installed and importable. Use `tesseract-runtime check` to validate your `tesseract_api.py`.

2. **Schema validation errors**: Use tracing to see the exact input being passed and compare it against your schema definition.

3. **Performance issues**: Enable profiling to identify slow functions. Look for:
   - Functions with high `tottime` (time spent in the function itself)
   - Functions called many times (`ncalls` column)
   - Unexpected functions appearing in the profile

### Using Python debuggers

When running without containerization via `tesseract-runtime` or `Tesseract.from_tesseract_api()`, you can use standard Python debugging tools:

```python
import pdb

def apply(inputs):
    pdb.set_trace()  # Execution will pause here
    result = expensive_computation(inputs.data)
    return OutputSchema(result=result)
```

Or use your IDE's debugger by setting breakpoints in your `tesseract_api.py` file.
