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

### Verbose build output

By default, `tesseract build` only prints output when something fails, which can make your shell appear unresponsive during long builds. To see real-time progress:

```bash
$ tesseract build --loglevel debug
```

### Inspecting the build context

`tesseract build` creates a temporary folder containing all files passed to `docker build`. To inspect this context (useful for debugging file inclusion issues):

```bash
$ tesseract build --build-dir ./debug_build
```

After running this, `./debug_build` will contain exactly what Docker sees during the build—nothing more, nothing less.

### Overriding configuration

Use `--config-override` to temporarily change settings from `tesseract_config.yaml` without editing the file:

```bash
$ tesseract build --config-override build_config.target_platform=linux/arm64
```

## Debugging tips

### Common issues

1. **Import errors**: If your Tesseract fails to load, check that all dependencies are installed and importable. Use `tesseract-runtime check` to validate your `tesseract_api.py`.

2. **Schema validation errors**: Use tracing to see the exact input being passed and compare it against your schema definition.

3. **Performance issues**: Enable profiling to identify slow functions. Look for:
   - Functions with high `tottime` (time spent in the function itself)
   - Functions called many times (`ncalls` column)
   - Unexpected functions appearing in the profile

### Troubleshooting dependencies

Dependency issues are a common source of build and runtime failures. Here's how to diagnose and fix them:

**Testing dependencies locally before building:**

Before running `tesseract build`, verify your dependencies work in a clean environment:

```bash
# Create a fresh virtual environment
$ python -m venv test_env
$ source test_env/bin/activate

# Install only what's in your requirements file
$ pip install -r tesseract_requirements.txt

# Test that your Tesseract loads
$ TESSERACT_API_PATH=/path/to/tesseract_api.py tesseract-runtime check
```

**Common dependency problems:**

- **Missing transitive dependencies**: Your code may depend on a package that's installed as a dependency of something else in your main environment, but not listed in `tesseract_requirements.txt`. The clean environment test above will catch this.

- **Version conflicts**: If you see errors about incompatible versions, use `pip install` with specific version constraints in your requirements file (e.g., `numpy>=1.20,<2.0`).

- **System library dependencies**: Some Python packages require system libraries (e.g., `libgomp` for OpenMP, `libffi` for cffi). If the build succeeds but the container fails at runtime with "library not found" errors, you may need to add system packages via `tesseract_config.yaml`:

  ```yaml
  build_config:
    system_packages:
      - libgomp1
  ```

### Inspecting containers with Docker

When a Tesseract builds successfully but behaves unexpectedly at runtime, you can use Docker commands to inspect the container state.

**Get a shell inside a running Tesseract:**

```bash
# First, find the container ID
$ docker ps
CONTAINER ID   IMAGE        COMMAND                  ...
a1b2c3d4e5f6   mytesseract  "uvicorn tesseract..."   ...

# Open an interactive shell
$ docker exec -it a1b2c3d4e5f6 /bin/bash
```

From inside the container, you can:

- Check installed packages: `pip list`
- Verify files are in place: `ls -la /tesseract`
- Test imports manually: `python -c "import your_module"`
- Check environment variables: `env | grep TESSERACT`

**Inspect a Tesseract image without running it:**

```bash
# Start a shell in a new container from the image
$ docker run -it --entrypoint /bin/bash mytesseract:latest
```

**View container logs:**

```bash
# For a running container
$ docker logs a1b2c3d4e5f6

# Follow logs in real-time
$ docker logs -f a1b2c3d4e5f6
```

**Check resource usage:**

```bash
$ docker stats a1b2c3d4e5f6
```

This is useful for diagnosing out-of-memory errors or CPU throttling issues.

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
