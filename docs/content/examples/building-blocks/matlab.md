# Wrapping MATLAB Code

## Context

MATLAB is one of the most widely used tools in engineering for simulation, control system design, and signal processing. Many organizations have decades of validated MATLAB code that they want to make accessible through modern APIs without rewriting. Tesseract provides a way to wrap MATLAB code by using the official [MathWorks MATLAB Docker image](https://hub.docker.com/r/mathworks/matlab) as the base image and calling MATLAB via `matlab -batch`.

This example demonstrates the pattern with a spring-mass-damper ODE solver. The approach works for any MATLAB script or function that can run in batch mode.

## Example Tesseract (examples/\_matlab_springmass)

### The MATLAB solver

The solver simulates a damped harmonic oscillator under a step force input:

$$m \ddot{x} + c \dot{x} + k x = F_0$$

using MATLAB's `ode45` (Dormand-Prince Runge-Kutta). It reads input parameters from a JSON file and writes results to a JSON file:

```{literalinclude} ../../../../examples/_matlab_springmass/matlab/spring_mass_damper.m
:language: matlab
```

### Input and output schemas

The `InputSchema` defines the physical parameters of the spring-mass-damper system:

```{literalinclude} ../../../../examples/_matlab_springmass/tesseract_api.py
:pyobject: InputSchema
:language: python
```

The `OutputSchema` returns time histories along with analytical system characteristics:

```{literalinclude} ../../../../examples/_matlab_springmass/tesseract_api.py
:pyobject: OutputSchema
:language: python
```

### Subprocess integration

The `apply` function writes input parameters as JSON, invokes MATLAB via `matlab -batch`, and reads back the JSON results:

```{literalinclude} ../../../../examples/_matlab_springmass/tesseract_api.py
:pyobject: apply
:language: python
```

### Build configuration

The `tesseract_config.yaml` uses the official MathWorks MATLAB Docker image as the base, so MATLAB is pre-installed --- no compilation or additional toolboxes are needed:

```{literalinclude} ../../../../examples/_matlab_springmass/tesseract_config.yaml
:language: yaml
```

Key points:

- **`base_image`**: Uses `mathworks/matlab:r2025b`, which has a full MATLAB installation
- **`package_data`**: Copies the `.m` source file into the container
- **No compilation step**: MATLAB runs the `.m` file directly via `matlab -batch`

## Runtime requirements

Because the container includes MATLAB but not a license, you must provide a network license server at runtime:

```bash
tesseract run matlab-springmass apply '{}' \
  --runtime-args "\
    -e MLM_LICENSE_FILE=27000@your-license-server \
    --add-host=your-license-server:YOUR_SERVER_IP \
    --shm-size=512M"
```

The `--shm-size=512M` flag is required because MATLAB's JVM uses POSIX shared memory, and Docker's default of 64MB is insufficient.

## Adapting this pattern

To wrap your own MATLAB code:

1. **Choose a base image**: Use `mathworks/matlab:<release>` matching your license. Tags are available for R2024a and later.
2. **Structure your MATLAB code**: Write a function that takes file paths as arguments for I/O. Use `jsondecode`/`jsonencode` (available since R2016b) for data exchange.
3. **Create schemas**: Map your MATLAB function's parameters to Pydantic models.
4. **Implement `apply()`**: Write the subprocess glue that calls `matlab -batch`.
5. **Configure licensing**: Ensure `MLM_LICENSE_FILE` is passed at runtime.

For distributing Tesseracts to users without a MATLAB license, consider the [MATLAB Compiler SDK](https://www.mathworks.com/products/compiler-sdk.html): compile `.m` files into standalone executables and bundle the free MATLAB Runtime instead.
