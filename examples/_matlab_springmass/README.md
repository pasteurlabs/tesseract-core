# MATLAB Spring-Mass-Damper Tesseract

This example demonstrates how to wrap MATLAB code as a Tesseract using
the official [MathWorks MATLAB Docker image](https://hub.docker.com/r/mathworks/matlab).
MATLAB is pre-installed in the base image — no compilation or additional
toolboxes required.

The solver uses MATLAB's `ode45` to simulate a spring-mass-damper system:

```
m * x'' + c * x' + k * x = F₀  (step force input)
```

## Prerequisites

- **A MATLAB network license server** reachable from the container
- **Docker**

No local MATLAB installation or additional toolboxes are needed.

## Docker Runtime Requirements

MATLAB's JVM requires more shared memory than Docker's default 64MB.
Pass `--shm-size=512M` via `--runtime-args` in all `tesseract run` /
`tesseract serve` commands below. You also need to pass your license
server and its DNS mapping:

```bash
MATLAB_RUNTIME_ARGS="\
  -e MLM_LICENSE_FILE=27000@your-license-server \
  --add-host=your-license-server:YOUR_SERVER_IP \
  --shm-size=512M"
```

Set this once and reuse it in the commands below.

## Quick Start

### 1. Build the Tesseract

```bash
tesseract build .
```

### 2. Run

```bash
tesseract run matlab-springmass apply '{}' \
  --runtime-args "$MATLAB_RUNTIME_ARGS"
```

With custom parameters:

```bash
tesseract run matlab-springmass apply '{
  "mass": 2.0,
  "damping": 1.0,
  "stiffness": 20.0,
  "force_amplitude": 5.0
}' --runtime-args "$MATLAB_RUNTIME_ARGS"
```

### 3. Or serve as a REST API

```bash
tesseract serve matlab-springmass --runtime-args "$MATLAB_RUNTIME_ARGS"

# In another terminal:
curl -X POST http://localhost:8000/apply \
  -H 'Content-Type: application/json' \
  -d '{"inputs": {"mass": 1.0, "stiffness": 10.0}}'
```

### 4. Use from Python

```python
from tesseract_core import Tesseract

with Tesseract.from_image("matlab-springmass") as t:
    result = t.apply(inputs={
        "mass": 1.0,
        "damping": 0.5,
        "stiffness": 10.0,
    })
    print(f"Damping ratio: {result['damping_ratio']:.3f}")
    print(f"Final displacement: {result['displacement'][-1]:.6f} m")
```

## Visualization

```bash
python visualize.py
# Or with custom parameters:
python visualize.py --mass 2.0 --damping 3.0 --stiffness 50.0
```

## How It Works

The `tesseract_config.yaml` uses `mathworks/matlab:r2025b` as the base
image, which has MATLAB pre-installed. The `.m` source file is copied in
at build time, and `tesseract_api.py` calls it via `matlab -batch`:

```
Python (tesseract_api.py)
  ├── Write input JSON → /tmp/input.json
  ├── subprocess: matlab -batch "addpath(...); spring_mass_damper('input.json', 'output.json')"
  │     └── MATLAB reads input JSON (jsondecode)
  │     └── ode45 solves the ODE system
  │     └── MATLAB writes output JSON (jsonencode)
  └── Read output JSON → OutputSchema
```

## Licensing

The container includes MATLAB but **not** a license. You must provide
access to a network license server at runtime via the `MLM_LICENSE_FILE`
environment variable. See [MathWorks: Licensing for Containers](https://www.mathworks.com/help/cloudcenter/ug/matlab-container-on-docker-hub.html)
for details.

## File Structure

```
matlab_springmass/
├── matlab/
│   └── spring_mass_damper.m    # MATLAB solver source
├── tesseract_api.py            # Python wrapper (InputSchema, OutputSchema, apply)
├── tesseract_config.yaml       # Build configuration
├── tesseract_requirements.txt  # Python dependencies
├── test_cases/
│   └── test_apply.json         # Expected output for default inputs
├── visualize.py                # Visualization script
├── plots/                      # Generated plots (not committed)
└── README.md
```
