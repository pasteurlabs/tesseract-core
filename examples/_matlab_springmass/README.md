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

- **Docker**
- **A valid MATLAB license**, either:
  - A **network license server** reachable from the container (e.g. `27000@server`), or
  - A **MATLAB Individual license file** (`.lic`) and its associated activation MAC address

No local MATLAB installation or additional toolboxes are needed.

## Build the wrapper base image

The upstream `mathworks/matlab` image needs two small adjustments before it
can be used as a Tesseract base image — see [`Dockerfile.matlab-base`](Dockerfile.matlab-base)
for the rationale. Build it once:

```bash
docker build -f Dockerfile.matlab-base \
  --build-arg MATLAB_RELEASE=r2026a \
  --build-arg MATLAB_USERNAME=matlab-user \
  -t matlab-root:r2026a .
```

For a **network license**, the value of `MATLAB_USERNAME` does not matter.
For an **Individual license file**, set `MATLAB_USERNAME` to the username
linked to your license — see [Using an Individual License](#using-an-individual-license)
below for how to find it.

## Build the Tesseract

```bash
tesseract build .
```

## Run (network license)

MATLAB's JVM requires more shared memory than Docker's default 64 MB.
Pass `--shm-size=512M` via `--docker-args`, your license server via
`--env MLM_LICENSE_FILE`, and the server's DNS mapping via `--docker-args`:

```bash
tesseract run matlab-springmass \
  --env MLM_LICENSE_FILE=27000@your-license-server \
  --docker-args "--shm-size=512M --add-host=your-license-server:YOUR_SERVER_IP" \
  apply '{"inputs": {}}'
```

With custom parameters:

```bash
tesseract run matlab-springmass \
  --env MLM_LICENSE_FILE=27000@your-license-server \
  --docker-args "--shm-size=512M --add-host=your-license-server:YOUR_SERVER_IP" \
  apply '{"inputs": {"mass": 2.0, "damping": 1.0, "stiffness": 20.0, "force_amplitude": 5.0}}'
```

## Or serve as a REST API

```bash
tesseract serve matlab-springmass \
  --env MLM_LICENSE_FILE=27000@your-license-server \
  --docker-args "--shm-size=512M --add-host=your-license-server:YOUR_SERVER_IP"

# In another terminal:
curl -X POST http://localhost:8000/apply \
  -H 'Content-Type: application/json' \
  -d '{"inputs": {"mass": 1.0, "stiffness": 10.0}}'
```

## Use from Python

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

## Using an Individual License

If you don't have access to a network license server, you can run the
Tesseract using your MATLAB Individual license file (`.lic`).

### 1. Get your authorized username and activation MAC address

An Individual license is bound to a specific OS username and the MAC
address of the machine it was activated on. You can find both in your
MathWorks Account license details.
You will need them in the next two steps.

### 2. Build the wrapper with that username

Rebuild `matlab-root:r2026a` passing `--build-arg MATLAB_USERNAME=<your-username>`
(see [Build the wrapper base image](#build-the-wrapper-base-image)). The
username inside the container will then match what your license expects,
and the file-based license check will succeed.

### 3. Run with the license file mounted

```bash
tesseract run matlab-springmass \
  --user 1000:1000 \
  --volume /path/to/your/license.lic:/licenses/license.lic \
  --env MLM_LICENSE_FILE=/licenses/license.lic \
  --docker-args "--shm-size=512M --mac-address=XX:XX:XX:XX:XX:XX" \
  apply '{"inputs": {}}'
```

- `--user 1000:1000` runs the container as the renamed user (UID 1000, which
  is where the `MATLAB_USERNAME` you set in the wrapper ends up).
- `--mac-address=XX:XX:XX:XX:XX:XX` must match the MAC your license was
  activated against.

## How It Works

The `tesseract_config.yaml` uses `matlab-root:r2026a` (the wrapper image
you built above) as the base. The `.m` source file is copied in at build
time, and `tesseract_api.py` invokes it via `matlab -batch`:

```
Python (tesseract_api.py)
  ├── Write input JSON → /tmp/input.json
  ├── subprocess: matlab -batch "addpath(...); spring_mass_damper('input.json', 'output.json')"
  │     └── MATLAB reads input JSON (jsondecode)
  │     └── ode45 solves the ODE system
  │     └── MATLAB writes output JSON (jsonencode)
  └── Read output JSON → OutputSchema
```

## Visualization

```bash
python visualize.py
# Or with custom parameters:
python visualize.py --mass 2.0 --damping 3.0 --stiffness 50.0
```

## File Structure

```
_matlab_springmass/
├── matlab/
│   └── spring_mass_damper.m    # MATLAB solver source
├── tesseract_api.py            # Python wrapper (InputSchema, OutputSchema, apply)
├── tesseract_config.yaml       # Build configuration
├── tesseract_requirements.txt  # Python dependencies
├── Dockerfile.matlab-base      # Wrapper for mathworks/matlab (see top)
├── test_cases/
│   └── test_apply.json         # Expected output for default inputs
├── visualize.py                # Visualization script
├── plots/                      # Generated plots (not committed)
└── README.md
```
