# MATLAB Spring-Mass-Damper Tesseract

This example demonstrates how to wrap MATLAB code as a Tesseract using
the official [MathWorks MATLAB Docker image](https://hub.docker.com/r/mathworks/matlab)
directly as the base image. MATLAB is pre-installed in the base image —
no compilation or additional toolboxes required.

The solver uses MATLAB's `ode45` to simulate a spring-mass-damper system:

```
m * x'' + c * x' + k * x = F₀  (step force input)
```

## Prerequisites

- **Docker**
- **A valid MATLAB license**, either:
  - A **network license server** reachable from the container (e.g. `27000@your-license-server`), or
  - A **MATLAB Individual license file** (`.lic`) along with the activation MAC address

No local MATLAB installation is needed.

## Build the Tesseract

```bash
tesseract build .
```

The Tesseract image is built directly on top of `mathworks/matlab:r2026a`
(see [tesseract_config.yaml](tesseract_config.yaml)). To use a different
MATLAB release, edit the `base_image` tag — no other change is necessary,
since `tesseract_api.py` auto-discovers the MATLAB binary in `/opt/matlab/`.

## Run (network license)

MATLAB's JVM requires more shared memory than Docker's 64 MB default.
Pass the license server via `--env`, plus shared memory and DNS mapping
via `--docker-args`:

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

### 2. Run with the license file mounted

```bash
tesseract run matlab-springmass \
  --user 0:0 \
  --env MATLAB_USERNAME=<your-username> \
  --env MLM_LICENSE_FILE=/licenses/license.lic \
  --volume /path/to/your/license.lic:/licenses/license.lic \
  --docker-args "--shm-size=512M --mac-address=XX:XX:XX:XX:XX:XX" \
  apply '{"inputs": {}}'
```

What the extra flags do:

- `--user 0:0` runs the container as root. This is needed for the next
  step. The MATLAB process itself does **not** run as root — it gets
  dropped to UID 1000 just before launch.
- `--env MATLAB_USERNAME=<your-username>` makes `tesseract_api.py`
  rename the `ubuntu` user in `/etc/passwd` (and the libnss_wrapper
  files in `/tmp/`) to `<your-username>` at startup, then spawn MATLAB
  as that user. This is required because MATLAB's file-based license
  check picks the `ubuntu` entry out of `/etc/passwd` regardless of the
  actual running UID — renaming it lets the check see the correct user.
- `--mac-address=XX:XX:XX:XX:XX:XX` must match the MAC your license
  was activated against.

## How It Works

The `tesseract_config.yaml` uses `mathworks/matlab:r2026a` directly as
the base image. The `.m` source file is copied in at build time, and
`tesseract_api.py` invokes it via `matlab -batch`:

```
Python (tesseract_api.py)
  ├── (if MATLAB_USERNAME set) Rename 'ubuntu' in /etc/passwd to <user>
  ├── Write input JSON → /tmp/<rand>/input.json
  ├── subprocess(user=<user>): matlab -batch "addpath(...); spring_mass_damper(...)"
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
├── test_cases/
│   └── test_apply.json         # Expected output for default inputs
├── visualize.py                # Visualization script
├── plots/                      # Generated plots (not committed)
└── README.md
```
