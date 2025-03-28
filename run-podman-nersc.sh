#!/bin/bash

source $SCRATCH/.venv/bin/activate

podman-hpc run -d -p 8080:8080 vectoradd:0.1.0 serve -p 8080
sleep 2
python3 <<'EOF'
from tesseract_core.sdk.tesseract import Tesseract

T = Tesseract("http://0.0.0.0:8080")
result = T.apply({"a": [1], "b": [2]})

print("Tesseract result:", result)
EOF

CONTAINER=$(podman-hpc ps -q)

podman-hpc kill $CONTAINER
podman-hpc rm $CONTAINER

