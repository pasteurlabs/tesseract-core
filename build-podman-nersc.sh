#!/bin/bash

source $SCRATCH/.venv/bin/activate
export DOCKER_HOST=$(podman info --format '{{.Host.RemoteSocket.Path}}')

# Make `docker` a valid executable that we can run from a python subprocess
mkdir -p bin
ln -sf $(which podman-hpc) bin/docker
export PATH=$PATH:$(pwd)/bin

tesseract --loglevel debug build examples/vectoradd --tag 0.1.0

echo
echo
echo "The python error is no big deal, we should have successfully built the image:"
podman-hpc images
