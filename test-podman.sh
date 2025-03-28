#!/bin/zsh

alias docker=podman
#export DOCKER_HOST='unix:///var/folders/_t/ybrp_vjs7dzdmzlc9t3nydd40000gn/T/podman/podman-machine-default-api.sock'
echo $DOCKER_HOST
uv run tesseract build examples/vectoradd --tag 0.1.0
uv run tesseract run vectoradd:0.1.0 apply '{"inputs": {"a": [1], "b": [2]}}'

