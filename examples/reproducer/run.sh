#!/bin/bash
set -e

tesseract --loglevel debug build .

docker system prune --force

tesseract run reproducer apply '{"inputs":{}}' --output-path outputs
