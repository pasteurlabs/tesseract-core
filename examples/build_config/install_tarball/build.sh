#!/bin/bash
set -e

pip download cowsay==6.1
tesseract --loglevel debug build . --keep-build-cache
