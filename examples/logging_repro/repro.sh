#!/bin/zsh

rm -rf reproducer_logs
mkdir -p reproducer_logs

uv run python -c '
from tesseract_core import Tesseract

with Tesseract.from_image("logging_repro", output_path="reproduce_logs") as tess:
    tess.apply({})
'

cat reproducer_logs/**/tesseract.log
