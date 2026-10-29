#!/usr/bin/env bash
# Redraws every figure in "Tesseract is now GPU-native" from the scripts and data
# in this directory. Needs numpy, matplotlib, and rsvg-convert (librsvg).
#
# rollouts.npz and ipc-benchmark.json hold the measured data, so this runs without
# a GPU. To recompute them, see rollouts.py and ipc-benchmark.py.
set -euo pipefail
cd "$(dirname "$0")"

for fig in multi_agent_dpc call_stack host_vs_ipc; do
    python fig_${fig}.py
    svg=${fig//_/-}.svg
    rsvg-convert -z 1.25 "$svg" -o "${svg%.svg}.png"
done
python ipc-benchmark.py --from-json ipc-benchmark.json --plot ipc-benchmark.png > /dev/null
