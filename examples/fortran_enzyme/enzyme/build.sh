#!/bin/bash
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build pipeline: Fortran -> LFortran -> LLVM IR -> Enzyme AD -> shared library
#
# This script implements the full compilation pipeline:
#   1. LFortran compiles Fortran to LLVM IR
#   2. LLVM opt cleans up the IR (-O1)
#   3. Clang compiles the C wrapper (with Enzyme calls) to LLVM IR
#   4. llvm-link merges the Fortran and C IR modules
#   5. Enzyme LLVM pass generates derivative code
#   6. Clang compiles the differentiated IR to a shared library

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENZYME_LIB="${ENZYME_LIB:-/usr/local/lib/LLVMEnzyme-19.so}"
OUTPUT="${1:-${SCRIPT_DIR}/libheat_ad.so}"

echo "=== Step 1: LFortran -> LLVM IR ==="
lfortran --show-llvm --no-array-bounds-checking \
    "${SCRIPT_DIR}/heat_step.f90" > /tmp/heat_step.ll

echo "=== Step 2: Optimize IR ==="
opt -O1 -S /tmp/heat_step.ll -o /tmp/heat_step_opt.ll

echo "=== Step 3: Compile C wrapper -> LLVM IR ==="
clang -emit-llvm -S -O1 "${SCRIPT_DIR}/wrapper.c" -o /tmp/wrapper.ll

echo "=== Step 4: Link IR modules ==="
llvm-link /tmp/wrapper.ll /tmp/heat_step_opt.ll -S -o /tmp/combined.ll

echo "=== Step 5: Enzyme AD pass ==="
opt --load-pass-plugin="${ENZYME_LIB}" -passes=enzyme \
    -S /tmp/combined.ll -o /tmp/ad.ll

echo "=== Step 6: Compile to shared library ==="
clang -shared -O2 /tmp/ad.ll -o "${OUTPUT}" -lm

echo "=== Built ${OUTPUT} ==="
