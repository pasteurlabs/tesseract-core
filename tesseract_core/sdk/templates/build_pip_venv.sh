#!/bin/bash

# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit immediately if a command exits with a non-zero status

# python_version and inherit_base_image_packages are mutually exclusive (enforced
# at config validation time), so at most one of these branches sets venv options.
if [ -n "${TESSERACT_PYTHON_VERSION:-}" ]; then
    uv python install "$TESSERACT_PYTHON_VERSION"
    uv venv --python "$TESSERACT_PYTHON_VERSION" /python-env
elif [ "${TESSERACT_INHERIT_BASE_IMAGE_PACKAGES:-0}" = "1" ]; then
    uv venv --system-site-packages /python-env
else
    uv venv /python-env
fi
source /python-env/bin/activate

# Install dependencies. Local dependencies (if any) are rewritten into the
# requirements file as paths under ./local_requirements/, so a single install
# from the requirements file covers both remote and local dependencies.
uv -v pip install --compile-bytecode -r tesseract_requirements.txt

# HACK: If `tesseract_core` is part of tesseract_requirements.txt, it may install an incompatible version
# of the runtime from PyPI. We remove the runtime folder and install the local version instead.
runtime_path=$(python -c "import tesseract_core; print(tesseract_core.__file__.replace('__init__.py', ''))" || true)
if [ -d "$runtime_path" ]; then
    rm -rf "$runtime_path"/runtime
fi

uv -v pip install --compile-bytecode ./tesseract_runtime

# Install pip itself into the virtual environment for use by any custom build steps
uv pip install pip

if [ -n "${TESSERACT_PYTHON_VERSION:-}" ]; then
    # The venv's python binary is a symlink into the uv-managed installation
    # (e.g. /root/.local/share/uv/python/cpython-3.12-.../). Merge that
    # installation (stdlib, binary) into /python-env so the venv is fully
    # self-contained after Docker multi-stage COPY (which preserves symlinks).
    UV_PYTHON_DIR=$(dirname "$(dirname "$(readlink -f /python-env/bin/python)")")
    rm /python-env/bin/python /python-env/bin/python3 /python-env/bin/python3.*
    cp -a "$UV_PYTHON_DIR"/bin/* /python-env/bin/
    cp -a "$UV_PYTHON_DIR"/lib/* /python-env/lib/
    cp -a "$UV_PYTHON_DIR"/include/* /python-env/include/

    # pyvenv.cfg still points `home` at the uv-managed installation, which does
    # not exist after the Docker multi-stage COPY. Point it at the now-local
    # binaries so the interpreter can locate its stdlib.
    sed -i "s|^home = .*|home = /python-env/bin|" /python-env/pyvenv.cfg
fi
