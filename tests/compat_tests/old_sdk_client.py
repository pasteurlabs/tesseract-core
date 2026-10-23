# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call a served Tesseract with whichever tesseract-core SDK is installed.

test_version_compat.py runs this with an older SDK, so it may only use API
that every supported SDK has. Usage::

    python old_sdk_client.py URL SERVER_OUTPUT_PATH < arguments.json

Prints the result of each call as JSON.
"""

import json
import sys

import numpy as np

import tesseract_core
from tesseract_core import Tesseract


def to_arrays(tree):
    if isinstance(tree, dict):
        return {k: to_arrays(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return np.asarray(tree, dtype=np.float32)
    return tree


def to_json(tree):
    if isinstance(tree, dict):
        return {k: to_json(v) for k, v in tree.items()}
    if isinstance(tree, np.ndarray | np.generic):
        return tree.tolist()
    return tree


def main() -> None:
    url, server_output_path = sys.argv[1:3]
    args = json.load(sys.stdin)
    inputs = to_arrays(args["inputs"])

    tess = Tesseract.from_url(url, server_output_path=server_output_path)
    results = {
        "sdk_version": tesseract_core.__version__,
        "health": tess.health(),
        "apply": tess.apply(inputs),
        "abstract_eval": tess.abstract_eval(args["abstract_inputs"]),
        "jacobian": tess.jacobian(inputs, ["a", "b"], ["result"]),
        "jacobian_vector_product": tess.jacobian_vector_product(
            inputs, ["a"], ["result"], to_arrays(args["tangent_vector"])
        ),
        "vector_jacobian_product": tess.vector_jacobian_product(
            inputs, ["a"], ["result"], to_arrays(args["cotangent_vector"])
        ),
    }
    try:
        tess.apply(to_arrays(args["invalid_inputs"]))
    except ValueError as e:
        results["invalid_inputs_error"] = f"{type(e).__name__}: {e}"
    else:
        results["invalid_inputs_error"] = None

    json.dump(to_json(results), sys.stdout)


if __name__ == "__main__":
    main()
