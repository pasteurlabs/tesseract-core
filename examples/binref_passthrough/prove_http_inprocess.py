# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prove the binref passthrough survives a real HTTP round-trip, in-process.

Uses the exact server machinery a container runs (create_rest_api + the same
serialisation path), driven via FastAPI's TestClient, and decodes the output
with the SDK's own decoder. This is the fast proof; run ``prove_container.py``
for the full build+serve-in-Docker version.

Exercises all three negotiable formats to show the serializer is format-aware:

* ``json+binref``  -> reference forwarded verbatim (zero-copy, no load)
* ``json+base64``  -> buffer loaded once and re-encoded, like a normal Array
* ``json``         -> buffer loaded once and inlined, like a normal Array
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from tesseract_core.runtime.config import get_config, update_config
from tesseract_core.runtime.core import load_module_from_path
from tesseract_core.runtime.serve import create_rest_api
from tesseract_core.sdk.tesseract import _decode_array

HERE = Path(__file__).parent
API_FILE = HERE / "tesseract_api.py"

N, SCALE = 8, 2.5
EXPECTED = np.arange(N, dtype=np.float64) * SCALE


def _apply(client: TestClient, accept: str) -> dict:
    resp = client.post(
        "/apply",
        json={"inputs": {"n": N, "scale": SCALE}},
        headers={"Accept": accept},
        params={"run_id": "demo"},
    )
    assert resp.status_code == 200, (accept, resp.status_code, resp.text)
    return resp.json()["result"]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="binref_out_") as tmp:
        # Configure the runtime as `tesseract serve` would.
        update_config(
            api_path=str(API_FILE),
            output_path=str(tmp),
            output_format="json+binref",
        )
        # get_config resolves symlinks (e.g. /var -> /private/var on macOS).
        output_path = Path(get_config().output_path)

        api_module = load_module_from_path(API_FILE)
        client = TestClient(create_rest_api(api_module))
        assert client.get("/health").status_code == 200

        # 1. binref: forwarded verbatim, decoded from the shared dir.
        result = _apply(client, "application/json+binref")
        print("json+binref result:")
        print(json.dumps(result, indent=2))
        assert result["data"]["encoding"] == "binref"
        assert result["shape"] == [N] and result["dtype"] == "float64"
        decoded = _decode_array(result, output_path=output_path)
        np.testing.assert_array_equal(decoded, EXPECTED)
        print("[ok] binref forwarded verbatim ->", decoded.tolist(), "\n")

        # 2. base64: serializer loaded the buffer once and re-encoded it.
        result = _apply(client, "application/json+base64")
        assert result["data"]["encoding"] == "base64"
        decoded = _decode_array(result)
        np.testing.assert_array_equal(decoded, EXPECTED)
        print("[ok] base64 fallback (load + re-encode) ->", decoded.tolist())

        # 3. json: buffer loaded once and inlined as a list of numbers.
        result = _apply(client, "application/json")
        assert result["data"]["encoding"] == "json"
        assert result["data"]["buffer"] == EXPECTED.tolist()
        decoded = _decode_array(result)
        np.testing.assert_array_equal(decoded, EXPECTED)
        print("[ok] json fallback (load + inline) ->", decoded.tolist())

        print("\n[ok] format-aware serializer honoured all three Accept headers.")


if __name__ == "__main__":
    main()
