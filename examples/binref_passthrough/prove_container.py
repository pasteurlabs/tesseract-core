# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prove the binref passthrough survives containerization + HTTP.

Spawns the built image in a Docker container serving over HTTP with
``json+binref`` output, calls ``apply`` through the SDK client, and checks the
returned array. The SDK mounts ``output_path`` into the container and decodes
the binref reference the component emitted -- reading the very bytes the
component wrote to disk during apply, never round-tripped through memory on the
server.

Prereqs: ``tesseract build examples/binref_passthrough --tag demo``
"""

import tempfile
from pathlib import Path

import numpy as np

from tesseract_core import Tesseract

IMAGE = "binref_passthrough:demo"


def main() -> None:
    n, scale = 8, 2.5
    expected = np.arange(n, dtype=np.float64) * scale

    # 1. json+binref: the container forwards the on-disk buffer verbatim; the
    #    SDK resolves it against the mounted output_path.
    with tempfile.TemporaryDirectory(prefix="binref_shared_") as shared:
        output_path = Path(shared)
        with Tesseract.from_image(
            IMAGE, output_format="json+binref", output_path=output_path
        ) as t:
            arr = t.apply({"n": n, "scale": scale})["result"]

        np.testing.assert_array_equal(arr, expected)
        assert arr.dtype == np.float64

        binfiles = list(output_path.rglob("*.bin"))
        print("json+binref: files written into the mounted volume:")
        for f in binfiles:
            print("   ", f.relative_to(output_path), f"({f.stat().st_size} bytes)")
        assert binfiles, "expected a .bin file in the shared volume"
        print("[ok] binref forwarded verbatim ->", arr.tolist(), "\n")

    # 2. json+base64: same image, no shared volume needed. The container loads
    #    the buffer once and re-encodes it inline, like a normal Array field.
    with Tesseract.from_image(IMAGE, output_format="json+base64") as t:
        arr = t.apply({"n": n, "scale": scale})["result"]
    np.testing.assert_array_equal(arr, expected)
    print("[ok] base64 fallback over HTTP ->", arr.tolist())

    print("\n[ok] format-aware passthrough survived containerization + HTTP.")


if __name__ == "__main__":
    main()
