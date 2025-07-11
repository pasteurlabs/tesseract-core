# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-generate test data for this Tesseract."""

from pathlib import Path

import numpy as np
from pydantic import RootModel

from tesseract_core.runtime import Array, Float32

rng = np.random.default_rng(0)

OUTDIR = Path(__file__).parent / "testdata"


class TestArray(RootModel):
    root: Array[(None, 3), Float32]


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    binref_uuid = "a2af4236-15c5-40ec-9014-b79f9cbad50c"
    (OUTDIR / f"{binref_uuid}.bin").write_bytes(b"")

    for idx in range(10):
        if idx < 9:
            data = rng.random((3, 3))
        else:
            # Create one sample with a different shape
            data = rng.random((4, 3))

        # Create some arrays of all supported encodings
        if idx % 3 == 0:
            encoding = "json"
        elif idx % 3 == 1:
            encoding = "base64"
        else:
            encoding = "binref"

        out_json = TestArray(data).model_dump_json(
            indent=2,
            context={
                "array_encoding": encoding,
                "base_dir": OUTDIR,
                "__binref_uuid": binref_uuid,
            },
        )

        with open(OUTDIR / f"sample_{idx}.json", "w") as f:
            f.write(out_json)
