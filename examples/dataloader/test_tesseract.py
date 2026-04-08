# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np

from tesseract_core import Tesseract

here = Path(__file__).parent
testdata_dir = here / "testdata"
num_samples = len(list(testdata_dir.glob("sample_*.json")))

# Absolute glob pattern for local runs
data_glob = f"@{testdata_dir}/sample_*.json"


def check_apply_result(result):
    assert len(result["data"]) == num_samples
    data_sum = np.asarray(result["data_sum"])
    assert data_sum.shape == (3,)
    # data_sum is the sum of each sample's column sums; must be positive
    assert np.all(data_sum > 0)


def check_jacobian_result(result):
    jac = np.asarray(result["data_sum"]["data.[0]"])
    # jacobian of data_sum w.r.t. data.[0] has shape (3, *data[0].shape)
    assert jac.ndim == 3
    assert jac.shape[0] == 3


with Tesseract.from_tesseract_api("tesseract_api.py") as tess:
    result = tess.apply({"data": data_glob})
    check_apply_result(result)

    result = tess.jacobian(
        inputs={"data": data_glob},
        jac_inputs=["data.[0]"],
        jac_outputs=["data_sum"],
    )
    check_jacobian_result(result)


with Tesseract.from_image(
    "dataloader",
    volumes=[f"{testdata_dir}:/mnt/testdata:ro"],
) as tess:
    result = tess.apply({"data": "@/mnt/testdata/sample_*.json"})
    check_apply_result(result)

    result = tess.jacobian(
        inputs={"data": "@/mnt/testdata/sample_*.json"},
        jac_inputs=["data.[0]"],
        jac_outputs=["data_sum"],
    )
    check_jacobian_result(result)
