# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

#
# Schemas
#


class InputSchema(BaseModel):
    a: float
    b: float


class OutputSchema(BaseModel):
    # Sum computed with torch and passed through numpy, so both packages from the
    # committed pylock.toml must be installed and importable for apply to succeed.
    result: float


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    import numpy as np
    import torch

    total = torch.tensor(inputs.a) + torch.tensor(inputs.b)
    return OutputSchema(result=float(np.asarray(total).item()))
