# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from pydantic import BaseModel

import mlflow

class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    pass

_logger = logging.getLogger("mlflow.tracking.context.registry")

def apply(inputs: InputSchema) -> OutputSchema:
    """This demonstrates logging parameters, metrics and artifacts."""
    print("This is a message from the apply function.")
    mlflow.start_run()
    return OutputSchema()
