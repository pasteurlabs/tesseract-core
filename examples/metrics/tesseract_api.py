# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from tesseract_core.runtime.experimental import log_artifact, log_metric, log_parameter


class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    pass


def apply(inputs: InputSchema) -> OutputSchema:
    """This demonstrates logging metrics to MLflow."""
    log_parameter("example_param", "value")

    for step in range(10):
        metric_value = step**2
        log_metric("squared_step", metric_value, step=step)

    text = "This is an output file we want to log as an artifact."
    with open("/tmp/artifact.txt", "w") as f:
        f.write(text)

    log_artifact("/tmp/artifact.txt")
    return OutputSchema()
