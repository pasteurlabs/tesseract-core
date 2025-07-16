# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from tesseract_core.runtime.experimental import mpa


class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    pass


def apply(inputs: InputSchema) -> OutputSchema:
    """This demonstrates logging metrics to MLflow."""
    with mpa.start_run():
        mpa.log_parameter("example_param", "value")

        for step in range(10):
            metric_value = step**2
            mpa.log_metric("squared_step", metric_value, step=step)

        text = "This is an output file we want to log as an artifact."
        with open("/tmp/artifact.txt", "w") as f:
            f.write(text)

        mpa.log_artifact("/tmp/artifact.txt")
    return OutputSchema()
