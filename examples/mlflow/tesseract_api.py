# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mlflow
from pydantic import BaseModel


class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    pass


def apply(inputs: InputSchema) -> OutputSchema:
    """This demonstrates logging metrics to MLflow."""
    print(f"MLflow logging to URI: {mlflow.get_tracking_uri()}")

    # Marking the start of an MLflow.
    with mlflow.start_run():
        for step in range(10):
            metric_value = step**2
            mlflow.log_metric("squared_step", metric_value, step=step)

        text = "This is an output file we want to log as an artifact."
        with open("artifact.txt", "w") as f:
            f.write(text)

        mlflow.log_artifact("artifact.txt")
    return OutputSchema()
