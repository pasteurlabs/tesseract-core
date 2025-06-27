# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mlflow
from pydantic import BaseModel, Field


class InputSchema(BaseModel):
    name: str = Field(description="Name of the person you want to greet.")


class OutputSchema(BaseModel):
    greeting: str = Field(description="A greeting!")


def apply(inputs: InputSchema) -> OutputSchema:
    """Greet a person whose name is given as input."""
    with mlflow.start_run():
        for step in range(10):
            metric_value = step**2  # Example metric
            mlflow.log_metric("squared_step", metric_value, step=step)
    return OutputSchema(greeting=f"Hello {inputs.name}!")
