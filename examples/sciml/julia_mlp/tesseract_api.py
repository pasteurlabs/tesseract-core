# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import itertools

import juliacall
import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Float64, ShapeDType

# setup julia environment
jl = juliacall.newmodule("julia_mlp")
jl.seval("using Pkg")
jl.seval('Pkg.activate("package")')
jl.seval("Pkg.instantiate()")
jl.include("package/train.jl")


# define metrics container
class Metrics(BaseModel):
    train_mse: float = Field(description="Train mean squared error.")
    test_mse: float = Field(description="Test mean squared error.")


#
# Schemas
#


class InputSchema(BaseModel):
    layers: list[int] = Field(description="Layer structure", default=[100, 100])
    activation: str = Field(description="Activation function", default="sigmoid")
    n_epochs: int = Field(description="Number of epochs to train", default=1000)
    state: bytes | None = Field(
        description="The current architecture state", default=None
    )


class OutputSchema(BaseModel):
    parameters: list[Array[..., Float64]] = Field(
        description="Parameters of trained MLP."
    )
    metrics: Metrics = Field(description="Schema of accuracy metrics.")
    state: bytes = Field(description="The updated architecture state.")


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    layers = inputs.layers
    activation = inputs.activation
    n_epochs = inputs.n_epochs

    # load dataset
    x_data, y_data = jl.load_data()
    data_train, data_test = jl.split_train_test(x_data, y_data)
    x_train, y_train = data_train
    x_test, y_test = data_test

    # initialize and train
    activation_func = jl.eval(jl.Symbol(activation))
    model = jl.initialize_mlp(jl.Vector(layers), activation_func)
    if inputs.state is not None:
        # write state to BSON file and call julia method to load it in
        with open("state_to_load.bson", "wb") as file:
            decoded = base64.b64decode(inputs.state)
            file.write(decoded)
        jl.load_state(model, "state_to_load.bson")
    jl.train_mlp_b(model, x_train, y_train, n_epochs=n_epochs)

    # parameters
    params = jl.collect(jl.Flux.params(model))
    params_out = [np.array(params[i]) for i in range(len(params))]

    # metrics
    metrics = {
        "train_mse": jl.mse(model, x_train, y_train),
        "test_mse": jl.mse(model, x_test, y_test),
    }

    # state
    jl.save_state(model, "flux_mlp.bson")
    with open("flux_mlp.bson", "rb") as file:
        file_content = file.read()

    encoded = base64.b64encode(file_content)
    return OutputSchema(parameters=params_out, metrics=metrics, state=encoded)


#
# Optional endpoints
#


def abstract_eval(abstract_inputs):
    neuron_sizes = [
        1,
        *abstract_inputs.layers,
        1,
    ]  # 1D input, 1D out, hidden layers given

    param_shapes = []
    for i, j in itertools.pairwise(neuron_sizes):
        param_shapes += [
            ShapeDType(shape=[j, i], dtype="float64"),
            ShapeDType(shape=[j], dtype="float64"),
        ]
    return {"parameters": param_shapes}
