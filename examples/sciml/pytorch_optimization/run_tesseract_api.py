# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple development file used to check that the methods in `tesseract_api.py` execute."""

from tesseract_api import InputSchema, apply, jacobian

# check that apply works
input_schema = InputSchema(x=3.5, y=4.5, a=1.0, b=100.0)
outputs = apply(input_schema)
print(outputs.loss)

# check that jacobian works
input_schema = InputSchema(x=3.5, y=4.5, a=1.0, b=100.0)
outputs = jacobian(input_schema, jac_inputs={"x", "y"}, jac_outputs={"y"})
print(outputs)
