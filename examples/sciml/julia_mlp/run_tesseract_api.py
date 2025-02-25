# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple development file used to check that the methods in `tesseract_api.py` execute."""

from tesseract_api import InputSchema, apply

# check that API works with no state passed
input_schema = InputSchema(n_epochs=100)
outputs = apply(input_schema)
state = outputs.state
print(outputs.metrics)

# check that API works with state passed
new_inputs = InputSchema(n_epochs=100, state=state)
new_outputs = apply(new_inputs)
new_state = new_outputs.state
print(new_outputs.metrics)
