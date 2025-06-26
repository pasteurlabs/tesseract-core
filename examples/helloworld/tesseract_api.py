# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

from tesseract_core.runtime import FileReference


class InputSchema(BaseModel):
    name: str = Field(description="Name of the person you want to greet.")
    input_file: FileReference = Field(description="A file that can be used as input.")


class OutputSchema(BaseModel):
    greeting: str = Field(description="A greeting!")
    output_file: FileReference = Field(description="We'll dump some output here.")


def apply(inputs: InputSchema) -> OutputSchema:
    """Greet a person whose name is given as input."""
    print(f"Received input: {inputs}")
    print(f"Dumped input: {inputs.model_dump()}")
    # read the file to demonstrate usage of FileReference
    with inputs.input_file.open() as f:
        file_content = f.read()
        print(f"File content: {file_content}")
    # Create output file "test_out.txt"
    output_file = inputs.input_file.with_name("test_out.txt")
    with output_file.open("w") as f:
        f.write("This is some output content.")
    return OutputSchema(
        greeting=f"Hello {inputs.name}! Read file: '{inputs.input_file}' with content '{file_content}'",
        output_file=output_file,
    )
