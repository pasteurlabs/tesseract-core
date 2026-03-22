# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import urllib.request
from urllib.error import URLError

from pydantic import BaseModel, Field


class InputSchema(BaseModel):
    url: str = Field(
        default="https://pasteurlabs.ai/",
        description="URL to fetch. Used to verify outbound network access.",
    )


class OutputSchema(BaseModel):
    reachable: bool = Field(description="Whether the URL was reachable.")
    status_code: int | None = Field(
        description="HTTP status code returned by the server, or null if unreachable."
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Fetch a URL and report whether it is reachable."""
    try:
        with urllib.request.urlopen(inputs.url, timeout=5) as response:
            return OutputSchema(reachable=True, status_code=response.status)
    except URLError:
        return OutputSchema(reachable=False, status_code=None)
