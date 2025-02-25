# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Simple development file used to show how to make requests to the julia_flux_mlp tesseract
# assumes the tesseract is located at: http://127.0.0.1:8000

import base64

import numpy as np
import requests


class Client:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.host = host
        self.port = port

    @property
    def get_base_api_url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def request(self, endpoint: str, method: str = "GET", payload=None) -> dict:
        url = self.get_base_api_url + endpoint.lstrip("/")

        if payload is not None:
            body_json = {k: v for k, v in payload.items() if v is not None}
        else:
            body_json = None

        response = requests.request(method=method, url=url, json=body_json)
        response.raise_for_status()
        data = response.json()

        return data


# NOTE: pass port argument to wherever the tesseract is served
client = Client()

data = client.request(
    "apply",
    method="POST",
    payload={"inputs": {"n_epochs": 1}},
)
buffer = data["state"]["data"]["buffer"]
decoded_buffer = base64.b64decode(buffer)
result = np.frombuffer(decoded_buffer)
