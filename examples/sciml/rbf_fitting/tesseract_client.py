# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import requests


class Client:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.host = host
        self.port = port

    @property
    def base_api_url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def request(self, endpoint: str, method: str = "GET", payload=None) -> dict:
        url = self.base_api_url + endpoint.lstrip("/")

        if payload is not None:
            body_json = {k: v for k, v in payload.items() if v is not None}
        else:
            body_json = None

        response = requests.request(method=method, url=url, json=body_json)
        response.raise_for_status()
        data = response.json()

        return data
