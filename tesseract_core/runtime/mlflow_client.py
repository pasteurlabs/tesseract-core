# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLflow HTTP client for interacting with MLflow Tracking Server without the mlflow package.

This module implements direct REST API access to MLflow, eliminating the need for the heavy
mlflow package dependency (which includes pandas, numpy, scipy, etc.). Only requires requests.

Key MLflow-specific behaviors:
- Timestamps must be Unix milliseconds (not seconds)
- Parameter values are limited to 6000 bytes, keys to 255 bytes
- Tags are passed as list of dicts: [{"key": "k", "value": "v"}], not plain dicts
- Artifact uploads require server started with --serve-artifacts flag
- Authentication can be embedded in URI: http://user:pass@host:port

Reference: https://mlflow.org/docs/latest/api_reference/rest-api.html
"""

import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests


class MLflowHTTPClient:
    """HTTP client for MLflow Tracking Server REST API.

    Maintains session state for efficient connection pooling and tracks the current run_id.
    Most methods default to using the current run_id if not explicitly provided.

    Context manager usage automatically ends the run with status FAILED on exception,
    FINISHED on normal exit.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_id: str = "0",
        timeout: int = 30,
    ) -> None:
        """Initialize MLflow HTTP client.

        Args:
            tracking_uri: MLflow server URI. Can include credentials: http://user:pass@host:port
            experiment_id: Experiment ID (default "0"). Note: This is a string, not an int.
            timeout: Request timeout in seconds
        """
        self.tracking_uri = tracking_uri.rstrip("/")
        self.experiment_id = experiment_id
        self.timeout = timeout
        self.session = requests.Session()
        self.run_id: str | None = None

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Make an HTTP request to the MLflow API.

        Automatically prepends /api/ to endpoint and extracts MLflow error messages from responses.

        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            endpoint: API endpoint path like "2.0/mlflow/runs/create" (without /api/ prefix)
            json_data: JSON data to send in request body
            **kwargs: Additional arguments for requests (e.g., params for GET)
        """
        url = urljoin(self.tracking_uri, f"/api/{endpoint}")

        response = self.session.request(
            method=method,
            url=url,
            json=json_data,
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            # Add response body to error message for debugging
            error_msg = f"MLflow API request failed: {e}"
            try:
                error_detail = response.json()
                if "message" in error_detail:
                    error_msg += f"\nMLflow error: {error_detail['message']}"
            except Exception:
                error_msg += f"\nResponse: {response.text}"
            raise requests.HTTPError(error_msg, response=response) from e

        return response

    def create_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Create a new run and set it as the current run_id.

        Note: Tags dict is converted to MLflow's list format: [{"key": k, "value": v}]

        Args:
            run_name: Optional name for the run
            tags: Optional dictionary of tags to attach to the run

        Returns:
            Run ID (also stored in self.run_id)
        """
        data: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "start_time": int(time.time() * 1000),  # Unix timestamp in milliseconds
        }

        if run_name:
            data["run_name"] = run_name

        if tags:
            data["tags"] = [{"key": k, "value": v} for k, v in tags.items()]

        response = self._make_request("POST", "2.0/mlflow/runs/create", json_data=data)
        result = response.json()
        self.run_id = result["run"]["info"]["run_id"]
        assert self.run_id is not None
        return self.run_id

    def update_run(
        self,
        run_id: str,
        status: str = "FINISHED",
        end_time: int | None = None,
    ) -> None:
        """Update a run's status and end time.

        Args:
            run_id: ID of the run to update
            status: RUNNING | SCHEDULED | FINISHED | FAILED | KILLED
            end_time: Unix timestamp in milliseconds (not seconds). Defaults to now.
        """
        if end_time is None:
            end_time = int(time.time() * 1000)

        data = {
            "run_id": run_id,
            "status": status,
            "end_time": end_time,
        }

        self._make_request("POST", "2.0/mlflow/runs/update", json_data=data)

    def end_run(
        self,
        run_id: str | None = None,
        status: str = "FINISHED",
    ) -> None:
        """End a run and clear self.run_id if it was the current run.

        Args:
            run_id: Defaults to self.run_id. Raises RuntimeError if both are None.
            status: FINISHED | FAILED | KILLED
        """
        if run_id is None:
            if self.run_id is None:
                raise RuntimeError("No active run to end")
            run_id = self.run_id

        self.update_run(run_id, status=status)

        if run_id == self.run_id:
            self.run_id = None

    def log_param(
        self,
        key: str,
        value: Any,
        run_id: str | None = None,
    ) -> None:
        """Log a parameter. Value is always converted to string.

        MLflow limits: key max 255 bytes, value max 6000 bytes (enforced server-side).

        Args:
            key: Parameter name (max 255 bytes)
            value: Parameter value (will be converted to string, max 6000 bytes)
            run_id: Defaults to self.run_id
        """
        if run_id is None:
            if self.run_id is None:
                raise RuntimeError("No active run. Call create_run() first.")
            run_id = self.run_id

        data = {
            "run_id": run_id,
            "key": key,
            "value": str(value),
        }

        self._make_request("POST", "2.0/mlflow/runs/log-parameter", json_data=data)

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
        timestamp: int | None = None,
        run_id: str | None = None,
    ) -> None:
        """Log a metric. Value is cast to float.

        Args:
            key: Metric name
            value: Metric value (will be cast to float)
            step: Optional step/iteration number for time-series metrics
            timestamp: Unix milliseconds (not seconds). Defaults to now.
            run_id: Defaults to self.run_id
        """
        if run_id is None:
            if self.run_id is None:
                raise RuntimeError("No active run. Call create_run() first.")
            run_id = self.run_id

        if timestamp is None:
            timestamp = int(time.time() * 1000)

        data = {
            "run_id": run_id,
            "key": key,
            "value": float(value),
            "timestamp": timestamp,
        }

        if step is not None:
            data["step"] = step

        self._make_request("POST", "2.0/mlflow/runs/log-metric", json_data=data)

    def log_batch(
        self,
        metrics: list[dict[str, Any]] | None = None,
        params: list[dict[str, Any]] | None = None,
        tags: list[dict[str, str]] | None = None,
        run_id: str | None = None,
    ) -> None:
        """Log multiple items in a single request. More efficient than individual calls.

        Args:
            metrics: Each dict needs: key, value, timestamp. Optional: step
            params: Each dict needs: key, value
            tags: Each dict needs: key, value
            run_id: Defaults to self.run_id
        """
        if run_id is None:
            if self.run_id is None:
                raise RuntimeError("No active run. Call create_run() first.")
            run_id = self.run_id

        data: dict[str, Any] = {"run_id": run_id}

        if metrics:
            data["metrics"] = metrics
        if params:
            data["params"] = params
        if tags:
            data["tags"] = tags

        self._make_request("POST", "2.0/mlflow/runs/log-batch", json_data=data)

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Upload an artifact file using PUT request to mlflow-artifacts endpoint.

        IMPORTANT: Requires MLflow server with --serve-artifacts flag

        This method implements the same approach as MLflow's http_artifact_repo.py

        Args:
            local_path: Path to the local file to upload
            artifact_path: Optional subdirectory within run's artifact root
            run_id: Defaults to self.run_id

        Raises:
            requests.HTTPError: If upload fails
        """
        if run_id is None:
            if self.run_id is None:
                raise RuntimeError("No active run. Call create_run() first.")
            run_id = self.run_id

        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {local_path}")

        # Construct the artifact upload URL following MLflow's convention
        # The endpoint format matches http_artifact_repo.py implementation
        import posixpath

        file_name = local_path.name
        paths = (artifact_path, file_name) if artifact_path else (file_name,)
        endpoint = posixpath.join("/", *paths)

        # Base URL for artifacts with run_id
        base_url = f"{self.tracking_uri}/api/2.0/mlflow-artifacts/artifacts/{self.experiment_id}/{run_id}/artifacts"
        url = f"{base_url}{endpoint}"

        # Guess MIME type for Content-Type header (matches MLflow's _guess_mime_type)
        import mimetypes

        mime_type, _ = mimetypes.guess_type(file_name)
        if mime_type is None:
            mime_type = "application/octet-stream"

        # Upload using PUT with Content-Type header (matches MLflow SDK)
        with open(local_path, "rb") as f:
            response = self.session.put(
                url,
                data=f,
                headers={"Content-Type": mime_type},
                timeout=self.timeout,
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            error_msg = f"Failed to upload artifact: {e}"
            # If proxied artifact access is not available, provide a helpful message
            if response.status_code == 503:
                error_msg += (
                    "\n\nNote: Artifact upload requires the MLflow server to be started "
                    "with the --serve-artifacts flag for proxied artifact access."
                )
            try:
                error_detail = response.json()
                if "message" in error_detail:
                    error_msg += f"\nMLflow error: {error_detail['message']}"
            except Exception:
                error_msg += f"\nResponse: {response.text}"
            raise requests.HTTPError(error_msg, response=response) from e

    def get_run(self, run_id: str | None = None) -> dict[str, Any]:
        """Get run metadata including params, metrics, tags, and status.

        Args:
            run_id: Defaults to self.run_id

        Returns:
            Dict with structure: {"run": {"info": {...}, "data": {...}}}
        """
        if run_id is None:
            if self.run_id is None:
                raise RuntimeError("No active run")
            run_id = self.run_id

        response = self._make_request(
            "GET",
            "2.0/mlflow/runs/get",
            params={"run_id": run_id},
        )
        return response.json()

    def close(self) -> None:
        """Close the HTTP session and release connection pool resources."""
        self.session.close()

    def __enter__(self) -> "MLflowHTTPClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End active run (FAILED on exception, FINISHED otherwise) and close session."""
        if self.run_id is not None:
            status = "FAILED" if exc_type is not None else "FINISHED"
            try:
                self.end_run(status=status)
            except Exception:
                pass  # Don't raise during cleanup
        self.close()
