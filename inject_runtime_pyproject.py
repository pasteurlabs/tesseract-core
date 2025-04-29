"""Custom hook to inject config from runtime pyproject.toml into the main pyproject.toml.

This allows us to specify runtime dependencies and scripts in only one place.
"""

from pathlib import Path

import toml
from hatchling.metadata.plugin.interface import MetadataHookInterface

RUNTIME_PYPROJECT_PATH = "tesseract_core/runtime/meta/pyproject.toml"

BASE_OPTIONAL_DEPS = {
    "docs": [
        "sphinx",
        "sphinx_autodoc_typehints",
        "furo",
        "myst-parser",
        "sphinx_click",
        "autodoc_pydantic",
        "sphinx_design",
        "sphinx_copybutton",
        "sphinxext_opengraph",
        "tesseract-core[runtime]",
    ],
    "dev": [
        "docker",
        "fastapi",
        "httpx",  # required by fastapi test client
        "jsf",
        "requests",
        "numpy",
        "pre-commit",
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "moto[server]",
        "aiobotocore>=2.19.0",  # without this pip dependency resolution fails
        "typeguard",
        # also add all other extras here
        "tesseract-core[runtime]",
        "tesseract-core[docs]",
    ],
    "runtime": [],
}

BASE_SCRIPTS = {
    "tesseract": "tesseract_core.sdk.cli:entrypoint",
}


class RuntimeDepenencyHook(MetadataHookInterface):
    """Injects runtime dependencies and scripts from a separate pyproject.toml file."""

    PLUGIN_NAME = "runtime-deps"

    def update(self, metadata: dict) -> dict:
        """Update the metadata with runtime dependencies and scripts."""
        runtime_metadata = toml.load(Path(self.root) / RUNTIME_PYPROJECT_PATH)
        metadata["optional-dependencies"] = {
            **BASE_OPTIONAL_DEPS,
            "runtime": runtime_metadata["project"]["dependencies"],
        }
        metadata["scripts"] = {**BASE_SCRIPTS, **runtime_metadata["project"]["scripts"]}
        return metadata
