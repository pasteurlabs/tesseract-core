"""Custom hook to inject config from runtime pyproject.toml into the main pyproject.toml.

This allows us to specify runtime dependencies and scripts in only one place.
"""

from pathlib import Path

import toml
from hatchling.metadata.plugin.interface import MetadataHookInterface

RUNTIME_PYPROJECT_PATH = "tesseract_core/runtime/meta/pyproject.toml"


class RuntimeDepenencyHook(MetadataHookInterface):
    PLUGIN_NAME = "runtime-deps"

    def update(self, metadata):
        runtime_metadata = toml.load(Path(self.root) / RUNTIME_PYPROJECT_PATH)
        metadata["optional-dependencies"]["runtime"] = runtime_metadata["project"][
            "dependencies"
        ]
        metadata["scripts"].update(runtime_metadata["project"]["scripts"])
        return metadata
