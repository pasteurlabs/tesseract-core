"""Rewrite relative image paths in README.md to absolute URLs for PyPI.

GitHub resolves relative paths against the branch being viewed, so README
previews work on PRs, but PyPI does not resolve them at all.
"""

import re
from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface

RAW_BASE_URL = "https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/"


class ReadmeHook(MetadataHookInterface):
    """Hatchling metadata hook that injects the rewritten README."""

    def update(self, metadata: dict) -> None:
        """Set the package readme to README.md with absolute image URLs."""
        readme = Path(self.root, "README.md").read_text(encoding="utf-8")
        readme = re.sub(
            r'(src="|srcset="|\]\()docs/', rf"\1{RAW_BASE_URL}docs/", readme
        )
        metadata["readme"] = {"content-type": "text/markdown", "text": readme}
