# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry of scheduled deprecation removals ("tombstones").

A tombstone pins a deprecation to the version it must be
gone by, and :func:`overdue_tombstones` reports any that are due. The
accompanying test checks this against the version being cut, so a release PR
with active tombstones fails.

To schedule a removal:

1. Add a :class:`Tombstone` to :data:`TOMBSTONES`, targeting a concrete future
   version.
2. When the release-PR test starts failing, delete the deprecated code *and* its
   tombstone in the same PR.

Deprecations always target a *minor* version (no breaking changes on patch releases).
"""

import re
from pathlib import Path
from typing import NamedTuple

from packaging.version import VERSION_PATTERN, InvalidVersion, Version


class Tombstone(NamedTuple):
    """A deprecation scheduled for removal by a specific version."""

    remove_at: str
    """Version by which the deprecated code must be gone (e.g. ``"1.13.0"``)."""

    what: str
    """Short name of the deprecated feature, shown when the removal is due."""

    hint: str
    """What to delete, so the person hitting the failure knows where to look."""


# Deprecations awaiting removal. Delete an entry together with its code once the
# scheduled version arrives.
TOMBSTONES: tuple[Tombstone, ...] = (
    # Example:
    # Tombstone(
    #     remove_at="0.99.0",
    #     what="--foo alias from `tesseract build`",
    #     hint="remove backend support from engine.py, too"
    # ),
)


def _repo_changelog_path() -> Path:
    """Path to the repository ``CHANGELOG.md``, if running from a source checkout."""
    return Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def _version_from_changelog(changelog: str) -> Version | None:
    """Return the topmost ``## [version]`` entry in a changelog, or None if absent."""
    match = re.search(
        rf"^\#\#\s*\[({VERSION_PATTERN})\]",
        changelog,
        re.MULTILINE | re.VERBOSE | re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        return Version(match.group(1))
    except InvalidVersion:
        return None


def latest_changelog_version() -> Version:
    """Return the topmost ``## [version]`` entry in a changelog.

    On a release PR the changelog is regenerated with the version being cut at the
    top, so this is the version a merge would release.
    """
    changelog_path = _repo_changelog_path()
    if not changelog_path.exists():
        raise FileNotFoundError(
            f"Could not find {changelog_path} to check for overdue deprecations"
        )
    changelog = changelog_path.read_text(encoding="utf-8")
    version = _version_from_changelog(changelog)
    if version is None:
        raise ValueError(f"Could not find a valid version entry in {changelog_path}")
    return version


def overdue_tombstones(version: Version) -> list[Tombstone]:
    """Return tombstones whose removal is due at or before ``version``."""
    return [t for t in TOMBSTONES if version >= Version(t.remove_at)]
