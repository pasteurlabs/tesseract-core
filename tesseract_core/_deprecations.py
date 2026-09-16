# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry of scheduled deprecation removals ("tombstones").

A tombstone pins a deprecation to the date it must be gone by, and
:func:`overdue_tombstones` reports any that are due. The accompanying test checks
this against the release being cut, so a release PR with active tombstones fails.

To schedule a removal:

1. Add a :class:`Tombstone` to :data:`TOMBSTONES`, targeting a concrete future
   date.
2. When the release-PR test starts failing, delete the deprecated code *and* its
   tombstone in the same PR.

Removals only ever happen on *minor* (or major) releases, never on patch
releases, so that a hotfix is never forced to drop a deprecated shim. A tombstone
is therefore overdue only when the release being cut both postdates its removal
date and bumps the minor version relative to the previous release.

Dating removals rather than pinning them to a version gives users a predictable
migration window. Releases can land in quick succession, so a version-based
target could pass within days of a deprecation warning first appearing.
"""

import re
from datetime import date
from pathlib import Path
from typing import NamedTuple

from packaging.version import InvalidVersion, Version


class Tombstone(NamedTuple):
    """A deprecation scheduled for removal after a specific date."""

    remove_after: str
    """Date by which the deprecated code must be gone (ISO ``"YYYY-MM-DD"``)."""

    what: str
    """Short name of the deprecated feature, shown when the removal is due."""

    hint: str
    """What to delete, so the person hitting the failure knows where to look."""


class Release(NamedTuple):
    """A dated release, as parsed from a ``## [version] - YYYY-MM-DD`` entry."""

    version: Version
    when: date


# Deprecations awaiting removal. Delete an entry together with its code once the
# scheduled date passes.
TOMBSTONES: tuple[Tombstone, ...] = (
    # Example:
    # Tombstone(
    #     remove_after="2026-12-01",
    #     what="--foo alias from `tesseract build`",
    #     hint="remove backend support from engine.py, too"
    # ),
    Tombstone(
        remove_after="2026-12-01",
        what="'python-pip' requirements provider alias",
        hint=(
            "Remove the 'python-pip' -> 'uv-pip' normalization in "
            "tesseract_core/sdk/api_parse.py (_normalize_provider) and its test."
        ),
    ),
    Tombstone(
        remove_after="2026-12-01",
        what="build_config.python_version alias",
        hint=(
            "Remove the deprecated TesseractBuildConfig.python_version field and its "
            "forwarding in _validate_python_version_provider "
            "(tesseract_core/sdk/api_parse.py); python_version now lives on "
            "PipRequirements. Update the tests, too."
        ),
    ),
    Tombstone(
        remove_after="2026-12-01",
        what="Tesseract(url) constructor",
        hint=(
            "Remove the deprecated Tesseract.__init__ shim in "
            "tesseract_core/sdk/tesseract.py (callers use Tesseract.from_url / "
            "from_image / from_tesseract_api); make __init__ raise instead. "
            "Update the tests, too."
        ),
    ),
    Tombstone(
        remove_after="2026-12-01",
        what="InputFileReference / OutputFileReference aliases",
        hint=(
            "Remove InputFileReference, OutputFileReference and their validators "
            "(_resolve_input_file, _strip_output_file) from "
            "tesseract_core/runtime/experimental/paths.py, and drop them from the "
            "experimental __init__ exports; use InputPath / OutputPath instead. "
            "Update the tests, too."
        ),
    ),
)


def _repo_changelog_path() -> Path:
    """Path to the repository ``CHANGELOG.md``, if running from a source checkout."""
    return Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def _releases_from_changelog(changelog: str) -> list[Release]:
    """Parse ``## [version] - YYYY-MM-DD`` entries, newest first, skipping malformed ones."""
    releases: list[Release] = []
    for raw_version, raw_date in re.findall(
        r"^\#\#\s*\[([^\]]+)\]\s*-\s*(\d{4}-\d{2}-\d{2})",
        changelog,
        re.MULTILINE,
    ):
        try:
            releases.append(Release(Version(raw_version), date.fromisoformat(raw_date)))
        except (InvalidVersion, ValueError):
            continue
    return releases


def latest_releases() -> list[Release]:
    """Return changelog releases, newest first.

    On a release PR the changelog is regenerated with the release being cut at the
    top, so the first entry is what a merge would release.
    """
    changelog_path = _repo_changelog_path()
    if not changelog_path.exists():
        raise FileNotFoundError(
            f"Could not find {changelog_path} to check for overdue deprecations"
        )
    changelog = changelog_path.read_text(encoding="utf-8")
    releases = _releases_from_changelog(changelog)
    if not releases:
        raise ValueError(
            f"Could not find a valid dated version entry in {changelog_path}"
        )
    return releases


def _is_minor_bump(current: Version, previous: Version | None) -> bool:
    """Whether ``current`` bumps the minor (or major) version over ``previous``.

    With no previous release to compare against we can't tell a minor from a
    patch, so we assume a minor bump. Surfacing an overdue removal is safer than
    silently letting a stale shim ship.
    """
    if previous is None:
        return True
    return (current.major, current.minor) > (previous.major, previous.minor)


def overdue_tombstones(releases: list[Release]) -> list[Tombstone]:
    """Return tombstones due for removal in the release being cut (``releases[0]``).

    A tombstone is overdue only when that release postdates its removal date and
    is a minor (or major) bump. Patch releases never force a removal.
    """
    current = releases[0]
    previous = releases[1].version if len(releases) > 1 else None
    if not _is_minor_bump(current.version, previous):
        return []
    return [t for t in TOMBSTONES if current.when > date.fromisoformat(t.remove_after)]
