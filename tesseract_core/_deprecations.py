# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry of scheduled deprecation removals ("tombstones").

Deprecations rot: a warning ships, the shim stays forever because nothing forces
anyone to revisit it. A tombstone pins a deprecation to the version it must be
gone by, and :func:`overdue_tombstones` reports any that are due. The
accompanying test checks this against the version being cut, so a release PR
that would ship a stale shim fails before it merges (merging a release PR is what
triggers the actual release).

To schedule a removal:

1. Add a :class:`Tombstone` to :data:`TOMBSTONES`, targeting a concrete future
   version.
2. When the release-PR test starts failing, delete the deprecated code *and* its
   tombstone in the same PR.

Target a version the release automation can actually reach. Automated bumps are
capped at ``minor`` (see ``.github/workflows/get_bump_type.sh``), so a new major
is never cut automatically; a ``2.0.0`` target would never fire. Use the next
minor or a later minor instead.
"""

import re
from pathlib import Path
from typing import NamedTuple

from packaging.version import Version


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
TOMBSTONES: tuple[Tombstone, ...] = ()


def _repo_changelog_path() -> Path:
    """Path to the repository ``CHANGELOG.md``, if running from a source checkout."""
    return Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def latest_changelog_version(changelog: str) -> Version | None:
    """Return the topmost ``## [version]`` entry in a changelog, or None if absent.

    On a release PR the changelog is regenerated with the version being cut at the
    top, so this is the version a merge would release.
    """
    match = re.search(r"^##\s*\[([^\]]+)\]", changelog, re.MULTILINE)
    if match is None:
        return None
    try:
        return Version(match.group(1))
    except ValueError:
        return None


def overdue_tombstones(version: Version) -> list[Tombstone]:
    """Return tombstones whose removal is due at or before ``version``."""
    return [t for t in TOMBSTONES if version >= Version(t.remove_at)]
