# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforce scheduled deprecation removals against the release being cut.

On a release PR the changelog is regenerated with the new dated entry at the top,
so this test fails there if a deprecation is due for removal. Because merging the
release PR is what triggers the release, the stale shim is caught before it
ships. On ordinary PRs the top changelog entry is the last release, so the test
only fires once a release actually crosses a tombstone's target date. Patch
releases never trip the check, so a hotfix is never forced to drop a shim.
"""

from datetime import date

import pytest
from packaging.version import Version

from tesseract_core import _deprecations
from tesseract_core._deprecations import (
    Release,
    Tombstone,
    _releases_from_changelog,
    _repo_changelog_path,
    latest_releases,
    overdue_tombstones,
)


def test_no_overdue_deprecations():
    changelog_path = _repo_changelog_path()
    if not changelog_path.exists():
        raise FileNotFoundError(
            f"Could not find {changelog_path} to check for overdue deprecations"
        )

    releases = latest_releases()
    overdue = overdue_tombstones(releases)
    assert len(overdue) == 0, "Deprecations due for removal in {}:\n{}".format(
        releases[0].version,
        "\n".join(
            f"  - {t.what} (remove_after {t.remove_after}): {t.hint}" for t in overdue
        ),
    )


def test_releases_from_changelog_parses_entries():
    changelog = "# Changelog\n\n## [1.13.0] - 2026-09-01\n\n## [1.12.0] - 2026-08-01\n"
    releases = _releases_from_changelog(changelog)
    assert releases == [
        Release(Version("1.13.0"), date(2026, 9, 1)),
        Release(Version("1.12.0"), date(2026, 8, 1)),
    ]
    assert _releases_from_changelog("# Changelog\n\nnothing here") == []


def _releases(*entries: tuple[str, str]) -> list[Release]:
    return [Release(Version(v), date.fromisoformat(d)) for v, d in entries]


@pytest.fixture
def one_tombstone(monkeypatch):
    """Replace the registry with a single tombstone dated 2026-12-01."""
    monkeypatch.setattr(
        _deprecations,
        "TOMBSTONES",
        (Tombstone(remove_after=date(2026, 12, 1), what="thing", hint="delete it"),),
    )


def _overdue(*entries: tuple[str, str]) -> int:
    return len(overdue_tombstones(_releases(*entries)))


def test_overdue_fires_after_target_date_on_minor_release(one_tombstone):
    """A tombstone is due only once a minor release postdates its target."""
    # Minor release, but date not yet past the target.
    assert _overdue(("1.13.0", "2026-11-30")) == 0
    # Not overdue on the target date itself; only afterwards.
    assert _overdue(("1.13.0", "2026-12-01")) == 0
    assert _overdue(("1.13.0", "2026-12-02")) == 1


def test_patch_release_never_fires(one_tombstone):
    """A patch release past the target date must not force a removal."""
    # 1.12.1 postdates the target, but a patch release doesn't count.
    assert _overdue(("1.12.1", "2026-12-15")) == 0
    # A post-release of a minor doesn't count either.
    assert _overdue(("1.13.0.post1", "2026-12-15")) == 0
    # The following minor release does trip it.
    assert _overdue(("1.13.0", "2027-01-01")) == 1


def test_major_release_fires(one_tombstone):
    """A major release (x.0.0) past the target date is overdue."""
    assert _overdue(("2.0.0", "2026-12-15")) == 1
