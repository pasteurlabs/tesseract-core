# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforce scheduled deprecation removals against the version being released.

On a release PR the changelog is regenerated with the new version at the top, so
this test fails there if a deprecation is due for removal. Because merging the
release PR is what triggers the release, the stale shim is caught before it
ships. On ordinary PRs the top changelog version is the last release, so the test
only fires once a release actually crosses a tombstone's target.
"""

from packaging.version import Version

from tesseract_core import _deprecations
from tesseract_core._deprecations import (
    Tombstone,
    _repo_changelog_path,
    _version_from_changelog,
    latest_changelog_version,
    overdue_tombstones,
)


def test_no_overdue_deprecations():
    changelog_path = _repo_changelog_path()
    if not changelog_path.exists():
        raise FileNotFoundError(
            f"Could not find {changelog_path} to check for overdue deprecations"
        )

    version = latest_changelog_version()
    overdue = overdue_tombstones(version)
    assert len(overdue) == 0, "Deprecations due for removal by version {}:\n{}".format(
        version,
        "\n".join(f"  - {t.what} (remove_at {t.remove_at}): {t.hint}" for t in overdue),
    )


def test_latest_changelog_version_parses_top_entry():
    changelog = "# Changelog\n\n## [1.13.0] - 2026-09-01\n\n## [1.12.0] - 2026-08-01\n"
    assert _version_from_changelog(changelog) == Version("1.13.0")
    assert _version_from_changelog("# Changelog\n\nnothing here") is None


def test_overdue_fires_at_target_but_not_on_prerelease(monkeypatch):
    """A tombstone is due at its target version, but not on a dev/rc of it."""
    monkeypatch.setattr(
        _deprecations,
        "TOMBSTONES",
        (Tombstone(remove_at="1.13.0", what="thing", hint="delete it"),),
    )
    assert overdue_tombstones(Version("1.12.0")) == []
    # dev/rc sort before the final release, so the release PR (which shows the
    # final version) is what trips the check, not an earlier dev build.
    assert overdue_tombstones(Version("1.13.0.dev5")) == []
    assert len(overdue_tombstones(Version("1.13.0"))) == 1
    assert len(overdue_tombstones(Version("1.14.0"))) == 1
