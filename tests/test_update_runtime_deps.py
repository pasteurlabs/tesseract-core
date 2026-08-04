# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

# NOTE: This tests .github/workflows/update_runtime_deps.py, which is not part of the
# package. It runs unattended every Monday and rewrites pyproject.toml in place, so its
# failure mode is a silently mangled manifest rather than a crash -- worth locking down.
# Being outside the package, it is not importable by name, so load it by path here rather
# than putting .github/workflows on the import path of the whole test suite.
SCRIPT = Path(__file__).parents[1] / ".github" / "workflows" / "update_runtime_deps.py"

spec = importlib.util.spec_from_file_location("update_runtime_deps", SCRIPT)
update_runtime_deps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update_runtime_deps)

get_updated_bounds = update_runtime_deps.get_updated_bounds
write_new_pyproject = update_runtime_deps.write_new_pyproject

# Exercises everything the rewrite has to leave alone: trailing comments, standalone
# comments, blank lines, a comment with no dep after it, single quotes, wide spacing,
# and a dep that is not part of the resolved set.
PYPROJECT = """\
[project]
name = "whatever"
dependencies = ["untouched<=1.0"]

[project.optional-dependencies]
runtime = [
    "pydantic<=2.13.4,>=2.10",
    # section header for the web bits
    "fastapi<=0.140.0,>=0.115",

    "typer<=0.27.0,>=0.26.8",  # sphinxcontrib-typer needs >=0.26.8
    's3fs<=2026.6.0,>=2024.12',
    "lz4<=4.4.5,>=4.0.0",   # not in the resolved set
    # dangling note before the bracket
]
# END RUNTIME DEPENDENCIES

docs = ["sphinx"]
"""

BUMPED_DEPS = [
    "pydantic<=2.99.0,>=2.10",
    "fastapi<=0.141.0,>=0.115",
    "typer<=0.28.0,>=0.26.8",
    "s3fs<=2026.7.0,>=2024.12",
]


@pytest.fixture
def pyproject(tmp_path):
    """A pyproject.toml whose runtime block covers every formatting case."""
    path = tmp_path / "pyproject.toml"
    path.write_text(PYPROJECT)
    return path


def test_bumps_upper_bounds_in_place(pyproject):
    """Resolved deps get their new upper bound, keeping their lower bound."""
    write_new_pyproject(BUMPED_DEPS, str(pyproject))
    result = pyproject.read_text()

    assert '"pydantic<=2.99.0,>=2.10",' in result
    assert '"fastapi<=0.141.0,>=0.115",' in result
    assert "2.13.4" not in result
    assert "0.140.0" not in result


def test_preserves_everything_but_the_requirement(pyproject):
    """Only the quoted requirements change; all other bytes stay put."""
    write_new_pyproject(BUMPED_DEPS, str(pyproject))
    old = PYPROJECT.splitlines()
    new = pyproject.read_text().splitlines()

    assert len(old) == len(new), "no lines were added or dropped"

    changed = [(o, n) for o, n in zip(old, new, strict=True) if o != n]
    assert len(changed) == len(BUMPED_DEPS), f"unexpected edits: {changed}"


@pytest.mark.parametrize(
    "line",
    [
        # Comments must survive verbatim, including their original spacing
        "    # section header for the web bits",
        "    # dangling note before the bracket",
        '    "typer<=0.28.0,>=0.26.8",  # sphinxcontrib-typer needs >=0.26.8',
        # Deps absent from the resolved set are left exactly as they were
        '    "lz4<=4.4.5,>=4.0.0",   # not in the resolved set',
    ],
)
def test_preserves_non_dependency_lines(pyproject, line):
    write_new_pyproject(BUMPED_DEPS, str(pyproject))
    assert line in pyproject.read_text().splitlines()


def test_preserves_blank_line_inside_block(pyproject):
    """A blank line separating groups of deps is not collapsed.

    Asserted by position rather than membership, since the file has blank lines
    outside the runtime block too.
    """
    write_new_pyproject(BUMPED_DEPS, str(pyproject))
    lines = pyproject.read_text().splitlines()

    fastapi = lines.index('    "fastapi<=0.141.0,>=0.115",')
    assert lines[fastapi + 1] == ""


def test_preserves_single_quoted_deps(pyproject):
    """Single-quoted deps are bumped without being rewritten to double quotes."""
    write_new_pyproject(BUMPED_DEPS, str(pyproject))
    assert "    's3fs<=2026.7.0,>=2024.12'," in pyproject.read_text().splitlines()


def test_leaves_rest_of_file_alone(pyproject):
    """Deps outside the runtime block are never touched."""
    write_new_pyproject(BUMPED_DEPS, str(pyproject))
    result = pyproject.read_text()

    assert 'dependencies = ["untouched<=1.0"]' in result
    assert 'docs = ["sphinx"]' in result


def test_raises_if_runtime_block_is_missing(tmp_path):
    """A pyproject without a runtime block is a hard error, not a silent no-op."""
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nname = "whatever"\n')

    with pytest.raises(ValueError, match="runtime"):
        write_new_pyproject(BUMPED_DEPS, str(path))


def test_get_updated_bounds_keeps_lower_bounds(pyproject):
    """Only the '<=' specifier is replaced; other operators are carried over."""
    resolved_env = """\
pydantic==2.99.0
fastapi==0.141.0
typer==0.28.0
s3fs==2026.7.0
lz4==4.9.9
"""
    updated = get_updated_bounds(str(pyproject), resolved_env)

    assert "pydantic<=2.99.0,>=2.10" in updated
    assert "typer<=0.28.0,>=0.26.8" in updated
