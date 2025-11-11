#!/usr/bin/env python3

"""Update upper bounds of runtime dependencies in pyproject.toml using UV."""

import subprocess
import sys
import tempfile

import toml
from packaging.requirements import Requirement
from packaging.specifiers import Specifier, SpecifierSet


def write_unbounded_pyproject(pyproject_file: str, workdir: str) -> None:
    """Write pyproject.toml without upper dependency bounds to a temp directory."""
    pyproject = toml.load(pyproject_file)
    old_deps = [Requirement(dep) for dep in pyproject["project"]["dependencies"]]

    new_deps = []
    for req in old_deps:
        req.specifier = SpecifierSet(
            [spec for spec in req.specifier if spec.operator != "<="]
        )
        new_deps.append(str(req))
    pyproject["project"]["dependencies"] = new_deps

    # Write modified pyproject.toml to temp directory
    temp_pyproject_path = f"{workdir}/pyproject.toml"
    with open(temp_pyproject_path, "w") as f:
        toml.dump(pyproject, f)


def update_requirements(tmpdir: str) -> str:
    """Use UV to resolve and bump all dependencies of a pyproject.toml in tmpdir.

    Returns the resolved environment as a requirements.txt string.
    """
    res = subprocess.run(
        ["uv", "export", "--no-hashes", "--refresh"],
        cwd=tmpdir,
        capture_output=True,
        text=True,
    )

    if res.returncode != 0:
        print("Error running UV to resolve dependencies:", res.stderr)
        sys.exit(1)

    resolved_env = res.stdout
    return resolved_env


def get_updated_bounds(pyproject_file: str, resolved_env: str) -> list[str]:
    """Parse resolved environment and update upper bounds from original pyproject.toml."""
    pyproject = toml.load(pyproject_file)
    current_deps = [Requirement(dep) for dep in pyproject["project"]["dependencies"]]

    new_upper_bounds = {}
    for line in resolved_env.splitlines():
        if "==" in line:
            req = Requirement(line)
            pkg = req.name
            ver = next(iter(req.specifier)).version

            if pkg in new_upper_bounds:
                if ver > new_upper_bounds[pkg]:
                    new_upper_bounds[pkg] = ver
            else:
                new_upper_bounds[pkg] = ver

    final_deps = []
    for dep in current_deps:
        pkg_name = dep.name
        if pkg_name in new_upper_bounds:
            upper_bound = new_upper_bounds[pkg_name]
            dep.specifier = SpecifierSet(
                [
                    *[spec for spec in dep.specifier if spec.operator != "<="],
                    Specifier(f"<= {upper_bound}"),
                ]
            )
        final_deps.append(str(dep))

    return final_deps


def write_new_pyproject(final_deps: list[str], pyproject_file: str) -> None:
    """Write updated dependencies with new upper bounds back to pyproject.toml.

    Preserves formatting of other parts of the file.
    """
    with open(pyproject_file) as f:
        original_lines = list(f.readlines())

    dep_start_line = None
    for idx, line in enumerate(original_lines):
        if line.strip() == "dependencies = [":
            dep_start_line = idx + 1
        if dep_start_line and line.strip() == "]":
            dep_end_line = idx
            break

    new_lines = [
        *original_lines[:dep_start_line],
        *[f'    "{dep}",\n' for dep in final_deps],
        *original_lines[dep_end_line:],
    ]

    with open(pyproject_file, "w") as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: update_runtime_deps.py <path_to_pyproject.toml>")
        sys.exit(1)

    pyproject_file = sys.argv[1]

    with tempfile.TemporaryDirectory() as tmpdir:
        write_unbounded_pyproject(pyproject_file, tmpdir)
        resolved_env = update_requirements(tmpdir)
        final_deps = get_updated_bounds(pyproject_file, resolved_env)
        write_new_pyproject(final_deps, pyproject_file)
