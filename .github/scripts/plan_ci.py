#!/usr/bin/env python3

"""Decide which jobs the test suite workflow (run_tests.yml) runs.

Pull requests run a fast tier that covers most code changes. The merge queue
runs the full matrix, which adds jobs that are slow, cost money, or mostly
catch platform and environment problems (GPU, demos, macOS, the remaining
Python versions, ARM, Podman, and pip installs).

A pull request also runs the full matrix when it carries the `ci:full` label
or changes dependencies or the CI setup itself. It runs the demo and GPU jobs
when it touches their files.

Writes job outputs to $GITHUB_OUTPUT and an explanation to $GITHUB_STEP_SUMMARY,
or prints both when those are unset.
"""

import argparse
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

FULL_LABEL = "ci:full"

# Changes to any of these run the full matrix on the pull request.
FULL_PATTERNS = [
    r"^pyproject\.toml$",
    r"^production\.uv\.lock$",
    r"^requirements\.txt$",
    r"^\.github/workflows/run_tests\.yml$",
    r"^\.github/scripts/plan_ci\.py$",
]

DEMO_PATTERNS = [r"^demo/"]

# The GPU-specific code paths. Broader changes that could still break GPU
# serving are caught by the merge queue.
GPU_PATTERNS = [
    r"^tesseract_core/runtime/cuda/",
    r"^tesseract_core/runtime/device_transport\.py$",
    r"^tesseract_core/runtime/array_encoding\.py$",
    r"^tests/test_cuda_ipc",
    r"^tests/endtoend_tests/test_serving_gpu\.py$",
    r"^examples/_gpu_",
]

# Shards for the example end-to-end tests, sized so a shard takes about as
# long as the base end-to-end job.
NUM_EXAMPLE_SHARDS = 6

# Examples that only build on arm64 and run on the ARM runner in the full tier.
ARM_ONLY_EXAMPLES = ["pyvista-arm64"]


def _base_job(
    os_name: str, python: str, deps: str = "latest", typeguard: bool = False
) -> dict:
    return {
        "os": os_name,
        "python-version": python,
        "runtime-deps": deps,
        "typeguard": typeguard,
    }


# Test with the oldest and latest supported Python versions.
# NOTE: If bumping the minimum Python version here, also update
# `requires-python` in pyproject.toml and the other workflows.
#
# Typeguard breaks coverage's branch mapping (gh#691), so one job runs with
# typeguard and without coverage.
BASE_FAST = [
    _base_job("ubuntu-latest", "3.10", deps="oldest"),
    _base_job("ubuntu-latest", "3.15", typeguard=True),
    _base_job("windows-latest", "3.15"),
]
BASE_FULL_ONLY = [
    _base_job("ubuntu-latest", "3.10"),
    _base_job("macos-latest", "3.10"),
    _base_job("macos-latest", "3.15"),
    _base_job("windows-latest", "3.10"),
]


def _e2e_job(
    name: str, arch: str = "x64", engine: str = "docker", examples: Iterable[str] = ()
) -> dict:
    return {
        "name": name,
        "arch": arch,
        "docker-engine": engine,
        "examples": " ".join(examples),
    }


def _matches(paths: Iterable[str], patterns: list[str]) -> list[str]:
    return sorted(p for p in paths if any(re.search(pat, p) for pat in patterns))


def _list_dirs(path: Path) -> list[str]:
    return sorted(
        p.name
        for p in Path(path).iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )


def plan(
    event: str, labels: list[str], changed_files: list[str] | None, repo_root: Path
) -> tuple[dict[str, str], list[str]]:
    """Return (outputs, reasons) for the given event.

    `changed_files` is None when the list of changed files is unknown, which
    runs the full matrix.
    """
    reasons = []
    if event != "pull_request":
        reasons.append(f"`{event}` event")
    if FULL_LABEL in labels:
        reasons.append(f"`{FULL_LABEL}` label")
    if changed_files is None and event == "pull_request":
        reasons.append("could not list the changed files")
    full_triggers = _matches(changed_files or [], FULL_PATTERNS)
    if full_triggers:
        reasons.append("changes to " + ", ".join(f"`{p}`" for p in full_triggers))
    full = bool(reasons)

    run_demos = full or bool(_matches(changed_files or [], DEMO_PATTERNS))
    run_gpu = full or bool(_matches(changed_files or [], GPU_PATTERNS))

    examples = [
        e for e in _list_dirs(repo_root / "examples") if e not in ARM_ONLY_EXAMPLES
    ]
    shards = [examples[i::NUM_EXAMPLE_SHARDS] for i in range(NUM_EXAMPLE_SHARDS)]
    e2e = [_e2e_job("base")] + [
        _e2e_job(f"examples-{i + 1}", examples=shard)
        for i, shard in enumerate(shards)
        if shard
    ]
    base = list(BASE_FAST)
    if full:
        base += BASE_FULL_ONLY
        e2e += [
            # Test on arm to ensure compatibility with Apple M1 chips
            # (OSX runners don't have access to Docker so we use Linux ARM runners instead)
            _e2e_job("base", arch="arm"),
            _e2e_job("examples-arm", arch="arm", examples=ARM_ONLY_EXAMPLES),
            _e2e_job("base", engine="podman"),
        ]

    outputs = {
        "full": json.dumps(full),
        "run-demos": json.dumps(run_demos),
        "run-gpu": json.dumps(run_gpu),
        "base-matrix": json.dumps(base),
        "e2e-matrix": json.dumps(e2e),
        "demos": json.dumps(_list_dirs(repo_root / "demo")),
    }
    return outputs, reasons


def _summary(outputs: dict[str, str], reasons: list[str]) -> str:
    if reasons:
        headline = "Running the **full** test matrix because of: " + "; ".join(reasons)
    else:
        headline = (
            "Running the **fast** test matrix. The merge queue runs the full one. "
            f"To run it here, add the `{FULL_LABEL}` label and re-run this workflow."
        )
    lines = [headline]
    lines.append(f"- Demos: {'yes' if outputs['run-demos'] == 'true' else 'no'}")
    lines.append(f"- GPU: {'yes' if outputs['run-gpu'] == 'true' else 'no'}")
    return "\n".join(lines) + "\n"


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    """Plan the jobs for the current event and write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, help="GitHub event name")
    parser.add_argument(
        "--labels", type=Path, help="File with the pull request's labels, one per line"
    )
    parser.add_argument(
        "--changed-files",
        type=Path,
        help="File with the pull request's changed paths, one per line. "
        "If missing, the changed files are treated as unknown.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    labels = _read_lines(args.labels) if args.labels else []
    changed_files = (
        _read_lines(args.changed_files)
        if args.changed_files and args.changed_files.exists()
        else None
    )
    outputs, reasons = plan(args.event, labels, changed_files, args.repo_root)

    output_text = "".join(f"{key}={value}\n" for key, value in outputs.items())
    summary_text = _summary(outputs, reasons)
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(output_text)
    else:
        print(output_text)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary_text)
    else:
        print(summary_text)


if __name__ == "__main__":
    main()
