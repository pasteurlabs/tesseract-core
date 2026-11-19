#!/usr/bin/env python3

"""Decide which jobs the test suite workflow (run_tests.yml) runs.

Pull requests run a fast tier that covers most code changes. The merge queue
runs the full matrix, which adds jobs that are slow, cost money, or mostly
catch platform and environment problems (GPU, demos, macOS, the remaining
Python versions, ARM, Podman, and pip installs).

A pull request also runs the full matrix when it carries the `ci:full` label
or changes dependencies or the CI setup itself. It runs the demo and GPU jobs
when it touches their files. run_tests.yml sets the paths that trigger each.

`plan` prints the job outputs and an explanation to the log, and also writes
them to $GITHUB_OUTPUT and $GITHUB_STEP_SUMMARY when those are set. `check`
fails if any of the jobs that ran failed.
"""

import argparse
import json
import os
import sys
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import NamedTuple

FULL_LABEL = "ci:full"

# Shards for the example end-to-end tests, sized so a shard takes about as
# long as the base end-to-end job.
NUM_EXAMPLE_SHARDS = 6

# Examples that only build on arm64 and run on the ARM runner in the full tier.
ARM_ONLY_EXAMPLES = ["pyvista-arm64"]


class PathTriggers(NamedTuple):
    """Glob patterns for the changed paths that trigger optional jobs."""

    full: list[str]
    demos: list[str]
    gpu: list[str]


class PythonVersions(NamedTuple):
    """The Python versions CI tests with, passed in from the workflow."""

    oldest: str
    latest: str
    # For jobs that need a single version, such as the end-to-end test host.
    default: str


def _base_job(
    os_name: str, python: str, deps: str = "latest", typeguard: bool = False
) -> dict:
    label = f"{os_name}, {python}, {deps}" + (", typeguard" if typeguard else "")
    return {
        "label": label,
        "os": os_name,
        "python-version": python,
        "runtime-deps": deps,
        "typeguard": typeguard,
    }


def _base_jobs(python: PythonVersions) -> tuple[list[dict], list[dict]]:
    """Return the fast and full-only unit test jobs.

    They test the oldest and latest supported Python versions. Typeguard breaks
    coverage's branch mapping (gh#691), so one job runs with typeguard and
    without coverage.
    """
    fast = [
        _base_job("ubuntu-latest", python.oldest, deps="oldest"),
        _base_job("ubuntu-latest", python.latest, typeguard=True),
        _base_job("macos-latest", python.latest),
        _base_job("windows-latest", python.latest),
    ]
    full_only = [
        _base_job("ubuntu-latest", python.oldest),
        _base_job("macos-latest", python.oldest),
        _base_job("windows-latest", python.oldest),
    ]
    return fast, full_only


def _e2e_job(
    name: str,
    python: str,
    arch: str = "x64",
    engine: str = "docker",
    examples: Iterable[str] = (),
) -> dict:
    return {
        "label": f"{name}, {arch}, {engine}",
        "name": name,
        "python-version": python,
        "arch": arch,
        "docker-engine": engine,
        "examples": " ".join(examples),
    }


def _pip_install_jobs(python: PythonVersions) -> list[dict]:
    return [
        {"os": os_name, "python-version": version, "dependencies": "pinned"}
        for os_name in ["ubuntu-latest", "macos-latest"]
        for version in [python.oldest, python.latest]
    ] + [
        {
            "os": "ubuntu-latest",
            "python-version": python.latest,
            "dependencies": "latest",
        }
    ]


def _matches(paths: Iterable[str], patterns: list[str]) -> list[str]:
    return sorted(p for p in paths if any(fnmatchcase(p, pat) for pat in patterns))


def _list_dirs(path: Path) -> list[str]:
    return sorted(
        p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith("_")
    )


def plan(
    event: str,
    labels: list[str],
    changed_files: list[str] | None,
    triggers: PathTriggers,
    python: PythonVersions,
    repo_root: Path,
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
    full_triggers = _matches(changed_files or [], triggers.full)
    if full_triggers:
        reasons.append("changes to " + ", ".join(f"`{p}`" for p in full_triggers))
    full = bool(reasons)

    run_demos = full or bool(_matches(changed_files or [], triggers.demos))
    run_gpu = full or bool(_matches(changed_files or [], triggers.gpu))

    examples = [
        e for e in _list_dirs(repo_root / "examples") if e not in ARM_ONLY_EXAMPLES
    ]
    shards = [examples[i::NUM_EXAMPLE_SHARDS] for i in range(NUM_EXAMPLE_SHARDS)]
    e2e = [_e2e_job("base", python.default)] + [
        _e2e_job(f"examples-{i + 1}", python.default, examples=shard)
        for i, shard in enumerate(shards)
        if shard
    ]
    base, base_full_only = _base_jobs(python)
    if full:
        base += base_full_only
        e2e += [
            # Test on arm to ensure compatibility with Apple M1 chips
            # (OSX runners don't have access to Docker so we use Linux ARM runners instead)
            _e2e_job("base", python.default, arch="arm"),
            _e2e_job(
                "examples-arm",
                python.default,
                arch="arm",
                examples=ARM_ONLY_EXAMPLES,
            ),
            _e2e_job("base", python.default, engine="podman"),
        ]

    outputs = {
        "full": json.dumps(full),
        "run-demos": json.dumps(run_demos),
        "run-gpu": json.dumps(run_gpu),
        "base-matrix": json.dumps(base),
        "e2e-matrix": json.dumps(e2e),
        "pip-install-matrix": json.dumps(_pip_install_jobs(python)),
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


def check(needs: dict[str, dict]) -> list[str]:
    """Return the failed jobs in `needs`, the workflow's `needs` context.

    Jobs outside this event's tier are skipped, which counts as passing. The
    plan job itself must succeed, since every other job is skipped without it.
    """
    return [
        f"{job}: {info['result']}"
        for job, info in needs.items()
        if info["result"] != "success"
        and (job == "plan" or info["result"] != "skipped")
    ]


def _split_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _read_lines(path: Path) -> list[str]:
    return _split_lines(path.read_text())


def _print_and_append(env_var: str, text: str) -> None:
    print(text)
    if env_var in os.environ:
        with open(os.environ[env_var], "a") as f:
            f.write(text)


def _run_plan(args: argparse.Namespace) -> None:
    labels = _read_lines(args.labels) if args.labels else []
    changed_files = (
        _read_lines(args.changed_files)
        if args.changed_files and args.changed_files.exists()
        else None
    )
    triggers = PathTriggers(
        full=_split_lines(args.full_paths),
        demos=_split_lines(args.demo_paths),
        gpu=_split_lines(args.gpu_paths),
    )
    python = PythonVersions(
        oldest=args.python_oldest,
        latest=args.python_latest,
        default=args.python_default,
    )
    outputs, reasons = plan(
        args.event, labels, changed_files, triggers, python, Path(".")
    )
    _print_and_append("GITHUB_STEP_SUMMARY", _summary(outputs, reasons))
    _print_and_append(
        "GITHUB_OUTPUT", "".join(f"{k}={v}\n" for k, v in outputs.items())
    )


def _run_check(args: argparse.Namespace) -> None:
    failed = check(json.loads(args.needs))
    if failed:
        print("Failed jobs:\n" + "\n".join(failed))
        sys.exit(1)
    print("All tests passed!")


def main() -> None:
    """Run the subcommand given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    plan_parser = subparsers.add_parser("plan", help="Plan the jobs for an event")
    plan_parser.set_defaults(func=_run_plan)
    plan_parser.add_argument("--event", required=True, help="GitHub event name")
    plan_parser.add_argument(
        "--python-oldest", required=True, help="Oldest supported Python version"
    )
    plan_parser.add_argument(
        "--python-latest", required=True, help="Latest supported Python version"
    )
    plan_parser.add_argument(
        "--python-default",
        required=True,
        help="Python version for jobs that test a single version",
    )
    for name, jobs in [
        ("full", "the full matrix"),
        ("demo", "the demo jobs"),
        ("gpu", "the GPU jobs"),
    ]:
        plan_parser.add_argument(
            f"--{name}-paths",
            default="",
            help=f"Path globs, one per line, whose changes run {jobs}",
        )
    plan_parser.add_argument(
        "--labels", type=Path, help="File with the pull request's labels, one per line"
    )
    plan_parser.add_argument(
        "--changed-files",
        type=Path,
        help="File with the pull request's changed paths, one per line. "
        "If missing, the changed files are treated as unknown.",
    )

    check_parser = subparsers.add_parser(
        "check", help="Fail if any of the jobs that ran failed"
    )
    check_parser.set_defaults(func=_run_check)
    check_parser.add_argument(
        "--needs", required=True, help="The workflow's `needs` context as JSON"
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
