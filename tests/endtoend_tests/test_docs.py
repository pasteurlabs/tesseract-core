# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the code snippets of the docs pages in DOC_PAGES, top to bottom.

Shell commands (`$ ...` lines in ``bash`` blocks) run in a subprocess, Python
blocks share one namespace per page, and ``>>>`` blocks run as doctests. Where a
page shows output, it must match the real output.

MyST comments, which are not rendered, steer the run:

- ``% skip: next "reason"`` skips the next snippet.
- ``% invisible-code-block: bash`` (or ``python``) followed by ``% ``-prefixed
  lines runs hidden setup code.

Each page gets one shell session. It starts in the repository root, keeps its
working directory across commands, and sets ``$REPO_ROOT`` and ``$DOC_TMPDIR``.
After ``tesseract serve``, the placeholders ``<tesseract-address>`` and ``<port>``
point at the served Tesseract, which is torn down after the page.

In rich tables, IDs, addresses, and container names are masked, and expected
rows only need to appear somewhere in the output, so other running Tesseracts
don't interfere.
"""

import doctest
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from sybil import Sybil
from sybil.parsers.myst import CodeBlockParser, PythonCodeBlockParser, SkipParser

REPO_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "content"

DOC_PAGES = [
    "tutorials/create.md",
    "tutorials/interact.md",
]

_VOLATILE_CELL = re.compile(
    r"(sha256:)?[0-9a-f]{12,}"  # image / container IDs
    r"|tesseract-[0-9a-z]{12}"  # container names
    r"|[0-9.]+:[0-9]+"  # host addresses
)


class ShellSession:
    """State carried between the shell commands of one page."""

    def __init__(self):
        self.cwd = REPO_ROOT
        self.tmpdir = Path(tempfile.mkdtemp(prefix="tesseract-docs-"))
        self.placeholders = {}
        self.served_containers = []

    def run(self, cmd: str) -> subprocess.CompletedProcess:
        for placeholder, value in self.placeholders.items():
            cmd = cmd.replace(placeholder, value)

        cwd_file = self.tmpdir / ".cwd"
        script = f'{cmd}\n__rc=$?; pwd > "{cwd_file}"; exit $__rc'
        env = {
            **os.environ,
            "REPO_ROOT": str(REPO_ROOT),
            "DOC_TMPDIR": str(self.tmpdir),
            # Keep rich from wrapping or truncating tables
            "COLUMNS": "1000",
        }
        res = subprocess.run(
            ["bash", "-c", script],
            cwd=self.cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if cwd_file.exists():
            self.cwd = Path(cwd_file.read_text().strip())

        if res.returncode == 0 and cmd.startswith("tesseract serve"):
            served = json.loads(res.stdout)
            self.served_containers.append(served["container_name"])
            container = served["containers"][0]
            self.placeholders["<tesseract-address>"] = container["ip"]
            self.placeholders["<port>"] = container["port"]

        return res

    def close(self):
        if self.served_containers:
            subprocess.run(
                ["tesseract", "teardown", *self.served_containers],
                capture_output=True,
                check=False,
            )
        shutil.rmtree(self.tmpdir, ignore_errors=True)


def _parse_rich_table(text: str) -> tuple[list[str], set[tuple[str, ...]]]:
    def cells(line, sep):
        return [cell.strip() for cell in line.strip().strip(sep).split(sep)]

    header, rows = [], []
    for line in text.splitlines():
        if line.startswith("┃"):
            header = cells(line, "┃")
        elif line.startswith("│"):
            row = cells(line, "│")
            if rows and not row[0]:
                # Continuation of a cell that rich wrapped onto the next line
                rows[-1] = [
                    f"{a} {b}".strip() for a, b in zip(rows[-1], row, strict=True)
                ]
            else:
                rows.append(row)

    def normalize(cell):
        if _VOLATILE_CELL.fullmatch(cell):
            return "<volatile>"
        # Podman qualifies unregistered image tags as localhost/<name>
        return cell.replace("'localhost/", "'")

    masked = {tuple(normalize(c) for c in row) for row in rows}
    return header, masked


def _compare_output(expected: str, actual: str) -> str | None:
    """Return a description of the mismatch, or None if the output matches."""
    if "┃" in expected:
        expected_header, expected_rows = _parse_rich_table(expected)
        actual_header, actual_rows = _parse_rich_table(actual)
        if expected_header != actual_header:
            return (
                f"Table header mismatch.\nExpected: {expected_header}\n"
                f"Got:      {actual_header}"
            )
        missing = expected_rows - actual_rows
        if not missing:
            return None
        # Rows that share a name with a missing row are the likely culprits
        names = {row[expected_header.index("Name")] for row in missing}
        candidates = [r for r in actual_rows if r[actual_header.index("Name")] in names]
        return (
            f"Table rows missing from output: {sorted(missing)}\n"
            f"Rows with the same name: {sorted(candidates)}"
        )

    try:
        if json.loads(expected) == json.loads(actual):
            return None
    except ValueError:
        if expected.strip() == actual.strip():
            return None
    return f"Output mismatch.\nExpected:\n{expected}\nGot:\n{actual}"


def evaluate_shell(example) -> str | None:
    """Evaluate a ``bash`` block as described in the module docstring."""
    commands = []
    for line in example.parsed.splitlines():
        if line.startswith("$ "):
            commands.append([line[2:], []])
        elif commands and commands[-1][0].endswith("\\") and not commands[-1][1]:
            commands[-1][0] = commands[-1][0][:-1] + line
        elif commands:
            commands[-1][1].append(line)

    session = example.namespace["shell"]
    for cmd, expected_lines in commands:
        res = session.run(cmd)
        if res.returncode != 0:
            return (
                f"`{cmd}` exited with code {res.returncode}.\n"
                f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
            )
        expected = "\n".join(expected_lines)
        if expected.strip():
            mismatch = _compare_output(expected, res.stdout)
            if mismatch:
                return f"`{cmd}`: {mismatch}"
    return None


sybil = Sybil(
    parsers=[
        SkipParser(),
        PythonCodeBlockParser(doctest_optionflags=doctest.ELLIPSIS),
        CodeBlockParser(language="bash", evaluator=evaluate_shell),
    ]
)

documents = {page: sybil.parse(DOCS_DIR / page) for page in DOC_PAGES}
snippets = [
    pytest.param(page, example, id=f"{page}:{example.line}")
    for page, document in documents.items()
    for example in document
]


@pytest.fixture(scope="module", autouse=True)
def shell_sessions():
    sessions = []
    for document in documents.values():
        session = ShellSession()
        document.namespace["shell"] = session
        sessions.append(session)
    yield
    for session in sessions:
        session.close()


# Snippets build on each other, so select whole pages rather than single snippets.
@pytest.mark.parametrize("page,example", snippets)
def test_doc_snippet(page, example):
    # Python snippets see the same working directory as shell snippets
    orig_cwd = Path.cwd()
    os.chdir(example.namespace["shell"].cwd)
    try:
        example.evaluate()
    finally:
        os.chdir(orig_cwd)
