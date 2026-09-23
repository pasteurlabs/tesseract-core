# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the Python environment a Tesseract needs, without a container.

Every Tesseract says what it needs in ``tesseract_config.yaml``, under
``build_config.requirements``. That names a provider (``uv-pip`` or ``conda``)
and therefore a requirements file (``tesseract_requirements.txt`` or
``tesseract_environment.yaml``).

``tesseract build`` turns that into an environment inside an image, using
:mod:`~tesseract_core.sdk.engine` and the ``build_*_venv.sh`` templates. This
module does the same job on the host, so that
:func:`~tesseract_core.sdk.local_client.serve` has an interpreter to run and the
caller does not have to set one up by hand.

Only :func:`resolve_python_executable` is used outside this module. All of it
runs before the Tesseract's process starts, so none of it affects request
latency, and an environment is built at most once.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    metadata,
)
from importlib.metadata import (
    version as installed_version,
)
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from .api_parse import TesseractBuildConfig, ValidationError, get_config
from .config import get_config as get_sdk_config
from .engine import _split_local_dependency
from .engine import parse_requirements as _parse_requirements
from .exceptions import UserError

logger = logging.getLogger("tesseract")


@functools.cache
def _requirements_of(path: Path, _mtime_ns: int) -> tuple[list[str], list[str]]:
    """Cache of :func:`engine.parse_requirements`, keyed on file and mtime.

    That function imports pip's internals and builds a new ``PipSession`` every
    time it runs, which is slow. We ask it the same question up to three times
    while resolving one Tesseract. The mtime is part of the key so that editing
    the file still gets it re-read.
    """
    local, remote = _parse_requirements(path)
    return local, remote


def parse_requirements(path: Path) -> tuple[list[str], list[str]]:
    """Local and remote requirements declared by a file."""
    return _requirements_of(path, path.stat().st_mtime_ns)


# These variables tie a process to one interpreter's packages. If a different
# interpreter inherits them, it looks in our site-packages first, which is the
# opposite of what a separate environment is for.
_SCRUBBED_IMPORT_VARS = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")


_MANAGED_VENV_NAME = ".venv"

# Looked for next to the api file before we create anything. These are the
# directory names GitHub's Python .gitignore calls "Environments", minus two:
# `ENV` is the same directory as `env` on macOS and Windows and is rare
# elsewhere, and `.env` is nearly always a dotenv file rather than a directory.
#
# An environment the user has *activated* needs no entry here, whether it is a
# venv or a conda prefix, because it is already `sys.executable`.
_VENV_CANDIDATES = (_MANAGED_VENV_NAME, "venv", "env")

# Records which environment file a conda environment was built from. conda has
# no quick way to check whether an environment is already up to date, and
# solving one takes minutes, so we compare a hash instead.
_CONDA_STAMP_NAME = ".tesseract-conda-stamp.json"


def _python_in(prefix: Path) -> Path:
    """Path to the interpreter inside an environment directory."""
    if os.name == "nt":
        return prefix / "Scripts" / "python.exe"
    return prefix / "bin" / "python"


def _dist_versions(python_executable: Path) -> dict[str, str]:
    """Read the version of everything installed in an environment.

    Taken from the ``.dist-info`` directory names, which is one directory scan
    (about a millisecond). The obvious alternatives are both much slower: asking
    uv costs a subprocess, and in a development checkout ``uv pip install``
    rebuilds the local wheel even when there is nothing to do. Running the
    interpreter to find its own site-packages costs tens of milliseconds.

    This runs before we know whether there is any work to do at all, so it needs
    to be cheap. Returns every distribution at once because callers ask about
    more than one.
    """
    prefix = python_executable.parent.parent
    if os.name == "nt":
        directories = [prefix / "Lib" / "site-packages"]
    else:
        directories = sorted((prefix / "lib").glob("python*/site-packages"))

    versions: dict[str, str] = {}
    for directory in directories:
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            name, separator, tail = entry.name.partition("-")
            if not separator or not tail.endswith(".dist-info"):
                continue
            version = tail[: -len(".dist-info")]
            # Versions never contain a hyphen. If this one does, we split in
            # the wrong place: the distribution's own name has a hyphen in it.
            if "-" not in version:
                versions.setdefault(name, version)
    return versions


def _can_serve(python_executable: Path, version: str | None = None) -> bool:
    """Whether an environment can serve a Tesseract, optionally at `version`.

    ``runtime`` is an optional extra, so having the SDK installed is not enough.
    This is why ``pip install tesseract-core`` on its own cannot serve anything.
    We test for uvicorn to tell the two apart: serving needs it, and the base
    dependencies do not include it.
    """
    installed = _dist_versions(python_executable)
    core = installed.get("tesseract_core")
    if core is None or (version is not None and core != version):
        return False
    return "uvicorn" in installed


@functools.cache
def _runtime_install_spec() -> str:
    """The spec to install so an environment has ``tesseract-core[runtime]``.

    For a released install this pins our own version, so the Tesseract runs
    against the same runtime as the SDK that started it. A development checkout
    installs from the source tree, since its version is not on any index.
    """
    try:
        dist = distribution("tesseract-core")
    except PackageNotFoundError:  # pragma: no cover - the SDK is what is running
        return "tesseract-core[runtime]"

    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        url = json.loads(direct_url).get("url", "")
        if url.startswith("file://"):
            local_path = Path(_split_local_dependency(url)[0])
            if (local_path / "pyproject.toml").is_file():
                return f"{local_path}[runtime]"

    return f"tesseract-core[runtime]=={dist.version}"


def _declared_requirements(api_path: Path) -> tuple[Any, Path] | None:
    """Read what a Tesseract says it needs installed.

    Returns its build config and the requirements file its provider names, or
    None if there is nothing to install. None covers three cases: no
    ``tesseract_config.yaml`` next to the api file, no requirements file, or a
    requirements file containing only comments.
    """
    src_dir = api_path.parent

    if not (src_dir / "tesseract_config.yaml").is_file():
        # No config is not an error: `tesseract_api.py` is all the runtime
        # needs. But a `tesseract_requirements.txt` sitting next to it still
        # says what to install, and the defaults are exactly the provider and
        # filename a build would assume, so use those.
        default = TesseractBuildConfig()
        requirements_file = src_dir / default.requirements._filename
        return (default, requirements_file) if requirements_file.is_file() else None

    try:
        build_config = get_config(src_dir).build_config
    except (ValidationError, yaml.YAMLError) as e:
        # `tesseract build` would reject this file, so say so now. Serving from
        # source is usually the step before building, and quietly carrying on
        # would hide a problem the user is going to hit anyway. Only these two
        # errors mean "bad file"; any other exception here is our bug and must
        # not be turned into a user-facing message.
        raise UserError(
            f"Could not read {src_dir / 'tesseract_config.yaml'}, so there is no "
            f"way to tell what this Tesseract needs installed: {e}"
        ) from e

    requirements = build_config.requirements
    requirements_file = src_dir / requirements._filename
    if not requirements_file.is_file():
        return None

    # A crude check on purpose. `parse_requirements` would be exact, but it
    # imports pip's internals (~150ms), which is a lot to pay just to find out
    # a Tesseract needs nothing. Option lines count as content so that this
    # agrees with `parse_requirements` about files like `-r other.txt`.
    declares_something = any(
        line.strip() and not line.strip().startswith("#")
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
    )
    if not declares_something:
        return None

    return build_config, requirements_file


def _uv() -> tuple[str, ...]:
    """How to invoke uv, or raise explaining what to do without it."""
    configured = get_sdk_config().uv_executable
    if configured is not None:
        return configured

    found = shutil.which("uv")
    if found is None:
        raise RuntimeError(
            "uv is needed to build an environment for this Tesseract and was "
            "not found on PATH. Install it (https://docs.astral.sh/uv/), point "
            "TESSERACT_UV_EXECUTABLE at it, or pass `python_executable` to name "
            "an interpreter that already has the Tesseract's dependencies and "
            "`tesseract-core[runtime]`."
        )
    return (found,)


def _conda() -> tuple[str, ...]:
    """How to invoke conda, or raise explaining what to do without it.

    CONDA_EXE comes first because conda's own shell setup exports it. If there
    are several installations, that is the one the user actually works in.
    """
    configured = get_sdk_config().conda_executable
    if configured is not None:
        return configured

    from_env = os.environ.get("CONDA_EXE")
    if from_env and Path(from_env).is_file():
        return (from_env,)

    for candidate in ("conda", "mamba", "micromamba"):
        found = shutil.which(candidate)
        if found is not None:
            return (found,)

    raise RuntimeError(
        "This Tesseract declares `requirements.provider: conda`, but no conda "
        "was found: CONDA_EXE is unset and none of conda, mamba or micromamba "
        "is on PATH. Install one, point TESSERACT_CONDA_EXECUTABLE at it, or "
        "pass `python_executable` to name an interpreter that already has the "
        "Tesseract's dependencies and `tesseract-core[runtime]`."
    )


def _capture(
    command: Sequence[Any], *, stdin: str | None = None, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    """Run a command, log it, and return the result for the caller to check."""
    argv = [str(part) for part in command]
    logger.debug("Running %s", " ".join(argv))
    # Importing a `tesseract_api.py` in this process sets PYTHONPATH as a side
    # effect. An installer that inherited it would look at our packages instead
    # of the ones in the environment it is building, so drop those variables.
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED_IMPORT_VARS}
    return subprocess.run(
        argv,
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        cwd=None if cwd is None else str(cwd),
    )


def _run(command: Sequence[Any], what: str, *, cwd: Path | None = None) -> None:
    """Run a command, raising with its output if it fails."""
    result = _capture(command, cwd=cwd)
    if result.returncode != 0:
        output = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"{what} failed:\n{output[-2000:]}")


def _install_from(requirements_file: Path) -> tuple[list[str], Path]:
    """Arguments and working directory to install a requirements file.

    `uv pip install -r` works out the format for itself, so a flat
    `tesseract_requirements.txt` and a PEP 751 lockfile are installed the same
    way. This is the same command `build_pip_venv.sh` runs in a container.

    Run from the file's own directory, because relative local paths inside it
    are written relative to the file. A build gets that for free by copying
    everything to one place first.
    """
    return ["-r", requirements_file.name], requirements_file.parent


def _venv_satisfies(python_executable: Path, requirements_file: Path) -> bool:
    """Whether an environment already satisfies a pip requirements file.

    We ask uv, so that markers, pins, extras and index options all mean what
    they mean to a build. When the answer is yes uv exits almost immediately,
    which matters because this runs every time a Tesseract is served.
    """
    args, cwd = _install_from(requirements_file)
    result = _capture(
        [*_uv(), "pip", "install", "--dry-run", "--python", python_executable, *args],
        cwd=cwd,
    )
    # uv prints this on stderr, along with the rest of its progress output.
    return result.returncode == 0 and "Would make no changes" in (
        result.stderr + result.stdout
    )


def _caller_shortfall(
    requirements_file: Path, is_pylock: bool = False
) -> tuple[list[str], list[str]] | None:
    """What this interpreter is missing for a requirements file.

    Returns two lists: packages that are not installed, and packages installed
    at a version the file disallows. Returns None if it cannot tell.

    Either one is reason enough to build an environment. They are reported
    separately so the log can say which it was.

    This uses ``importlib.metadata`` instead of uv because uv refuses to look at
    an externally managed interpreter at all (PEP 668), so it cannot answer for
    a system or Homebrew Python.

    When in doubt it returns None, so the caller builds an environment instead
    of guessing. That covers local paths, direct URLs, environment markers and
    option lines.

    A PEP 751 lockfile always returns None. It is TOML, so the requirements
    parser cannot read it, and a lockfile asks for one exact set of versions,
    which is a request for its own environment rather than for whatever happens
    to be installed here.
    """
    if is_pylock:
        return None

    from importlib.metadata import PackageNotFoundError

    local, remote = parse_requirements(requirements_file)
    if local:
        return None

    missing: list[str] = []
    mismatched: list[str] = []
    for spec in remote:
        if spec.startswith("-"):
            return None
        try:
            requirement = Requirement(spec)
        except InvalidRequirement:
            return None
        if requirement.marker is not None or requirement.url is not None:
            return None
        try:
            have = installed_version(requirement.name)
        except PackageNotFoundError:
            missing.append(requirement.name)
            continue
        if not requirement.specifier.contains(have, prereleases=True):
            mismatched.append(f"{requirement.name} {have} (wanted {requirement})")

    return missing, mismatched


@functools.cache
def _python_bounds() -> tuple[int, int] | None:
    """Oldest and newest Python minor version we may build an environment on.

    Bounds only. We always start from this interpreter's version and let uv tell
    us which way to go from there, so there is no list of preferences to keep
    (see :func:`_build_python_version`).

    The ceiling is whatever uv says it can install here, so new releases and
    prereleases are picked up without this module knowing about them. The floor
    is the SDK's own ``Requires-Python``. That also excludes end-of-life
    versions: uv offers 3.8 and 3.9, and the runtime will not install on them.
    """
    result = _capture([*_uv(), "python", "list", "--output-format", "json"])
    if result.returncode != 0:
        return None

    supported = SpecifierSet(metadata("tesseract-core")["Requires-Python"] or "")

    minors = set()
    for entry in json.loads(result.stdout):
        if entry.get("implementation") != "cpython":
            continue
        parts = entry.get("version_parts") or {}
        major, minor = parts.get("major"), parts.get("minor")
        if major != sys.version_info.major or minor is None:
            continue
        # Test "X.Y.0", not "X.Y". PEP 440 padding means a bare "3.10" does
        # not always satisfy ">=3.10".
        if supported and f"{major}.{minor}.0" not in supported:
            continue
        minors.add(minor)

    if not minors:
        return None
    return min(minors), max(minors)


def _nearest(minors: Iterable[int]) -> str | None:
    """Pick the best version to build on out of those offered.

    Closest to ours wins, so the environment we build stays near the one the
    caller develops against. On a tie the older version wins, because Tesseracts
    are pinned behind the running Python far more often than ahead of it.
    """
    bounds = _python_bounds()
    usable = sorted(minors)
    if bounds is not None:
        usable = [minor for minor in usable if bounds[0] <= minor <= bounds[1]]
    if not usable:
        return None
    ours = sys.version_info.minor
    best = min(usable, key=lambda minor: (abs(minor - ours), minor > ours))
    return f"{sys.version_info.major}.{best}"


@functools.cache
def _compile_cached(
    specs: tuple[str, ...], python_version: str, *, wheels_only: bool
) -> str | None:
    """Resolve requirements for a Python version. None if they resolve.

    On failure, returns uv's error text. This only resolves; it creates no
    environment and downloads nothing, so it is cheap enough to call repeatedly.

    ``wheels_only`` passes uv's ``--no-build``, and we call this both ways
    because each reports a different problem:

    * without it, uv reports a version excluded by a dependency's
      ``Requires-Python``;
    * with it, uv reports a version with no wheel. Without it uv would quietly
      start building from source, and a stale pin then fails slowly.
    """
    with tempfile.TemporaryDirectory() as scratch:
        result = _capture(
            [
                *_uv(),
                "pip",
                "compile",
                "--quiet",
                *(("--no-build",) if wheels_only else ()),
                "--python-version",
                python_version,
                "-",
                "-o",
                Path(scratch) / "resolved.txt",
            ],
            stdin="\n".join(specs) + "\n",
        )
    if result.returncode == 0:
        return None
    return result.stderr or result.stdout or "resolution failed"


def _compile(
    specs: Sequence[str], python_version: str, *, wheels_only: bool
) -> str | None:
    """Cached :func:`_compile_cached`, since each call is a uv subprocess.

    :func:`_build_python_version` can reach a version it has already asked
    about.
    """
    return _compile_cached(tuple(specs), python_version, wheels_only=wheels_only)


# uv gives a dependency's Requires-Python twice in the same message: once as
# "... depends on Python>=3.12" and again as "... only supports >=3.12)" in the
# hint underneath. We accept either.
_REQUIRES_PYTHON = re.compile(
    r"(?:only supports|depends on Python)\s*"
    r"((?:[><=!~]=?\s*[\d.]+)(?:\s*,\s*[><=!~]=?\s*[\d.]+)*)"
)
# uv lists the versions a package does have wheels for, like
# "Python ABI tags: `cp39`, `cp310`, `cp311`, `cp312`".
_ABI_TAGS = re.compile(r"cp(\d)(\d+)")


def _minors_from_abi_tags(error: str) -> list[int]:
    """Python minor versions a package has wheels for, taken from uv's hint."""
    _, _, tail = error.partition("Python ABI tags")
    if not tail:
        return []
    return [
        int(minor)
        for major, minor in _ABI_TAGS.findall(tail)
        if int(major) == sys.version_info.major
    ]


def _requires_python(error: str) -> SpecifierSet | None:
    """The Requires-Python range uv named, if its message included one."""
    match = _REQUIRES_PYTHON.search(error)
    if not match:
        return None
    try:
        return SpecifierSet(match.group(1).strip())
    except InvalidSpecifier:
        return None


def _pylock_requires_python(lockfile: Path) -> SpecifierSet | None:
    """The Python range a PEP 751 lockfile says it is for, if it says one.

    A lockfile states this itself, which is just as well: `uv pip compile`
    refuses a `pylock.toml` outright, so the resolver cannot be asked the
    question the way it is for a flat requirements file.

    `requires-python` is a top-level key, so on Pythons without `tomllib` we
    read the lines above the first table header rather than take a dependency
    on a TOML parser for one field.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = None

    try:
        if tomllib is not None:
            with lockfile.open("rb") as handle:
                declared = tomllib.load(handle).get("requires-python")
        else:
            declared = None
            for line in lockfile.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("["):
                    break
                match = re.match(r"""requires-python\s*=\s*["'](.+?)["']""", stripped)
                if match:
                    declared = match.group(1)
                    break
    except (OSError, ValueError) as e:
        logger.debug("Could not read requires-python from %s: %s", lockfile, e)
        return None

    if not declared:
        return None
    try:
        return SpecifierSet(declared)
    except InvalidSpecifier:
        return None


def _minors_matching(wanted: SpecifierSet) -> list[int]:
    """Python minor versions a Requires-Python range allows.

    The upper limit is arbitrary and generous; `_nearest` applies the real
    bounds.
    """
    return [
        minor
        for minor in range(sys.version_info.minor + 50)
        if f"{sys.version_info.major}.{minor}.0" in wanted
    ]


def _build_python_version(build_config: Any, requirements_file: Path) -> str | None:
    """Choose a Python version to build on. None means use this interpreter.

    In a container the Python comes from the base image: 3.11 for the default
    ``debian:bookworm-slim``. That is why a Tesseract pinning ``numpy==1.26.4``
    builds fine there but not against 3.13, which has no wheel for it. We cannot
    know an arbitrary base image's Python version from here, so we ask uv what
    the requirements themselves need.

    We start at this interpreter and let uv tell us which way to go:

    1. If a dependency's ``Requires-Python`` excludes our version, uv names the
       range it does support. That is the answer.
    2. If a dependency simply has no wheel for us, uv often lists the versions
       it does publish. Also the answer.
    3. If neither, one resolution at each bound tells us the direction, and we
       walk that way only.

    Usually step 2 succeeds, so the common case is one resolution and no change
    of interpreter.

    Local paths are left out of the question we ask uv. They are always built
    from source, so including them would make ``--no-build`` reject every
    version and tell us nothing.
    """
    # The Tesseract named a version, so use it and nothing else. A build does
    # the same (`build_pip_venv.sh` passes it to `uv venv --python`). Picking a
    # different one would ignore what the author asked for and disagree with the
    # container. If it cannot be satisfied, the install will say so.
    declared = build_config.effective_python_version
    if declared is not None:
        return declared

    # Config validation rejects this together with `python_version`, so there
    # is no version to choose here.
    if build_config.inherit_base_image_packages:
        return None

    ours = sys.version_info.minor
    here = f"{sys.version_info.major}.{ours}"

    def chosen(version: str | None, why: str) -> str | None:
        """Log and return a pick, or None when it is the interpreter we are on."""
        if version is None:
            return None
        logger.debug("%s %s; building on %s", requirements_file.name, why, version)
        return None if version == here else version

    # A lockfile states its own Python range, and `uv pip compile` will not
    # accept one, so this is both the better answer and the only one available.
    if build_config.requirements.is_pylock:
        wanted = _pylock_requires_python(requirements_file)
        if wanted is None:
            return None
        return chosen(
            _nearest(_minors_matching(wanted)), f"is locked for Python {wanted}"
        )

    _, remote = parse_requirements(requirements_file)
    if not remote:
        return None

    # 1. Is our version excluded by something's Requires-Python? If so uv names
    #    the range it wants, which tells us where to go without trying.
    error = _compile(remote, here, wheels_only=False)
    wanted = _requires_python(error) if error is not None else None
    if wanted is not None:
        picked = chosen(_nearest(_minors_matching(wanted)), f"needs Python {wanted}")
        if picked is not None:
            return picked

    # 2. It resolves in principle. Is there a wheel for our version?
    wheel_error = _compile(remote, here, wheels_only=True)
    if wheel_error is None:
        return None

    #    uv often lists the versions the package does publish wheels for. That
    #    is the answer itself, not just a direction.
    from_tags = _minors_from_abi_tags(wheel_error) or _minors_from_abi_tags(error or "")
    if from_tags:
        picked = chosen(_nearest(from_tags), "publishes wheels for other versions")
        if picked is not None:
            return picked

    # 3. uv listed nothing, so work out the direction instead of assuming it.
    #    Only walking down would be wrong when we are already on the oldest
    #    version we support. A package that ships wheels only for a newer Python
    #    without declaring a floor is a common build-matrix slip, and step 1
    #    cannot catch it.
    bounds = _python_bounds()
    if bounds is None:
        return None
    floor, ceiling = bounds

    if (
        ceiling > ours
        and _compile(remote, f"{sys.version_info.major}.{ceiling}", wheels_only=True)
        is None
    ):
        direction = range(ours + 1, ceiling + 1)
    elif floor < ours:
        direction = range(ours - 1, floor - 1, -1)
    else:
        return None

    for minor in direction:
        version = f"{sys.version_info.major}.{minor}"
        if _compile(remote, version, wheels_only=True) is None:
            return chosen(version, f"has no wheels for Python {here}")

    # Nothing resolves on any version, so this tells us nothing. Build on ours
    # and let the install report whatever the real problem is.
    logger.debug("No Python version resolves %s; using this one", requirements_file)
    return None


def _ensure_runtime(
    python_executable: Path, installer: Sequence[Any] | None = None
) -> None:
    """Install ``tesseract-core[runtime]`` into an environment if it is missing.

    We check the installed version ourselves instead of letting the installer
    work it out. In a development checkout the runtime comes from a local source
    tree, and asking uv about it rebuilds the wheel every time, which costs
    about as long as starting a Tesseract.
    """
    from tesseract_core import __version__

    if _can_serve(python_executable, __version__):
        return

    spec = _runtime_install_spec()
    what = "Installing the Tesseract runtime"
    if installer is not None:
        _run([*installer, "install", spec], what)
    else:
        _run([*_uv(), "pip", "install", "--python", python_executable, spec], what)

    # An installer can report success and still leave the environment unable to
    # serve, for instance by installing somewhere other than where we asked.
    # Complain here, where we know what was attempted, instead of letting the
    # Tesseract fail to import much later on.
    if not _can_serve(python_executable):
        raise RuntimeError(
            f"Installed {spec} into {python_executable.parent.parent}, but it "
            "still cannot serve a Tesseract. Check that the installer put it "
            "there, or pass `python_executable` to name an environment "
            "yourself."
        )


def _ensure_venv(
    dest: Path,
    python_version: str | None = None,
    *,
    system_site_packages: bool = False,
) -> Path:
    """Create a uv virtual environment at `dest`, unless one is already there."""
    python_executable = _python_in(dest)
    if python_executable.is_file():
        return python_executable

    command = [*_uv(), "venv", str(dest)]
    # Config validation makes these mutually exclusive, as in a build.
    if python_version:
        command += ["--python", python_version]
    elif system_site_packages:
        command += ["--system-site-packages"]
    _run(command, f"Creating a virtual environment at {dest}")
    return python_executable


def _build_pip_venv(dest: Path, build_config: Any, requirements_file: Path) -> Path:
    """Create or update a uv virtual environment for a Tesseract.

    Follows ``templates/build_pip_venv.sh``, which a container build runs, so a
    Tesseract gets the same environment either way. Every step is idempotent, so
    this both creates and updates.
    """
    python_executable = _ensure_venv(
        dest,
        _build_python_version(build_config, requirements_file),
        system_site_packages=build_config.inherit_base_image_packages,
    )
    args, cwd = _install_from(requirements_file)
    _run(
        [*_uv(), "pip", "install", "--python", python_executable, *args],
        f"Installing {requirements_file.name}",
        cwd=cwd,
    )
    _ensure_runtime(python_executable)
    return python_executable


def _build_conda_env(dest: Path, requirements_file: Path) -> Path:
    """Create or update a conda environment for a Tesseract.

    Follows ``templates/build_conda_venv.sh``. conda has no quick way to check
    whether an environment already matches the file, and solving one takes
    minutes, so we compare a hash of the file against a stamp we wrote.
    """
    conda = _conda()
    python_executable = _python_in(dest)
    digest = hashlib.sha256(requirements_file.read_bytes()).hexdigest()
    stamp_path = dest / _CONDA_STAMP_NAME

    stamped = None
    if stamp_path.is_file():
        try:
            stamped = json.loads(stamp_path.read_text(encoding="utf-8")).get("digest")
        except ValueError:
            stamped = None

    if not python_executable.is_file():
        action = "create"
    elif stamped != digest:
        action = "update"
    else:
        action = None

    if action is not None:
        _run(
            [*conda, "env", action, "--file", requirements_file, "-p", dest, "--quiet"],
            f"Running `conda env {action}` for {dest}",
        )

    # Use the environment's own pip, not uv. Packages installed from conda
    # channels are not all visible to uv, so uv would decide they are missing
    # and reinstall them from PyPI.
    #
    # Invoked as `<env>/bin/python -m pip` rather than `conda run -p <env> pip`,
    # which is what `build_conda_venv.sh` does. Inside an image there is only one
    # pip to find, but on a host `conda run` resolves pip from PATH and can pick
    # one belonging to another environment, installing there instead.
    _ensure_runtime(python_executable, installer=[python_executable, "-m", "pip"])

    dest.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps({"digest": digest}), encoding="utf-8")
    return python_executable


def _managed_env_dir(api_path: Path) -> Path:
    """Where to put an environment we create for a Tesseract.

    Next to the ``tesseract_api.py``, since that is where people look for a
    ``.venv`` and where their tooling already ignores one. Some Tesseracts live
    somewhere we cannot write, such as site-packages or a read-only mount, so
    fall back to the user cache directory.
    """
    src_dir = api_path.parent
    if os.access(src_dir, os.W_OK):
        return src_dir / _MANAGED_VENV_NAME

    digest = hashlib.sha256(str(src_dir).encode()).hexdigest()[:12]
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "tesseract" / "envs" / f"{src_dir.name}-{digest}"


def resolve_python_executable(api_path: Path) -> Path:
    """Pick the interpreter to serve a Tesseract on, building one if needed.

    A Tesseract needs an environment with both its own dependencies and
    ``tesseract-core[runtime]`` in it. That runtime is an optional extra, so
    even the interpreter running this may not have it.

    We work out which interpreter to use from ``tesseract_config.yaml``, trying
    in order:

    1. an environment next to the ``tesseract_api.py`` that already has
       everything;
    2. the SDK's interpreter, if it already has everything.
       nothing at all;
    3. otherwise, create or update an environment next to the
       ``tesseract_api.py``.
    """
    this_interpreter = Path(sys.executable)
    declared = _declared_requirements(api_path)

    if declared is None:
        # Nothing to install, but the runtime is still needed, so this is not
        # automatically free. An SDK-only install cannot serve anything.
        if _can_serve(this_interpreter):
            return this_interpreter
        dest = _managed_env_dir(api_path)
        logger.info("Installing the Tesseract runtime into %s", dest)
        python_executable = _ensure_venv(dest)
        _ensure_runtime(python_executable)
        return python_executable

    build_config, requirements_file = declared

    if build_config.requirements.provider == "conda":
        # No point checking first. Packages from conda channels are not all
        # visible as pip distributions, so neither uv nor importlib.metadata can
        # tell us whether the environment is up to date. The stamp inside
        # `_build_conda_env` is what makes the second call cheap.
        return _build_conda_env(_managed_env_dir(api_path), requirements_file)

    for candidate in _VENV_CANDIDATES:
        python_executable = _python_in(api_path.parent / candidate)
        if not python_executable.is_file() or not _can_serve(python_executable):
            continue
        if _venv_satisfies(python_executable, requirements_file):
            logger.debug("Serving %s from %s", api_path.name, python_executable)
            return python_executable

    if _can_serve(this_interpreter):
        shortfall = _caller_shortfall(
            requirements_file, build_config.requirements.is_pylock
        )
        if shortfall == ([], []):
            logger.debug("Serving %s from the current interpreter", api_path.name)
            return this_interpreter
        if shortfall is not None:
            # A version that disagrees is reason enough to build. A Tesseract
            # pinning an old numpy may well be pinned because it breaks on a new
            # one, and serving it on the wrong version gives quietly wrong
            # answers. Building costs a couple of seconds, once.
            logger.debug(
                "Building for %s: missing %s; wrong version %s",
                api_path.name,
                shortfall[0] or "nothing",
                shortfall[1] or "nothing",
            )

    dest = _managed_env_dir(api_path)
    logger.info("Building an environment for %s at %s", api_path.name, dest)
    return _build_pip_venv(dest, build_config, requirements_file)
