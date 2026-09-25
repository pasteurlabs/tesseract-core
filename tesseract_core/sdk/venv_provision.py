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
    metadata,
)
from pathlib import Path
from typing import Any

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from .api_parse import ValidationError, get_config
from .config import get_config as get_sdk_config
from .engine import declared_requirements_file, get_runtime_dir
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
SCRUBBED_IMPORT_VARS = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")

# These redirect where pip installs to, regardless of the interpreter it was
# invoked with. A user who exports one would otherwise get an environment we
# reported building and that has nothing in it. Other `PIP_*` settings are left
# alone: an index url or a netrc is configuration we should honour.
_SCRUBBED_INSTALL_VARS = ("PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_USER")


# The environment we build belongs to us, so it gets a name nothing else uses.
# A `.venv` beside a Tesseract is the user's, and installing a Tesseract's pinned
# dependencies into it would be rude; we do not read it either, because picking
# up whatever happens to be lying next to the api file is the same guessing that
# `python_executable` exists to replace.
_MANAGED_VENV_NAME = ".tesseract-venv"

# Records what the environment was built from, so that a serve which changes
# nothing costs a file read instead of asking an installer.
_STAMP_NAME = ".tesseract-stamp.json"


def _python_in(prefix: Path) -> Path:
    """Path to the interpreter inside an environment directory."""
    if os.name == "nt":
        return prefix / "Scripts" / "python.exe"
    return prefix / "bin" / "python"


def _declared_requirements(api_path: Path) -> tuple[Any, Path | None]:
    """Read what a Tesseract says it needs installed.

    Returns its build config, and the requirements file its provider names or
    None when there is nothing to install. The build config comes back either
    way, because it carries settings that still apply to an environment with no
    requirements in it, such as ``python_version``.
    """
    src_dir = api_path.parent

    if not (src_dir / "tesseract_config.yaml").is_file():
        raise UserError(
            f"No tesseract_config.yaml next to {api_path.name}, so there is no "
            "way to tell what this Tesseract needs installed. Add one, or pass "
            "`python_executable` to name an interpreter that already has what "
            "it needs."
        )

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

    # Shared with the build, which asks the same question about the same file
    # and raises the same way when the provider cannot do without it. Whether
    # the file declares anything is left to the installer: an empty one is
    # legal to uv and to conda alike.
    return build_config, declared_requirements_file(src_dir, build_config)


def _uv() -> tuple[str, ...]:
    """How to invoke uv, or raise explaining what to do without it."""
    configured = get_sdk_config().uv_executable
    if configured:
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
    if configured:
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


@functools.cache
def _runtime_source_hash() -> str:
    """Digest of the runtime we would install.

    The SDK's `__version__` cannot stand in for this. It is baked into a
    generated `_version.py` when the SDK is installed, so editing the runtime in
    a development checkout leaves it unchanged and an environment built earlier
    would keep serving the old code.
    """
    digest = hashlib.sha256()
    for path in sorted(get_runtime_dir().rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _expected_stamp(requirements_file: Path | None) -> str:
    """What an environment built for these requirements should be stamped with."""
    digest = hashlib.sha256()
    digest.update(_runtime_source_hash().encode())
    if requirements_file is not None:
        digest.update(requirements_file.read_bytes())
    return digest.hexdigest()


def _stamp_matches(dest: Path, requirements_file: Path | None) -> bool:
    """Whether there is an environment at `dest` built from exactly this."""
    if not _python_in(dest).is_file():
        # A stamp without an interpreter beside it means someone removed part
        # of the environment; rebuild rather than hand back a path that is not
        # there.
        return False
    try:
        stamped = json.loads((dest / _STAMP_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return stamped.get("digest") == _expected_stamp(requirements_file)


def _write_stamp(dest: Path, requirements_file: Path | None) -> None:
    """Record what the environment was built from."""
    (dest / _STAMP_NAME).write_text(
        json.dumps({"digest": _expected_stamp(requirements_file)}), encoding="utf-8"
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
    dropped = SCRUBBED_IMPORT_VARS + _SCRUBBED_INSTALL_VARS
    env = {k: v for k, v in os.environ.items() if k not in dropped}
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


def _build_python_version(
    build_config: Any, requirements_file: Path | None
) -> str | None:
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

    if requirements_file is None:
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
    """Install the Tesseract runtime into an environment if it is missing.

    Built from this SDK's own source rather than fetched from an index, which
    is what a container does too. That works offline, works however the SDK
    itself was installed, and guarantees the Tesseract runs against the same
    code as the SDK that started it.

    Always installs: the stamp decides whether an environment needs rebuilding
    at all, so reaching here means it does.
    """
    from .engine import stage_runtime_package

    what = "Installing the Tesseract runtime"
    with tempfile.TemporaryDirectory() as scratch:
        staged = stage_runtime_package(Path(scratch) / "tesseract_runtime")
        if installer is not None:
            # pip compiles bytecode by default; uv has to be asked.
            _run([*installer, "install", staged], what)
        else:
            _run(
                [
                    *_uv(),
                    "pip",
                    "install",
                    "--compile-bytecode",
                    "--python",
                    python_executable,
                    staged,
                ],
                what,
            )

    # An installer can report success and still leave the environment unable to
    # serve, for instance by installing somewhere other than where we asked.
    # Complain here, where we know what was attempted, instead of letting the
    # Tesseract fail to import much later on.


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


def _build_pip_venv(
    dest: Path, build_config: Any, requirements_file: Path | None
) -> Path:
    """Create or update a uv virtual environment for a Tesseract.

    Follows ``templates/build_pip_venv.sh``, which a container build runs, so a
    Tesseract gets the same environment either way. Every step is idempotent, so
    this both creates and updates.

    Bytecode is compiled during the install, as it is in a build. Otherwise the
    first import pays for it, and that happens while the caller is waiting for
    the Tesseract to answer a health check.
    """
    python_executable = _ensure_venv(
        dest,
        _build_python_version(build_config, requirements_file),
        system_site_packages=build_config.inherit_base_image_packages,
    )
    if requirements_file is not None:
        args, cwd = _install_from(requirements_file)
        _run(
            [
                *_uv(),
                "pip",
                "install",
                "--compile-bytecode",
                "--python",
                python_executable,
                *args,
            ],
            f"Installing {requirements_file.name}",
            cwd=cwd,
        )
    _ensure_runtime(python_executable)
    _write_stamp(dest, requirements_file)
    return python_executable


def _build_conda_env(dest: Path, requirements_file: Path) -> Path:
    """Create or update a conda environment for a Tesseract.

    Follows ``templates/build_conda_venv.sh``. Reaching here means the stamp
    said something changed, so the environment is created or brought up to date
    unconditionally.
    """
    conda = _conda()
    python_executable = _python_in(dest)

    action = "create" if not python_executable.is_file() else "update"
    _run(
        [*conda, "env", action, "--file", requirements_file, "-p", dest, "--quiet"],
        f"Running `conda env {action}` for {dest}",
    )

    if not python_executable.is_file():
        # conda is happy to create an environment from a file that asks for
        # nothing, and what it makes then has no interpreter in it. Say that
        # here, rather than failing on a missing `python` a step later.
        raise UserError(
            f"{requirements_file.name} produced an environment with no Python "
            f"in {dest}. Declare a python version in it, for example "
            "`dependencies: [python=3.12]`."
        )

    # Use the environment's own pip, not uv. Packages installed from conda
    # channels are not all visible to uv, so uv would decide they are missing
    # and reinstall them from PyPI.
    _ensure_runtime(python_executable, installer=[python_executable, "-m", "pip"])

    _write_stamp(dest, requirements_file)
    return python_executable


def _managed_env_dir(api_path: Path) -> Path:
    """Where to put the environment we build for a Tesseract.

    Next to the ``tesseract_api.py``, under a name of our own. uv writes a
    ``.gitignore`` into every environment it creates, so this stays out of the
    user's repository despite not being a name their tooling knows.
    """
    src_dir = api_path.parent
    if not os.access(src_dir, os.W_OK):
        raise UserError(
            f"Cannot build an environment for {api_path.name}: {src_dir} is not "
            "writable. Make it writable, or pass `python_executable` to name an "
            "interpreter that already has what the Tesseract needs."
        )
    return src_dir / _MANAGED_VENV_NAME


def resolve_python_executable(api_path: Path) -> Path:
    """Pick the interpreter to serve a Tesseract on, building one if needed.

    The environment is one we build and own, next to the ``tesseract_api.py``.
    Nothing else is considered: not the interpreter the SDK runs on, and not a
    ``.venv`` the user happens to keep beside their Tesseract. Either would make
    behaviour depend on what is lying around rather than on what the Tesseract
    declares, which shows up as a constructor that was instant taking seconds
    because something unrelated was upgraded, or a Tesseract importing a package
    it never declared. ``python_executable`` is how to supply your own, and
    skips all of this.

    A stamp of the requirements and of the runtime source says whether the
    environment is still what it should be, so a serve that changes nothing
    costs a file read. It covers the runtime source because the SDK's version
    is fixed when the SDK is installed, and would not notice a runtime edited
    in a development checkout.
    """
    dest = _managed_env_dir(api_path)
    build_config, requirements_file = _declared_requirements(api_path)

    if _stamp_matches(dest, requirements_file):
        logger.debug("Serving %s from %s", api_path.name, dest)
        return _python_in(dest)

    if build_config.requirements.provider == "conda":
        return _build_conda_env(dest, requirements_file)

    logger.info("Building an environment for %s at %s", api_path.name, dest)
    return _build_pip_venv(dest, build_config, requirements_file)
