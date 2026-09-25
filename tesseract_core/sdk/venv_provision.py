# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a Tesseract's Python environment on the host, for ``from_source``.

The host-side counterpart of the ``build_*_venv.sh`` templates run by
``tesseract build``. Only :func:`resolve_python_executable` and :func:`declared_env` are public.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from .api_parse import ValidationError, get_config
from .config import get_config as get_sdk_config
from .engine import (
    _split_local_dependency,
    declared_requirements_file,
    get_runtime_dependencies,
    get_runtime_dir,
    parse_requirements,
)
from .exceptions import UserError

logger = logging.getLogger("tesseract")


# Would point another interpreter at this process's packages.
SCRUBBED_IMPORT_VARS = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")

# Would redirect installs away from the environment we are building. Other
# PIP_* settings (index urls etc.) are honoured.
_SCRUBBED_INSTALL_VARS = ("PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_USER")


# The Python in the default `build_config.base_image`, which `tesseract build`
# uses when no python_version is set. debian:bookworm-slim is Debian 12, which
# ships 3.11 for its whole lifetime; a test fails if the default image changes.
DEFAULT_BASE_IMAGE_PYTHON = "3.11"

# Deliberately not `.venv`: that one is the user's, and we neither read nor write it.
_MANAGED_VENV_NAME = ".tesseract-venv"

# Digest of what the environment was built from; see resolve_python_executable.
_STAMP_NAME = ".tesseract-stamp.json"


def _python_in(prefix: Path) -> Path:
    """Path to the interpreter inside an environment directory."""
    if os.name == "nt":
        return prefix / "Scripts" / "python.exe"
    return prefix / "bin" / "python"


def _declared_requirements(api_path: Path) -> tuple[Any, Path | None]:
    """Return the build config and requirements file (None if there is none)."""
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
    except ValidationError as e:
        raise UserError(
            f"Could not read {src_dir / 'tesseract_config.yaml'}, so there is no "
            f"way to tell what this Tesseract needs installed: {e}"
        ) from e

    return build_config, declared_requirements_file(src_dir, build_config)


def _uv() -> tuple[str, ...]:
    """How to invoke uv, or raise explaining what to do without it."""
    uv = get_sdk_config().uv_executable
    if shutil.which(uv[0]) is None:
        raise RuntimeError(
            "uv is needed to build an environment for this Tesseract and was "
            "not found on PATH. Install it "
            "(https://docs.astral.sh/uv/getting-started/installation/), point "
            "TESSERACT_UV_EXECUTABLE at it, or pass `python_executable` to name "
            "an interpreter that already has the Tesseract's dependencies and "
            "`tesseract-core[runtime]`."
        )
    return uv


def _conda() -> tuple[str, ...]:
    """How to invoke conda, or raise explaining what to do without it."""
    conda = get_sdk_config().conda_executable
    if shutil.which(conda[0]) is None:
        raise RuntimeError(
            "This Tesseract declares `requirements.provider: conda`, but conda "
            f"(`{conda[0]}`) was not found. Install it "
            "(https://docs.conda.io/projects/conda/en/latest/user-guide/install/), "
            "point TESSERACT_CONDA_EXECUTABLE at it or at a conda-compatible tool "
            "such as mamba, or pass `python_executable` to name an interpreter "
            "that already has the Tesseract's dependencies and "
            "`tesseract-core[runtime]`."
        )
    return conda


@functools.cache
def _runtime_source_hash() -> str:
    """Digest of the runtime source.

    Not `__version__`, which does not change when the runtime is edited in a
    development checkout.
    """
    digest = hashlib.sha256()
    for path in sorted(get_runtime_dir().rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _expected_stamp(build_config: Any, requirements_file: Path | None) -> str:
    """Digest of everything an environment is built from."""
    digest = hashlib.sha256()
    digest.update(_runtime_source_hash().encode())
    digest.update(json.dumps(get_runtime_dependencies()).encode())
    provider = build_config.requirements.provider
    settings = {
        "provider": provider,
        "python_version": _build_python_version(build_config, requirements_file)
        if provider == "uv-pip"
        else None,
        "inherit_base_image_packages": build_config.inherit_base_image_packages,
        "build_env": build_config.build_env,
    }
    digest.update(json.dumps(settings, sort_keys=True).encode())
    if requirements_file is not None:
        digest.update(requirements_file.read_bytes())
    return digest.hexdigest()


def _stamp_matches(dest: Path, stamp: str) -> bool:
    """Whether there is an environment at `dest` stamped with `stamp`."""
    if not _python_in(dest).is_file():
        return False
    try:
        stamped = json.loads((dest / _STAMP_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return stamped.get("digest") == stamp


def _write_stamp(dest: Path, stamp: str) -> None:
    """Record what the environment was built from."""
    (dest / _STAMP_NAME).write_text(json.dumps({"digest": stamp}), encoding="utf-8")


def _run(
    command: Sequence[Any],
    what: str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command with extra `env` on top of ours, raising if it fails."""
    argv = [str(part) for part in command]
    logger.debug("Running %s", " ".join(argv))
    # Importing a tesseract_api.py sets PYTHONPATH in this process.
    dropped = SCRUBBED_IMPORT_VARS + _SCRUBBED_INSTALL_VARS
    full_env = {k: v for k, v in os.environ.items() if k not in dropped}
    full_env.update(env or {})
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        env=full_env,
        cwd=None if cwd is None else str(cwd),
    )
    if result.returncode != 0:
        output = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"{what} failed:\n{output[-2000:]}")


def _local_directories(requirements_file: Path) -> list[str]:
    """Local directory requirements, as editable install specs.

    Installed editable so that edits show up without a rebuild; a container
    build installs a copy instead.
    """
    local, _ = parse_requirements(requirements_file)
    specs = []
    for line in local:
        path, extras = _split_local_dependency(line)
        resolved = (requirements_file.parent / path).resolve()
        if resolved.is_dir():
            specs.append(f"{resolved}{extras}")
    return specs


def _pylock_requires_python(lockfile: Path) -> SpecifierSet | None:
    """A PEP 751 lockfile's top-level `requires-python`, if any.

    The key is optional, but an unreadable file or invalid value raises. Without
    `tomllib` (3.10) the lines before the first table are scanned.
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
        raise UserError(f"Could not read {lockfile}: {e}") from e

    if declared is None:
        return None
    try:
        return SpecifierSet(declared)
    except (InvalidSpecifier, TypeError) as e:
        raise UserError(
            f"{lockfile.name} has an invalid requires-python: {declared!r}"
        ) from e


def _build_python_version(
    build_config: Any, requirements_file: Path | None
) -> str | None:
    """The Python to build a uv-pip environment on, as a `uv venv --python` request.

    Whatever a container build would use: the declared ``python_version``, else
    the default base image's Python. A lockfile whose ``requires-python``
    excludes that gets its range instead. None (for
    ``inherit_base_image_packages``) leaves the choice to uv.
    """
    declared = build_config.effective_python_version
    if declared is not None:
        return declared
    if build_config.inherit_base_image_packages:
        return None
    if requirements_file is not None and build_config.requirements.is_pylock:
        wanted = _pylock_requires_python(requirements_file)
        if wanted is not None and f"{DEFAULT_BASE_IMAGE_PYTHON}.0" not in wanted:
            return str(wanted)
    return DEFAULT_BASE_IMAGE_PYTHON


def _ensure_runtime(
    python_executable: Path,
    installer: Sequence[Any] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Install the runtime, staged from this SDK's source as in a container build."""
    from .engine import stage_runtime_package

    what = "Installing the Tesseract runtime"
    with tempfile.TemporaryDirectory() as scratch:
        staged = stage_runtime_package(Path(scratch) / "tesseract_runtime")
        if installer is not None:
            # pip compiles bytecode by default; uv has to be asked.
            _run([*installer, "install", staged], what, env=env)
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
                env=env,
            )


def _create_venv(
    dest: Path,
    python_version: str | None = None,
    *,
    system_site_packages: bool = False,
    env: dict[str, str] | None = None,
) -> Path:
    """Create a uv virtual environment at `dest` and return its interpreter."""
    command = [*_uv(), "venv", str(dest)]
    # Config validation makes these mutually exclusive, as in a build.
    if python_version:
        command += ["--python", python_version]
    elif system_site_packages:
        command += ["--system-site-packages"]
    _run(command, f"Creating a virtual environment at {dest}", env=env)
    return _python_in(dest)


def _build_pip_venv(
    dest: Path, build_config: Any, requirements_file: Path | None
) -> Path:
    """Build a uv environment, following ``build_pip_venv.sh``."""
    env = build_config.build_env
    python_version = _build_python_version(build_config, requirements_file)
    python_executable = _create_venv(
        dest,
        python_version,
        system_site_packages=build_config.inherit_base_image_packages,
        env=env,
    )
    if requirements_file is not None:
        try:
            # `-r` handles flat files and PEP 751 lockfiles alike, as in
            # build_pip_venv.sh. Run from the file's directory so relative local
            # paths in it resolve.
            _run(
                [
                    *_uv(),
                    "pip",
                    "install",
                    "--compile-bytecode",
                    "--python",
                    python_executable,
                    "-r",
                    requirements_file.name,
                ],
                f"Installing {requirements_file.name}",
                cwd=requirements_file.parent,
                env=env,
            )
        except RuntimeError as e:
            if build_config.effective_python_version is not None:
                raise
            raise RuntimeError(
                f"{e}\n\nThis environment was built on Python {python_version}, "
                "the default (as in `tesseract build`). If a dependency needs a "
                "different Python, set `build_config.requirements.python_version` "
                "in tesseract_config.yaml."
            ) from e
        if not build_config.requirements.is_pylock:
            editable = _local_directories(requirements_file)
            if editable:
                _run(
                    [
                        *_uv(),
                        "pip",
                        "install",
                        "--python",
                        python_executable,
                        *(arg for spec in editable for arg in ("-e", spec)),
                    ],
                    "Installing local requirements as editable",
                    env=env,
                )
    _ensure_runtime(python_executable, env=env)
    return python_executable


def _build_conda_env(
    dest: Path, requirements_file: Path, env: dict[str, str] | None = None
) -> Path:
    """Build a conda environment, following ``build_conda_venv.sh``."""
    python_executable = _python_in(dest)
    _run(
        [
            *_conda(),
            "env",
            "create",
            "--file",
            requirements_file,
            "-p",
            dest,
            "--quiet",
        ],
        f"Running `conda env create` for {dest}",
        env=env,
    )

    if not python_executable.is_file():
        # conda happily creates an environment with no python in it.
        raise UserError(
            f"{requirements_file.name} produced an environment with no Python "
            f"in {dest}. Declare a python version in it, for example "
            "`dependencies: [python=3.12]`."
        )

    # Not uv, which cannot see all conda-installed packages and would
    # reinstall them from PyPI.
    _ensure_runtime(
        python_executable, installer=[python_executable, "-m", "pip"], env=env
    )
    return python_executable


def declared_env(api_path: Path) -> dict[str, str]:
    """The config's top-level ``env``, which a container sets in its image.

    Empty if there is no readable config, since an explicit ``python_executable``
    does not require one.
    """
    try:
        return dict(get_config(api_path.parent).env)
    except (OSError, ValidationError):
        return {}


@contextlib.contextmanager
def _build_lock(dest: Path):
    """Hold an exclusive, cross-process lock for building `dest`.

    The lock file lives in the temp dir, keyed on `dest`, so nothing is left
    in the Tesseract's directory.
    """
    key = hashlib.sha256(str(dest.resolve()).encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"tesseract-venv-{key}.lock"
    with open(lock_path, "a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    pass  # LK_LOCK gives up after ~10s; keep waiting.
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
            yield  # Released when the file is closed.


def resolve_python_executable(api_path: Path) -> Path:
    """Return the interpreter of the Tesseract's managed environment, building it if needed.

    The environment lives in ``.tesseract-venv`` next to ``tesseract_api.py``.
    Neither ``sys.executable`` nor a user's ``.venv`` is ever used; pass
    ``python_executable`` to ``serve`` to supply your own. When the stamp (a
    digest of the requirements, build settings and runtime source) goes stale,
    the environment is rebuilt from scratch.
    """
    build_config, requirements_file = _declared_requirements(api_path)
    src_dir = api_path.parent
    dest = src_dir / _MANAGED_VENV_NAME
    stamp = _expected_stamp(build_config, requirements_file)

    if _stamp_matches(dest, stamp):
        logger.debug("Serving %s from %s", api_path.name, dest)
        return _python_in(dest)

    if not os.access(src_dir, os.W_OK):
        raise UserError(
            f"Cannot build an environment for {api_path.name}: {src_dir} is not "
            "writable. Make it writable, or pass `python_executable` to name an "
            "interpreter that already has what the Tesseract needs."
        )

    with _build_lock(dest):
        # Another process may have built it while we waited.
        if _stamp_matches(dest, stamp):
            return _python_in(dest)
        return _build(api_path, dest, stamp, build_config, requirements_file)


def _build(
    api_path: Path,
    dest: Path,
    stamp: str,
    build_config: Any,
    requirements_file: Path | None,
) -> Path:
    """Build the managed environment at `dest` from scratch and stamp it."""
    logger.info("Building an environment for %s at %s", api_path.name, dest)
    if build_config.host_credentials:
        logger.warning(
            "build_config.host_credentials only applies to `tesseract build` and "
            "is ignored here; authenticated hosts must be reachable with your own "
            "credentials (e.g. ~/.netrc, a keyring, or UV_INDEX_* variables)."
        )
    # Installing over the old environment would keep packages no longer declared.
    if dest.exists():
        shutil.rmtree(dest)

    if build_config.requirements.provider == "conda":
        python_executable = _build_conda_env(
            dest, requirements_file, build_config.build_env
        )
    else:
        python_executable = _build_pip_venv(dest, build_config, requirements_file)

    _write_stamp(dest, stamp)
    return python_executable
