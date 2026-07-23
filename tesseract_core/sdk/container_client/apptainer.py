# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apptainer container backend for Tesseract usage.

Apptainer (formerly Singularity) is the de-facto container runtime on HPC systems,
where a root Docker daemon is unavailable. Unlike Docker it has no daemon-side
image store, cannot build from Dockerfiles, and is host-network only. This backend
implements the :class:`~.base.ContainerClient` protocol on top of the Apptainer
CLI, managing its own on-disk store of SIF images.

See ``apptainer-backend-plan.md`` and ``apptainer-phase0-findings.md`` for the
design rationale and the spike results that shaped it (notably: serving uses
``apptainer instance start`` followed by ``apptainer exec instance:// … serve``,
because ``instance start`` does not run the OCI entrypoint of a converted image).
"""

import contextlib
import fcntl
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

# store a reference to the list type, which is shadowed by some method names below
from typing import List as list_  # noqa: UP035

from tesseract_core.sdk.config import get_config

from .base import AbstractContainer, AbstractImage, Capabilities
from .exceptions import (
    APIError,
    ContainerError,
    ImageNotFound,
    NotFound,
)

logger = logging.getLogger("tesseract")

BoolOrCallable: TypeAlias = bool | Callable[[str], Any]

#: Flags applied to every apptainer run/exec to emulate Docker isolation: contain
#: the filesystem/PID/IPC/env, clean host env vars, and provide a writable tmpfs
#: overlay so the read-only SIF behaves like a writable container root.
ISOLATION_FLAGS = ("--containall", "--cleanenv", "--writable-tmpfs")

#: Path to the image entrypoint inside a Tesseract SIF. Serving goes through
#: ``apptainer exec`` (which bypasses the runscript), so the entrypoint is invoked
#: explicitly to seed nss_wrapper's passwd/group files (see phase 0 findings).
ENTRYPOINT_PATH = "/tesseract/entrypoint.sh"
RUNTIME_ENTRYPOINT = (ENTRYPOINT_PATH, "tesseract-runtime")


# --------------------------------------------------------------------------- #
# Process helpers (mirrors the Docker backend, kept local to avoid coupling)   #
# --------------------------------------------------------------------------- #


def _get_apptainer_executable() -> list_[str]:  # noqa: UP006
    """Get the Apptainer executable command, validating it exists."""
    config = get_config()
    exe = config.apptainer_executable
    if isinstance(exe, str):
        exe = shlex.split(exe)
    exe = list(exe)
    resolved = shutil.which(exe[0])
    if resolved is None:
        raise APIError(
            f"Apptainer executable {exe[0]!r} not found. Install Apptainer or set "
            "TESSERACT_APPTAINER_EXECUTABLE."
        )
    return [resolved, *exe[1:]]


def _get_io_callable(
    stream: BoolOrCallable,
    default_stream: Callable[[str], Any],
) -> Callable[[str], Any] | None:
    """Resolve a stream flag to a callable sink (or None)."""
    if stream is False:
        return None
    if stream is True:
        return default_stream
    if callable(stream):
        return stream
    raise ValueError(
        "stream_stdout/stream_stderr must be a boolean or a callable that accepts a string."
    )


def _read_stream(stream: Any, collected: list[bytes], echo_to: Any = None) -> None:
    """Read lines from a subprocess pipe, collecting and optionally echoing them."""
    while True:
        line = stream.readline()
        if not line:
            break
        collected.append(line)
        if echo_to is not None:
            decoded = line.decode("utf-8", errors="replace")
            if callable(echo_to) and not hasattr(echo_to, "write"):
                echo_to(decoded.rstrip("\n"))
            else:
                echo_to.write(decoded)
                echo_to.flush()


def _run_process(
    cmd: list[str],
    *,
    stream_stdout: Callable[[str], Any] | None = None,
    stream_stderr: Callable[[str], Any] | None = None,
) -> tuple[int, bytes, bytes]:
    """Run a subprocess with threaded stream reading, mirroring the Docker backend."""
    logger.debug(f"Running command: {cmd}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    stdout_lines: list[bytes] = []
    stderr_lines: list[bytes] = []
    try:
        stdout_thread = threading.Thread(
            target=_read_stream, args=(proc.stdout, stdout_lines, stream_stdout)
        )
        stdout_thread.start()
        stderr_thread = threading.Thread(
            target=_read_stream, args=(proc.stderr, stderr_lines, stream_stderr)
        )
        stderr_thread.start()
        proc.wait()
        stdout_thread.join()
        stderr_thread.join()
    except (KeyboardInterrupt, Exception):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    finally:
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()
    return proc.returncode, b"".join(stdout_lines), b"".join(stderr_lines)


# --------------------------------------------------------------------------- #
# SIF image store                                                              #
# --------------------------------------------------------------------------- #


def get_image_dir() -> Path:
    """Return the SIF store directory, creating it if necessary.

    Defaults to ``~/.local/share/tesseract/images`` (honoring ``XDG_DATA_HOME``),
    overridable via ``TESSERACT_APPTAINER_IMAGE_DIR``. Configurable because ``$HOME``
    is often small on clusters where scratch storage is preferred.
    """
    configured = get_config().apptainer_image_dir
    if configured:
        base = Path(configured).expanduser()
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        root = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
        base = root / "tesseract" / "images"
    base.mkdir(parents=True, exist_ok=True)
    return base


@contextlib.contextmanager
def _store_lock():
    """Advisory file lock serializing mutations of the SIF store.

    Guards against races when parallel jobs on a shared filesystem pull or remove
    images concurrently (see the risks table in the plan).
    """
    lock_path = get_image_dir() / ".lock"
    with open(lock_path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _sif_path(name: str, tag: str) -> Path:
    """Return the on-disk path for a stored image ``name:tag``."""
    return get_image_dir() / name / f"{tag}.sif"


def _sidecar_path(sif: Path) -> Path:
    """Return the metadata sidecar path for a SIF file."""
    return sif.with_suffix(".json")


def _split_ref(reference: str) -> tuple[str, str]:
    """Split a store reference ``name:tag`` into ``(name, tag)``, defaulting tag.

    The name may itself contain slashes (registry-style), but not a tag with a
    colon in the final segment ambiguously — we split on the last colon only if
    the segment after it has no slash.
    """
    if ":" in reference:
        head, tail = reference.rsplit(":", 1)
        if "/" not in tail:
            return head, tail
    return reference, "latest"


def inspect_sif(sif_path: str | Path) -> dict:
    """Return the parsed ``apptainer inspect --json`` metadata for a SIF file."""
    apptainer = _get_apptainer_executable()
    result = subprocess.run(
        [*apptainer, "inspect", "--json", str(sif_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise APIError(f"Cannot inspect {sif_path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _sif_labels(inspect_data: dict) -> dict:
    """Extract the labels dict from ``apptainer inspect --json`` output."""
    return inspect_data.get("data", {}).get("attributes", {}).get("labels", {}) or {}


def _sif_environment(sif_path: str | Path) -> dict[str, str]:
    """Return the image environment variables baked into a SIF.

    Parses ``apptainer inspect --environment`` output of the form
    ``export KEY="${KEY:-"value"}"``. Used to detect Tesseract images via
    ``TESSERACT_NAME`` (which survives conversion; the metadata label may be empty).
    """
    apptainer = _get_apptainer_executable()
    result = subprocess.run(
        [*apptainer, "inspect", "--environment", str(sif_path)],
        capture_output=True,
        text=True,
    )
    env: dict[str, str] = {}
    if result.returncode != 0:
        return env
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        assignment = line[len("export ") :]
        if "=" not in assignment:
            continue
        key, _, raw = assignment.partition("=")
        # Values look like: "${KEY:-"actual value"}" — pull out the default.
        value = raw.strip()
        if ":-" in value:
            value = value.split(":-", 1)[1].rstrip('}"')
        env[key.strip()] = value.strip().strip('"')
    return env


def _is_tesseract_sif(sif_path: str | Path) -> bool:
    """Check whether a SIF is a Tesseract image (has TESSERACT_NAME)."""
    return "TESSERACT_NAME" in _sif_environment(sif_path)


# --------------------------------------------------------------------------- #
# Image / Container data classes                                              #
# --------------------------------------------------------------------------- #


@dataclass
class Image(AbstractImage):
    """Apptainer SIF image, mirroring the Docker backend's Image interface."""

    id: str | None
    short_id: str | None
    tags: list[str] | None
    attrs: dict
    #: Absolute path to the SIF file backing this image.
    path: str | None = None

    @classmethod
    def from_store(cls, sif_path: Path, ref: str, inspect_data: dict) -> "Image":
        """Build an Image from a stored SIF and its inspect metadata."""
        # SIF files have no content-addressable Docker-style ID; use the file's
        # canonical path as a stable identifier.
        image_id = str(sif_path.resolve())
        return cls(
            id=image_id,
            short_id=image_id,
            tags=[ref],
            attrs={
                "Id": image_id,
                "RepoTags": [ref],
                "Config": {
                    "Env": [f"{k}={v}" for k, v in _sif_environment(sif_path).items()]
                },
                "Labels": _sif_labels(inspect_data),
                "Path": image_id,
            },
            path=image_id,
        )


class Images:
    """Namespace for Tesseract SIF images in the store."""

    @staticmethod
    def get(image_id_or_name: str | bytes, tesseract_only: bool = True) -> Image:
        """Return the image for a store reference or a direct path to a SIF file."""
        if not image_id_or_name:
            raise ValueError("Image name cannot be empty.")
        ref = (
            image_id_or_name.decode()
            if isinstance(image_id_or_name, bytes)
            else str(image_id_or_name)
        )

        # A direct path to a .sif file bypasses the store.
        candidate = Path(ref).expanduser()
        if candidate.suffix == ".sif" and candidate.is_file():
            sif_path = candidate.resolve()
            store_ref = f"{sif_path.parent.name}/{sif_path.stem}"
        else:
            name, tag = _split_ref(ref)
            sif_path = _sif_path(name, tag)
            store_ref = f"{name}:{tag}"
            if not sif_path.is_file():
                raise ImageNotFound(
                    f"Image {ref} not found in the Apptainer image store "
                    f"({get_image_dir()}). Pull or build it first."
                )

        if tesseract_only and not _is_tesseract_sif(sif_path):
            raise ImageNotFound(f"Image {ref} is not a Tesseract image.")
        return Image.from_store(sif_path, store_ref, inspect_sif(sif_path))

    @staticmethod
    def list(tesseract_only: bool = True) -> list_[Image]:  # noqa: UP006
        """List all images in the SIF store."""
        images = []
        image_dir = get_image_dir()
        for sif_path in sorted(image_dir.glob("*/*.sif")):
            ref = f"{sif_path.parent.name}:{sif_path.stem}"
            try:
                if tesseract_only and not _is_tesseract_sif(sif_path):
                    continue
                images.append(Image.from_store(sif_path, ref, inspect_sif(sif_path)))
            except APIError as ex:
                logger.warning(f"Skipping unreadable SIF {sif_path}: {ex}")
        return images

    @staticmethod
    def remove(image: str) -> None:
        """Remove an image (store reference or SIF path) from the store."""
        with _store_lock():
            candidate = Path(image).expanduser()
            if candidate.suffix == ".sif" and candidate.is_file():
                sif_path = candidate.resolve()
            else:
                name, tag = _split_ref(image)
                sif_path = _sif_path(name, tag)
            if not sif_path.is_file():
                raise ImageNotFound(f"Cannot remove image {image}: not found.")
            sif_path.unlink()
            sidecar = _sidecar_path(sif_path)
            if sidecar.is_file():
                sidecar.unlink()
            # Clean up an empty name directory.
            parent = sif_path.parent
            if parent != get_image_dir() and not any(parent.iterdir()):
                parent.rmdir()

    @staticmethod
    def store(sif_source: str | Path, name: str, tag: str = "latest") -> Image:
        """Move/copy a SIF into the store under ``name:tag`` and return its Image.

        Used by ``pull`` and ``build --output-format sif``. Validates that the SIF
        is a Tesseract image before committing it to the store.
        """
        sif_source = Path(sif_source)
        if not _is_tesseract_sif(sif_source):
            raise ImageNotFound(
                f"{sif_source} is not a Tesseract image (no TESSERACT_NAME); "
                "refusing to add it to the store."
            )
        with _store_lock():
            dest = _sif_path(name, tag)
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Replace atomically-ish: move onto the destination.
            shutil.move(str(sif_source), str(dest))
        return Images.get(f"{name}:{tag}")


@dataclass
class Container(AbstractContainer):
    """Apptainer instance, mirroring the Docker backend's Container interface.

    Backs a served Tesseract, which under Apptainer is an ``apptainer instance``
    plus a detached ``exec`` running ``serve``. Because Apptainer is host-network
    only, the container binds a host port directly, so ``host_port`` is simply the
    port the runtime was told to serve on (recorded in the sidecar metadata).
    """

    id: str
    short_id: str
    name: str
    attrs: dict = field(default_factory=dict)

    @classmethod
    def from_instance(cls, instance: dict, sidecar: dict) -> "Container":
        """Build a Container from an ``instance list --json`` entry and sidecar."""
        name = instance.get("instance", "")
        # Populate the full image environment (TESSERACT_NAME/VERSION/DESCRIPTION,
        # ...) so the CLI can render the same metadata it does for Docker.
        sif_path = sidecar.get("sif_path")
        env_list = []
        if sif_path and Path(sif_path).is_file():
            env_list = [f"{k}={v}" for k, v in _sif_environment(sif_path).items()]
        if not any(item.startswith("TESSERACT_NAME=") for item in env_list):
            env_list.append(f"TESSERACT_NAME={sidecar.get('tesseract_name', '')}")
        return cls(
            id=name,
            short_id=name,
            name=name,
            attrs={
                "Instance": instance,
                "Sidecar": sidecar,
                "Config": {
                    "Env": env_list,
                    "Cmd": ["serve", "--port", str(sidecar.get("port", ""))],
                    "Entrypoint": list(RUNTIME_ENTRYPOINT),
                },
            },
        )

    @property
    def image(self) -> Image | None:
        """Return the backing image, resolved from the sidecar's SIF path."""
        sif_path = self.attrs.get("Sidecar", {}).get("sif_path")
        if not sif_path or not Path(sif_path).is_file():
            return None
        return Image.from_store(
            Path(sif_path),
            self.attrs.get("Sidecar", {}).get("image_ref", ""),
            inspect_sif(sif_path),
        )

    @property
    def host_port(self) -> str | None:
        """Return the host port the served Tesseract is bound to."""
        port = self.attrs.get("Sidecar", {}).get("port")
        return str(port) if port is not None else None

    @property
    def api_port(self) -> str | None:
        """Alias of host_port; Apptainer has no host/container port distinction."""
        return self.host_port

    @property
    def host_ip(self) -> str | None:
        """Return the host IP the served Tesseract is bound to (always localhost)."""
        return self.attrs.get("Sidecar", {}).get("host_ip", "127.0.0.1")

    @property
    def host_debugpy_port(self) -> str | None:
        """Return the host port debugpy is listening on, if debug was enabled."""
        port = self.attrs.get("Sidecar", {}).get("debugpy_port")
        return str(port) if port is not None else None

    @property
    def docker_network_ips(self) -> dict | None:
        """Apptainer has no user-defined networks; always None."""
        return None

    @property
    def status(self) -> str:
        """Return 'running' if the instance is live, else 'exited'."""
        return "running" if _instance_exists(self.name) else "exited"

    def exec_run(self, command: list) -> tuple[int, bytes]:
        """Run a command inside the instance."""
        apptainer = _get_apptainer_executable()
        result = subprocess.run(
            [*apptainer, "exec", f"instance://{self.name}", *command],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise ContainerError(
                self.id,
                result.returncode,
                shlex.join(command),
                self.image.id if self.image else "unknown",
                result.stderr,
            )
        return result.returncode, result.stdout

    def stop(self) -> None:
        """Stop the underlying instance."""
        apptainer = _get_apptainer_executable()
        result = subprocess.run(
            [*apptainer, "instance", "stop", self.name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and _instance_exists(self.name):
            raise APIError(f"Cannot stop instance {self.name}: {result.stderr.strip()}")

    def logs(self, stdout: bool = True, stderr: bool = True) -> bytes:
        """Return the served Tesseract's captured logs.

        Serve runs via ``exec`` with stdout+stderr merged into a single captured
        log file (see serve()), so stdout/stderr cannot be separated here; either
        flag returns the merged log. Falls back to Apptainer's own instance log
        files for anything the instance process itself emitted.
        """
        if not stdout and not stderr:
            raise ValueError("At least one of stdout or stderr must be True.")
        chunks: list[bytes] = []
        log_path = _serve_log_path(self.name)
        if log_path.is_file():
            chunks.append(log_path.read_bytes())
        else:
            # Fall back to Apptainer's own instance logs.
            out_path, err_path = _instance_log_paths(self.name)
            if out_path and Path(out_path).is_file():
                chunks.append(Path(out_path).read_bytes())
            if err_path and Path(err_path).is_file():
                chunks.append(Path(err_path).read_bytes())
        return b"".join(chunks)

    def wait(self) -> dict:
        """Wait is a no-op for detached instances; report success."""
        return {"StatusCode": 0}

    def remove(self, v: bool = False, link: bool = False, force: bool = False) -> str:
        """Stop the instance (Apptainer has no separate remove step)."""
        self.stop()
        _remove_sidecar(self.name)
        return self.name


# --------------------------------------------------------------------------- #
# Instance helpers                                                             #
# --------------------------------------------------------------------------- #


def _list_instances() -> list[dict]:
    """Return the current user's Apptainer instances as dicts."""
    apptainer = _get_apptainer_executable()
    result = subprocess.run(
        [*apptainer, "instance", "list", "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise APIError(f"Cannot list Apptainer instances: {result.stderr.strip()}")
    data = json.loads(result.stdout or "{}")
    return data.get("instances", []) or []


def _instance_exists(name: str) -> bool:
    """Check whether an instance with the given name is running."""
    try:
        return any(inst.get("instance") == name for inst in _list_instances())
    except APIError:
        return False


def _instance_log_paths(name: str) -> tuple[str | None, str | None]:
    """Return (stdout_log, stderr_log) paths for an instance, if known."""
    for inst in _list_instances():
        if inst.get("instance") == name:
            return inst.get("logOutPath"), inst.get("logErrPath")
    return None, None


# --------------------------------------------------------------------------- #
# Sidecar metadata (per-served-instance state Apptainer does not track)        #
# --------------------------------------------------------------------------- #


def _instances_meta_dir() -> Path:
    """Directory holding per-instance sidecar metadata files."""
    d = get_image_dir().parent / "instances"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_sidecar(name: str, meta: dict) -> None:
    """Persist per-instance metadata (port, sif path, tesseract name, ...)."""
    (_instances_meta_dir() / f"{name}.json").write_text(json.dumps(meta))


def _read_sidecar(name: str) -> dict:
    """Read per-instance metadata, returning an empty dict if absent."""
    path = _instances_meta_dir() / f"{name}.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _remove_sidecar(name: str) -> None:
    """Delete per-instance metadata and the captured serve log if present."""
    path = _instances_meta_dir() / f"{name}.json"
    if path.is_file():
        path.unlink()
    log_path = _serve_log_path(name)
    if log_path.is_file():
        log_path.unlink()


def _serve_log_path(name: str) -> Path:
    """Path to the captured stdout/stderr of an instance's serve process.

    Serve runs as a separate ``apptainer exec`` rather than the instance's own
    process, so its output is not written to Apptainer's own instance log files;
    we capture it here so ``logs()`` can read it back.
    """
    return _instances_meta_dir() / f"{name}.log"


# --------------------------------------------------------------------------- #
# Containers namespace                                                         #
# --------------------------------------------------------------------------- #


def _resolve_run_target(image: str) -> Path:
    """Resolve a run target (store ref or .sif path) to a SIF path."""
    candidate = Path(image).expanduser()
    if candidate.suffix == ".sif" and candidate.is_file():
        return candidate.resolve()
    name, tag = _split_ref(image)
    sif_path = _sif_path(name, tag)
    if not sif_path.is_file():
        raise ImageNotFound(
            f"Image {image} not found in the Apptainer image store "
            f"({get_image_dir()}). Pull or build it first, or pass a path to a "
            ".sif file."
        )
    return sif_path


def _translate_run_flags(
    volumes: dict | None,
    environment: dict[str, str] | None,
    device_requests: list_[int | str] | None,  # noqa: UP006
    memory: str | None,
    capabilities: Capabilities,
) -> tuple[list[str], dict[str, str]]:
    """Translate Docker-style run options into apptainer flags.

    Returns (flags, extra_env) where extra_env holds variables that must be passed
    through the environment rather than as flags (e.g. CUDA_VISIBLE_DEVICES).
    """
    flags: list[str] = list(ISOLATION_FLAGS)
    extra_env: dict[str, str] = {}

    if volumes:
        for host_path, info in volumes.items():
            mode = info.get("mode", "rw")
            flags.extend(["--bind", f"{host_path}:{info['bind']}:{mode}"])

    if environment:
        for key, value in environment.items():
            flags.extend(["--env", f"{key}={value}"])

    if device_requests:
        # --nv exposes all GPUs; select specific ones via CUDA_VISIBLE_DEVICES.
        flags.append("--nv")
        gpu_ids = [str(g) for g in device_requests]
        if gpu_ids and gpu_ids != ["all"]:
            extra_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    if memory:
        # Memory limits require cgroups v2 delegation; warn but continue if the
        # runtime later rejects it (Apptainer applies it best-effort).
        flags.extend(["--memory", memory])
        logger.debug(
            "Applying --memory %s; requires cgroups v2 delegation to take effect.",
            memory,
        )

    return flags, extra_env


class Containers:
    """Namespace for served Tesseract instances backed by Apptainer."""

    @staticmethod
    def list(all: bool = False, tesseract_only: bool = True) -> list_[Container]:  # noqa: UP006
        """List running Tesseract instances.

        ``all`` is accepted for interface parity but has no effect: Apptainer only
        tracks running instances, so stopped ones cannot be listed.
        """
        containers = []
        for inst in _list_instances():
            name = inst.get("instance", "")
            sidecar = _read_sidecar(name)
            if tesseract_only and not sidecar.get("tesseract_name"):
                # Not a Tesseract-managed instance (no sidecar we wrote).
                continue
            containers.append(Container.from_instance(inst, sidecar))
        return containers

    @staticmethod
    def get(id_or_name: str, tesseract_only: bool = True) -> Container:
        """Return the served Tesseract instance with the given name."""
        for inst in _list_instances():
            if inst.get("instance") == id_or_name:
                sidecar = _read_sidecar(id_or_name)
                if tesseract_only and not sidecar.get("tesseract_name"):
                    raise NotFound(
                        f"Instance {id_or_name} is not a Tesseract instance."
                    )
                return Container.from_instance(inst, sidecar)
        raise NotFound(f"Instance {id_or_name} not found.")

    @staticmethod
    def run(
        image: str,
        command: list_[str],  # noqa: UP006
        volumes: dict | None = None,
        device_requests: list_[int | str] | None = None,  # noqa: UP006
        environment: dict[str, str] | None = None,
        network: str | None = None,
        detach: bool = False,
        remove: bool = False,
        ports: dict | None = None,
        stdout: bool = True,
        stderr: bool = False,
        user: str | None = None,
        memory: str | None = None,
        extra_args: list_[str] | None = None,  # noqa: UP006
        stream_stdout: BoolOrCallable = False,
        stream_stderr: BoolOrCallable = False,
    ) -> Container | tuple[bytes, bytes] | bytes:
        """Run a command in a container from a SIF image.

        Only the one-shot (non-detached) path is implemented here; serving
        (``detach=True``) is handled by the engine via the instance API (phase 4).
        ``user`` is ignored (Apptainer runs as the invoking user natively).
        ``network``/``ports`` are unsupported and rejected per capability flags.
        """
        capabilities = ApptainerClient.capabilities

        if detach:
            raise NotImplementedError(
                "Detached runs are served via the Apptainer instance API; use the "
                "engine serve path rather than containers.run(detach=True)."
            )
        if network is not None and network != "host":
            raise NotFound(
                "The Apptainer backend does not support user-defined networks. "
                "Apptainer is host-network only."
            )
        if ports:
            raise NotFound(
                "The Apptainer backend does not support port mapping. Apptainer is "
                "host-network only; the runtime binds host ports directly."
            )
        if user is not None:
            logger.debug(
                "Ignoring user=%s: Apptainer runs as the invoking user natively.",
                user,
            )

        if isinstance(command, str):
            command = [command]

        sif_path = _resolve_run_target(image)
        flags, extra_env = _translate_run_flags(
            volumes, environment, device_requests, memory, capabilities
        )
        if extra_env:
            for key, value in extra_env.items():
                flags.extend(["--env", f"{key}={value}"])

        apptainer = _get_apptainer_executable()
        full_cmd = [
            *apptainer,
            "run",
            *flags,
            *(extra_args or []),
            str(sif_path),
            *command,
        ]

        returncode, stdout_data, stderr_data = _run_process(
            full_cmd,
            stream_stdout=_get_io_callable(stream_stdout, sys.stdout.write),
            stream_stderr=_get_io_callable(stream_stderr, sys.stderr.write),
        )

        if returncode != 0:
            raise ContainerError(
                None,
                returncode,
                shlex.join(full_cmd),
                image,
                stderr_data,
            )

        if stdout and stderr:
            return stdout_data, stderr_data
        if stderr:
            return stderr_data
        return stdout_data

    @staticmethod
    def serve(
        image: str,
        args: list_[str],  # noqa: UP006
        volumes: dict | None = None,
        device_requests: list_[int | str] | None = None,  # noqa: UP006
        environment: dict[str, str] | None = None,
        memory: str | None = None,
        debug: bool = False,
        debugpy_host_port: str | None = None,
    ) -> Container:
        """Serve a Tesseract as a long-running Apptainer instance.

        Implements the pattern validated in phase 0: start a bare instance (to
        create the persistent namespace), then ``exec instance:// entrypoint serve``
        detached. ``instance start`` alone does not run the OCI entrypoint of a
        converted image, so serve is driven through ``exec`` — with the entrypoint
        invoked explicitly so nss_wrapper's passwd/group files are seeded.

        ``args`` already contains the resolved ``--port/--host/--num-workers`` flags
        built by the engine. The instance binds the host port directly (Apptainer is
        host-network only).

        Returns a Container backed by the instance, with a sidecar recording the
        port so ``ps``/``logs``/``teardown`` can find it later.
        """
        sif_path = _resolve_run_target(image)
        capabilities = ApptainerClient.capabilities
        flags, extra_env = _translate_run_flags(
            volumes, environment, device_requests, memory, capabilities
        )
        for key, value in extra_env.items():
            flags.extend(["--env", f"{key}={value}"])

        # Parse the port the runtime was told to serve on out of args so we can
        # record it in the sidecar (host port == container port under host network).
        port = None
        for i, tok in enumerate(args):
            if tok == "--port" and i + 1 < len(args):
                port = args[i + 1]
                break

        instance_name = f"tesseract-{uuid.uuid4().hex[:12]}"
        apptainer = _get_apptainer_executable()

        # 1. Start the (bare) instance to create the namespace.
        start_cmd = [
            *apptainer,
            "instance",
            "start",
            *flags,
            str(sif_path),
            instance_name,
        ]
        start = subprocess.run(start_cmd, capture_output=True, text=True)
        if start.returncode != 0:
            raise ContainerError(
                instance_name,
                start.returncode,
                shlex.join(start_cmd),
                str(sif_path),
                start.stderr.encode(),
            )

        # 2. Persist sidecar metadata before exec so ps/teardown can find it even
        #    if the exec is still coming up.
        sidecar = {
            "tesseract_name": _sif_environment(sif_path).get("TESSERACT_NAME", ""),
            "sif_path": str(sif_path),
            "image_ref": f"{sif_path.parent.name}:{sif_path.stem}",
            "port": port,
            "host_ip": "127.0.0.1",
            "debugpy_port": debugpy_host_port,
        }
        _write_sidecar(instance_name, sidecar)

        # 3. Run the entrypoint + serve inside the instance, detached. The
        #    entrypoint is invoked explicitly (exec bypasses the runscript) so
        #    nss_wrapper is seeded. Because serve runs as a separate exec (not the
        #    instance's own process), Apptainer does not capture its output to the
        #    instance log files, so we redirect it to a log file we own and read
        #    back in Container.logs().
        exec_cmd = [
            *apptainer,
            "exec",
            f"instance://{instance_name}",
            *RUNTIME_ENTRYPOINT,
            "serve",
            *args,
        ]
        log_path = _serve_log_path(instance_name)
        # Launch serve fully detached: an intermediate shell backgrounds the exec
        # and exits immediately, so the serve process is reparented to init rather
        # than tracked as a child of this Python process. That avoids leaving an
        # unreaped zombie once the instance (and thus the serve process) is stopped.
        # Its output is redirected to a log file we own for Container.logs().
        launcher = f"exec {shlex.join(exec_cmd)} > {shlex.quote(str(log_path))} 2>&1 &"
        subprocess.run(
            ["sh", "-c", launcher],
            check=True,
            start_new_session=True,
        )

        return Container.from_instance({"instance": instance_name}, sidecar)


# --------------------------------------------------------------------------- #
# Volumes / Networks (unsupported — present for protocol parity)               #
# --------------------------------------------------------------------------- #


class Volumes:
    """Apptainer has no managed volumes; binds are plain host paths."""

    @staticmethod
    def create(name: str) -> Any:
        """Volumes are unsupported; bind host directories directly instead."""
        raise NotFound(
            "The Apptainer backend does not support named volumes. Bind a host "
            "directory with --volume host:container instead."
        )

    @staticmethod
    def get(name: str) -> Any:
        """Volumes are unsupported."""
        raise NotFound("The Apptainer backend does not support named volumes.")

    @staticmethod
    def list() -> list[str]:
        """No managed volumes exist."""
        return []


class Networks:
    """Apptainer is host-network only; user-defined networks are unsupported."""

    @staticmethod
    def get(name: str) -> dict:
        """Networks are unsupported."""
        raise NotFound("The Apptainer backend does not support user-defined networks.")

    @staticmethod
    def create(name: str) -> dict:
        """Networks are unsupported."""
        raise NotFound("The Apptainer backend does not support user-defined networks.")


# --------------------------------------------------------------------------- #
# Client                                                                       #
# --------------------------------------------------------------------------- #


class ApptainerClient:
    """Apptainer implementation of the ContainerClient protocol.

    Manages an on-disk SIF store, runs one-shot commands via ``apptainer run``, and
    (via the engine) serves Tesseracts as Apptainer instances. Only sees
    Tesseract-relevant images and instances unless ``tesseract_only=False``.
    """

    capabilities = Capabilities(
        supports_networks=False,
        supports_port_mapping=False,
        supports_restart_policy=False,
        supports_build=False,
        needs_user_mapping=False,
        is_podman=False,
        name="apptainer",
    )

    def __init__(self) -> None:
        self.containers = Containers()
        self.images = Images()
        self.volumes = Volumes()
        self.networks = Networks()

    @staticmethod
    def info() -> tuple:
        """Return ``apptainer --version`` output, raising if unavailable."""
        apptainer = _get_apptainer_executable()
        try:
            result = subprocess.run(
                [*apptainer, "--version"],
                check=True,
                capture_output=True,
            )
            return result.stdout, result.stderr
        except subprocess.CalledProcessError as ex:
            raise APIError("Could not reach Apptainer.") from ex
