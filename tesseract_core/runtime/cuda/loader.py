# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery and loading of the CUDA runtime and driver shared libraries.

This module owns the messy part of talking to CUDA through ctypes: finding
``libcudart`` / ``libcuda`` on disk (system toolkit, pip CUDA wheels, or bare
sonames) and returning a loaded, signature-declared ``ctypes.CDLL`` for each.

Everything ctypes here stays here. The only ctypes objects that escape are the
two ``CDLL`` handles, consumed exclusively by
:mod:`tesseract_core.runtime.cuda.api`, which wraps them in a plain-Python
API. The one exception is :func:`iter_cudart_candidates`, a *string*-only
discovery surface (soname/path list) exported for out-of-process consumers.
"""

import ctypes
import ctypes.util
import importlib.util
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any

# CUDA runtime major versions we probe for by number, newest first. cuda_ipc
# only calls a handful of long-stable runtime symbols (cudaIpc*, cudaMemcpy,
# cudaMalloc, cudaSetDevice, cudaDeviceSynchronize), so a newer major than any
# listed here is very likely to work -- the range is generous and open-ended at
# the top precisely so a freshly released CUDA does not need a code change. The
# floor is the oldest major whose ABI we still expect to encounter in the wild.
CUDART_MAJOR_NEWEST = 20
CUDART_MAJOR_OLDEST = 11
_CUDART_MAJORS = tuple(range(CUDART_MAJOR_NEWEST, CUDART_MAJOR_OLDEST - 1, -1))


def _cudart_sonames() -> tuple[str, ...]:
    """Candidate CUDA runtime library filenames, most-preferred first.

    Unversioned names come first: on a system with a CUDA toolkit the loader
    resolves ``libcudart.so`` / ``libcudart.dylib`` via the dev symlink to
    whatever major is installed, so we never have to know the number. The
    versioned names that follow are generated over :data:`_CUDART_MAJORS`
    (newest first) rather than hand-enumerated, so a new CUDA major is picked up
    without editing this file.
    """
    names = ["libcudart.so", "libcudart.dylib"]
    names += [f"libcudart.so.{major}" for major in _CUDART_MAJORS]
    names += [f"cudart64_{major}.dll" for major in _CUDART_MAJORS]
    return tuple(names)


# Glob patterns for locating a wheel-shipped runtime by filename in a known lib
# directory. Unlike the loader-name list above, here we have a concrete
# directory to scan, so we can match *any* version present rather than probe a
# fixed set -- fully forward-compatible for the wheel case.
_CUDART_GLOBS = ("libcudart.so.*", "libcudart.so", "libcudart.dylib", "cudart64_*.dll")


def _cudart_soname_sort_key(path: Path) -> tuple[int, int]:
    """Sort key placing higher CUDA majors first among wheel candidates.

    Returns ``(0 if versioned else 1, -major)`` so ``sorted`` yields versioned
    names first and, among them, newest-major-first. The major is parsed from
    ``libcudart.so.<major>`` / ``cudart64_<major>.dll``; names without a
    parseable version (e.g. an unversioned ``libcudart.so`` symlink) sort after
    all versioned ones so a concrete version wins.
    """
    name = path.name
    major = -1
    if name.startswith("libcudart.so."):
        tail = name[len("libcudart.so.") :].split(".", 1)[0]
        if tail.isdigit():
            major = int(tail)
    elif name.startswith("cudart64_") and name.endswith(".dll"):
        tail = name[len("cudart64_") : -len(".dll")]
        if tail.isdigit():
            major = int(tail)
    # Versioned first (-major ascending == major descending); unversioned last.
    return (0 if major >= 0 else 1, -major)


def _iter_wheel_cudart_paths() -> Iterator[str]:
    """Yield candidate absolute paths to libcudart shipped inside pip wheels.

    The pip CUDA wheels (``nvidia-cuda-runtime-cuXX``, pulled in transitively by
    ``jax[cudaXX]`` / ``cupy-cudaXXx``) install the runtime under a
    ``site-packages/nvidia/<pkg>/lib/libcudart.so.NN`` directory (``<pkg>`` is
    typically ``cuda_runtime``). That directory is on neither
    ``LD_LIBRARY_PATH`` nor the ``ldconfig`` cache, so both
    :func:`ctypes.util.find_library` and a bare ``ctypes.CDLL(soname)`` miss it
    -- observed on GPU CI runners with no system CUDA toolkit installed. We
    locate the wheel directory ourselves so this venv-local runtime is found
    (and, per :func:`iter_cudart_candidates`, preferred over a system one).
    """

    def _spec_locations(name: str) -> Iterable[str]:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            return ()
        return () if spec is None else (spec.submodule_search_locations or ())

    lib_dirs: list[Path] = []
    seen: set[str] = set()

    def _add(lib_dir: Path) -> None:
        key = str(lib_dir)
        if key not in seen:
            seen.add(key)
            lib_dirs.append(lib_dir)

    # Preferred: ask importlib where the runtime wheel's package lives, so we do
    # not hardcode the site-packages layout.
    for location in _spec_locations("nvidia.cuda_runtime"):
        _add(Path(location) / "lib")

    # Fallback: glob every nvidia namespace package on the import path, in case
    # the runtime is bundled under a differently named package. Anchoring on the
    # nvidia namespace itself (rather than a fixed depth off __file__) keeps this
    # correct for editable installs and non-standard layouts.
    for location in _spec_locations("nvidia"):
        for lib in Path(location).glob("*/lib"):
            _add(lib)

    for lib_dir in lib_dirs:
        candidates: set[Path] = set()
        for pattern in _CUDART_GLOBS:
            candidates.update(p for p in lib_dir.glob(pattern) if p.is_file())
        # Newest major first, so a wheel dir that somehow holds several runtimes
        # (or a dev symlink alongside a versioned .so) prefers the highest one.
        for candidate in sorted(candidates, key=_cudart_soname_sort_key):
            yield str(candidate)


def iter_cudart_candidates() -> Iterator[str]:
    """Yield libcudart names/paths to try loading, most-preferred first.

    Each item is an argument suitable for ``ctypes.CDLL`` **and** for a raw
    ``dlopen``/``LoadLibrary``: either a bare soname the system loader resolves
    (e.g. ``"libcudart.so.12"``) or an absolute path to a wheel-shipped runtime.
    The order encodes the search strategy:

    1. absolute paths to pip-wheel CUDA installs (see
       :func:`_iter_wheel_cudart_paths`), so a venv's runtime wins over a system
       one -- this matches how JAX and PyTorch load libcudart (their loaders
       ``dlopen`` the wheel copy by absolute path first, falling back to the
       system library only if no wheel is present). Agreeing with them on which
       runtime is loaded matters for a codec that hands device memory to them;
    2. ``ctypes.util.find_library`` results (search the ``ldconfig`` cache, and
       on non-glibc platforms other loader paths), unversioned name first then
       generated per-major Windows stems;
    3. bare sonames, for systems where ``find_library`` misses the versioned name
       but the loader can still resolve it (e.g. an already-loaded copy, or via
       ``LD_LIBRARY_PATH``, which ``dlopen`` honours but ``find_library`` does
       not).

    This is the public discovery surface: non-Python consumers (e.g. the
    ``tesseract_jax`` C++ FFI shim, which ``dlopen``s libcudart itself) can use
    it to locate the same runtime this module loads, so both agree on the wheel
    preference and stay forward-compatible with new CUDA majors without their own
    hardcoded soname list. Discovery only -- the caller does the actual load.

    Items are de-duplicated preserving order; existence is not guaranteed (a
    candidate may still fail to load), so callers should try each in turn.
    """
    # find_library wants a stem (not a full soname): the unversioned name first,
    # then generated per-major Windows stems.
    find_library_stems = ("cudart", *(f"cudart64_{major}" for major in _CUDART_MAJORS))
    from_find_library = (ctypes.util.find_library(stem) for stem in find_library_stems)
    candidates = chain(
        _iter_wheel_cudart_paths(),
        filter(None, from_find_library),
        _cudart_sonames(),
    )

    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


# cudaIpcMemHandle_t is an opaque 64-byte blob; the runtime layer needs the size
# to declare the handle struct, so it lives with the ABI details here.
CUDA_IPC_HANDLE_SIZE = 64


class CudaIpcMemHandle(ctypes.Structure):
    """ctypes mirror of ``cudaIpcMemHandle_t`` (an opaque 64-byte blob).

    This *must* be a Structure (not a bare ``c_byte`` array) so that ctypes
    passes it **by value** to the CUDA runtime, matching the C ABI where
    ``cudaIpcMemHandle_t`` is a struct argument. Passing a ``c_byte`` array
    would be marshalled as a pointer, which makes ``cudaIpcOpenMemHandle``
    fail with ``cudaErrorInvalidValue`` (error code 1).
    """

    _fields_ = [("reserved", ctypes.c_byte * CUDA_IPC_HANDLE_SIZE)]


def _find_cudart() -> Any:
    """Locate and load libcudart (a ``ctypes.CDLL``), or ``None`` if not found.

    Tries each candidate from :func:`iter_cudart_candidates` in order and
    returns the first that loads.
    """
    for candidate in iter_cudart_candidates():
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    return None


def load_cudart() -> Any:
    """Load the CUDA runtime library and declare the signatures we call.

    Returns a ``ctypes.CDLL`` with argument/return types declared so ctypes
    marshals 64-bit pointers and the 64-byte handle struct correctly (the
    defaults assume C int, which truncates pointers and passes structs by
    reference). Raises ``RuntimeError`` if libcudart cannot be found.
    """
    cudart = _find_cudart()
    if cudart is None:
        raise RuntimeError(
            "Could not find CUDA runtime library (libcudart). Make sure CUDA is "
            "installed and on the loader path (set LD_LIBRARY_PATH), or install a "
            "CUDA runtime wheel (e.g. nvidia-cuda-runtime-cu13)."
        )

    cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    cudart.cudaSetDevice.restype = ctypes.c_int
    cudart.cudaIpcGetMemHandle.argtypes = [
        ctypes.POINTER(CudaIpcMemHandle),
        ctypes.c_void_p,
    ]
    cudart.cudaIpcGetMemHandle.restype = ctypes.c_int
    # NOTE: the handle is the second argument *by value* (a struct), not a
    # pointer. This is the crux of getting IPC to work through ctypes.
    cudart.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        CudaIpcMemHandle,
        ctypes.c_uint,
    ]
    cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int
    cudart.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    cudart.cudaIpcCloseMemHandle.restype = ctypes.c_int
    cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
    cudart.cudaGetErrorString.restype = ctypes.c_char_p
    # Used to drain the runtime API's sticky last-error after an expected failure
    # (see the ``_check`` helper in the api module).
    cudart.cudaGetLastError.argtypes = []
    cudart.cudaGetLastError.restype = ctypes.c_int
    # Used by the VMM staging-buffer fallback (see api.stage_for_legacy_ipc).
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int
    cudart.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    cudart.cudaMemcpy.restype = ctypes.c_int
    # Used by decode to block until a device-to-device copy completes before the
    # IPC mapping is closed.
    cudart.cudaDeviceSynchronize.argtypes = []
    cudart.cudaDeviceSynchronize.restype = ctypes.c_int
    return cudart


def load_cuda_driver() -> Any:
    """Load the CUDA driver library (libcuda) and declare the signatures we call.

    The driver API is only needed for ``cuMemGetAddressRange``, which recovers
    the base pointer and size of the allocation backing a device pointer. This
    is required because IPC handles reference the *whole* allocation, while a
    given array may point partway into it (common with pooled allocators like
    CuPy and PyTorch). Raises ``RuntimeError`` if libcuda cannot be found.
    """
    driver = None
    path = ctypes.util.find_library("cuda")
    if path:
        driver = ctypes.CDLL(path)
    if driver is None:
        for name in ("libcuda.so", "libcuda.so.1", "nvcuda.dll"):
            try:
                driver = ctypes.CDLL(name)
                break
            except OSError:
                continue
    if driver is None:
        raise RuntimeError(
            "Could not find CUDA driver library (libcuda). "
            "Make sure an NVIDIA driver is installed."
        )

    # CUdeviceptr is an unsigned integer the width of a pointer.
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuMemGetAddressRange_v2.argtypes = [
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_ulonglong,
    ]
    driver.cuMemGetAddressRange_v2.restype = ctypes.c_int
    driver.cuInit(0)
    return driver
