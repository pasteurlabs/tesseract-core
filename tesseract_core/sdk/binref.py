# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Binref array encoding for the HTTP client.

The ``json+binref`` output format passes array buffers as files in a directory
that is mounted into (and read directly by) the Tesseract container, rather than
base64-encoded inside the HTTP body. On a native Linux host serving Linux
containers -- where client and server share a page cache via a shared-memory
tmpfs -- this becomes a fast, near-zero-copy same-machine IPC path.

This module holds the client-side machinery for that path:

- :func:`encode_array_binref` writes an array to a fresh ``.bin`` file per request.
- :class:`BinrefWritePool` / :func:`encode_array_binref_pooled` reuse warm,
  pre-faulted, memory-mapped buffers to avoid the per-request page-fault cost.
- :func:`read_binref_array` / :func:`mmap_binref_array` decode a binref buffer
  eagerly (portable) or as a zero-copy mmap view (POSIX only).

The pool and the lazy mmap decode are gated on Linux via
:data:`SUPPORTS_BINREF_POOL`; see the note there for why other platforms do not
qualify.
"""

from __future__ import annotations

import mmap
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np


def _fast_tobytes(arr: np.ndarray) -> memoryview:
    """Convert a numpy array to bytes without copying if possible."""
    return np.ascontiguousarray(arr).data


def encode_array_binref(arr: Any, input_dir: Path, written_files: list[Path]) -> dict:
    """Encode an array as a binref reference, writing its buffer to ``input_dir``.

    The array is written to a uniquely named ``.bin`` file directly under
    ``input_dir``, which must be mounted into the Tesseract container as its
    input path. The returned reference is relative to that directory, so the
    server resolves it against its configured input path. Written files are
    appended to ``written_files`` so the caller can clean them up afterwards.
    """
    arr = np.asanyarray(arr, order="A")
    filename = f"{uuid.uuid4()}.bin"
    target = input_dir / filename
    target.write_bytes(_fast_tobytes(arr))
    written_files.append(target)
    return {
        "shape": arr.shape,
        "dtype": arr.dtype.name,
        "data": {
            "buffer": f"{filename}:0",
            "encoding": "binref",
        },
    }


class BinrefSlot:
    """A reusable, pre-faulted, file-backed buffer for binref inputs.

    The slot owns a ``.bin`` file under the mounted input directory, kept open
    and memory-mapped read-write. Writing an array copies it into the mapping at
    memory-copy bandwidth (the pages stay resident between uses), avoiding the
    page-fault cost of writing a fresh file each request.
    """

    def __init__(self, input_dir: Path, capacity: int) -> None:
        self.filename = f"pool_{uuid.uuid4()}.bin"
        self.path = input_dir / self.filename
        self.capacity = 0
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        self._mm: mmap.mmap | None = None
        self._grow(capacity)

    def _grow(self, capacity: int) -> None:
        if self._mm is not None:
            self._mm.close()
        os.ftruncate(self._fd, capacity)
        self._mm = mmap.mmap(self._fd, capacity)
        # Pre-fault the pages so the first real write runs at memcpy speed.
        self._mm[:] = b"\x00" * capacity
        self.capacity = capacity

    def write(self, data: memoryview) -> None:
        """Copy ``data`` into the slot's mapping, growing it if needed."""
        n = data.nbytes
        if n > self.capacity:
            self._grow(n)
        self._mm[:n] = data

    def close(self) -> None:
        """Unmap the buffer, close the file descriptor, and delete the file."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        os.close(self._fd)
        self.path.unlink(missing_ok=True)


class BinrefWritePool:
    """Opt-in pool of reusable warm buffers for binref input encoding.

    A pool of :class:`BinrefSlot` objects, each a mounted ``.bin`` file kept
    memory-mapped and pre-faulted. Checking out a slot and copying into its warm
    mapping runs at memory-copy bandwidth, versus writing a fresh file per
    request which pays a page fault on every page.

    Bounded by ``max_slots`` and ``max_bytes``: a checkout that cannot be served
    from the pool (all slots busy, or granting it would exceed the byte cap)
    returns ``None`` so the caller falls back to a plain fresh-file write. The
    pool is safe for concurrent requests: each in-flight request holds its own
    slot until it returns the slot after the server has read it.
    """

    def __init__(
        self, input_dir: Path, max_slots: int = 4, max_bytes: int | None = None
    ) -> None:
        self._input_dir = input_dir
        self._max_slots = max_slots
        self._max_bytes = max_bytes
        self._free: list[BinrefSlot] = []
        self._all: list[BinrefSlot] = []
        self._lock = threading.Lock()

    def _total_bytes(self) -> int:
        return sum(s.capacity for s in self._all)

    def checkout(self, nbytes: int) -> BinrefSlot | None:
        """Return a warm slot for ``nbytes``, or ``None`` to signal fallback."""
        with self._lock:
            # Prefer a free slot that already fits, to avoid a re-map/re-fault.
            fitting = [s for s in self._free if s.capacity >= nbytes]
            if fitting:
                slot = min(fitting, key=lambda s: s.capacity)
                self._free.remove(slot)
                return slot
            # Reuse a free (smaller) slot by growing it, if still within caps.
            if self._free:
                slot = self._free.pop()
                extra = nbytes - slot.capacity
                if (
                    self._max_bytes is None
                    or self._total_bytes() + extra <= self._max_bytes
                ):
                    # write() grows the slot's mapping to fit before copying.
                    return slot
                # Growing would exceed the cap: keep it free and fall back.
                self._free.append(slot)
                return None
            # Allocate a new slot if we are under both caps.
            if len(self._all) >= self._max_slots:
                return None
            if (
                self._max_bytes is not None
                and self._total_bytes() + nbytes > self._max_bytes
            ):
                return None
            slot = BinrefSlot(self._input_dir, nbytes)
            self._all.append(slot)
            return slot

    def checkin(self, slot: BinrefSlot) -> None:
        """Return a checked-out slot to the pool for reuse."""
        with self._lock:
            self._free.append(slot)

    def close(self) -> None:
        """Close and delete every slot's backing file, emptying the pool."""
        with self._lock:
            for slot in self._all:
                slot.close()
            self._all.clear()
            self._free.clear()


def encode_array_binref_pooled(
    arr: Any, pool: BinrefWritePool, checked_out: list, written_files: list[Path]
) -> dict:
    """Encode an array as binref using a warm pool slot, or fall back to a file.

    On a pool hit, copies the array into a checked-out warm slot and records it
    in ``checked_out`` for return after the request. On a miss (pool exhausted
    or over cap), falls back to :func:`encode_array_binref`.
    """
    arr = np.asanyarray(arr, order="A")
    data = _fast_tobytes(arr)
    slot = pool.checkout(data.nbytes)
    if slot is None:
        return encode_array_binref(arr, pool._input_dir, written_files)
    slot.write(data)
    checked_out.append(slot)
    return {
        "shape": arr.shape,
        "dtype": arr.dtype.name,
        "data": {
            "buffer": f"{slot.filename}:0",
            "encoding": "binref",
        },
    }


# The binref write pool is a fast same-machine IPC path: the client writes binref
# inputs into a directory the server container reads directly, ideally a shared-
# memory tmpfs (/dev/shm), and decodes outputs as zero-copy mmap views. That only
# pays off when client and server share a page cache, i.e. a native Linux host
# running Linux containers. On macOS (and Windows) the container runs inside a
# Linux VM, so bind mounts cross the VM boundary and there is no host tmpfs the
# container sees as the same memory -- the premise does not hold. So the pool
# (and, with it, the lazy zero-copy decode) is offered on Linux only.
SUPPORTS_BINREF_POOL = sys.platform.startswith("linux")


def read_binref_array(
    full_path: Path, offset: int, num_bytes: int, dtype: np.dtype, count: int
) -> np.ndarray:
    """Read an uncompressed binref buffer into an owned, writable array.

    Copies the buffer into a freshly allocated array via ``readinto``. The array
    owns its data and does not depend on the file afterwards, so the file may be
    removed or recycled once this returns. Portable across platforms; the
    default decode path.
    """
    out = np.empty(count, dtype=dtype)
    with open(full_path, "rb") as f:
        if offset:
            f.seek(offset)
        f.readinto(memoryview(out).cast("B"))
    return out


def mmap_binref_array(
    full_path: Path, offset: int, num_bytes: int, dtype: np.dtype, count: int
) -> np.ndarray:
    """Map an uncompressed binref buffer as a read-only array without copying.

    The returned array is a view over a memory map of the file, so decoding does
    not copy the buffer and pages fault in lazily as the array is read. numpy
    keeps the map alive via the array's ``base``, so the data stays valid even
    if the file is later removed (the OS keeps the mapped inode alive). The
    array is read-only.

    Relies on POSIX mmap semantics (a read-only ``PROT_READ`` mapping, and a
    mapped inode staying valid after the file is removed). The mapped file must
    not be overwritten while a returned view is still in use.
    """
    map_len = offset + num_bytes
    fd = os.open(full_path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, map_len, prot=mmap.PROT_READ)
    finally:
        os.close(fd)
    return np.frombuffer(mm, dtype=dtype, count=count, offset=offset)
