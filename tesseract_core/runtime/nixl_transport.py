# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL array encoding: point-to-point GPU array exchange via NIXL.

This module holds everything specific to the experimental ``json+nixl``
encoding. Like :mod:`tesseract_core.runtime.cuda_ipc` it passes a GPU array
between a producer and a consumer process without a host round-trip, but where
``cuda_ipc`` is same-host only (a legacy CUDA IPC handle), NIXL is a
point-to-point transfer library that auto-selects its backend -- CUDA IPC /
shared memory same-host, UCX (RDMA where the hardware supports it, TCP
otherwise) across hosts -- behind one API. Nothing here is imported unless a
Tesseract actually encodes or decodes a NIXL array.

The transfer is **initiator-driven READ**: the producer (server) registers the
array's device memory with its NIXL agent and publishes, in the JSON response,
its agent metadata plus the array's serialized transfer descriptor. The consumer
(client) adds the producer as a remote agent, allocates its own buffer,
registers it, and posts a matched READ that pulls the bytes across. This is the
same receiver-driven shape as ``cuda_ipc`` -- the producer's registration is
inert until the consumer reads -- so it slots into the same request lifecycle.

Public entry points used via the :class:`NixlTransport` backend:

* :func:`dump_nixl_arraydict` / :func:`load_nixl_arraydict` -- encode/decode,
* :func:`release_nixl_exports` -- deregister a request's exported buffers.

The consumer-facing result is the framework-agnostic
:class:`tesseract_core.runtime.cuda_ipc.IpcDeviceArray`, reused unchanged: it
already owns a plain device buffer and exposes ``__cuda_array_interface__`` /
``__dlpack__``, so Torch/JAX/CuPy adopt it zero-copy exactly as for ``cuda_ipc``.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any

import numpy as np
import pybase64

from tesseract_core.runtime.array_encoding import ArrayDict
from tesseract_core.runtime.cuda_ipc import (
    IpcDeviceArray,
    _cuda_error_string,
    _get_cudart,
    _is_c_contiguous,
    has_cuda_array_interface,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


# ---------------------------------------------------------------------------
# Lazy NIXL agent (one per process, per role)
# ---------------------------------------------------------------------------
#
# A NIXL agent owns the backend (UCX) and the registration table. Both the
# producer (server) and the consumer (client) need one; a single process-global
# agent per process is enough because a Tesseract serves requests serially (the
# same assumption cuda_ipc's export registry already relies on).

_NIXL_AGENT: Any = None
_CUDA_RUNTIME_PRELOADED = False


def _nixl_agent_name() -> str:
    """A NIXL agent name unique to this process.

    NIXL rejects adding a remote agent whose name equals the local agent's, so
    the producer and consumer processes must not share a name. The pid alone is
    not enough across hosts (pids collide), so mix in a random component.
    """
    import os
    import uuid

    return f"tesseract-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _preload_cuda_runtime() -> None:
    """Load libcudart into the global symbol namespace so UCX can find it.

    NIXL's bundled UCX enables its CUDA transport by ``dlopen``-ing libcudart by
    bare soname. On a host with no system CUDA toolkit -- e.g. a GPU CI runner
    where ``cupy`` / ``jax[cudaXX]`` pull the runtime in as pip wheels -- that
    soname is on neither ``LD_LIBRARY_PATH`` nor the ``ldconfig`` cache, so UCX
    silently fails to load it and reports GPU memory as host memory ("VRAM
    detected as host by UCX"), which makes VRAM registration fail.

    We fix that in-process rather than via a launcher-set ``LD_LIBRARY_PATH``:
    load the same libcudart the cuda_ipc codec resolves (its
    :func:`~tesseract_core.runtime.cuda_ipc.iter_cudart_candidates` prefers the
    wheel copy, matching JAX/PyTorch) with ``RTLD_GLOBAL``, so a subsequent bare
    ``dlopen("libcudart.so.NN")`` from UCX resolves to the already-loaded handle.
    Best-effort and idempotent: if none load, UCX is left to its own resolution
    (a host with system CUDA needs no help), and registration surfaces the error.
    """
    global _CUDA_RUNTIME_PRELOADED
    if _CUDA_RUNTIME_PRELOADED:
        return
    _CUDA_RUNTIME_PRELOADED = True

    from tesseract_core.runtime.cuda_ipc import iter_cudart_candidates

    for candidate in iter_cudart_candidates():
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


def _get_nixl_agent() -> Any:
    """Lazily create this process's NIXL agent, or explain the missing extra."""
    global _NIXL_AGENT
    if _NIXL_AGENT is not None:
        return _NIXL_AGENT
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "The 'json+nixl' encoding requires NIXL. Install it with the "
            "optional extra: pip install tesseract-core[nixl]."
        ) from exc

    # Make libcudart resolvable to UCX's CUDA module before the agent (and its
    # UCX backend) initialise.
    _preload_cuda_runtime()
    _NIXL_AGENT = nixl_agent(_nixl_agent_name(), nixl_agent_config(backends=["UCX"]))
    return _NIXL_AGENT


# ---------------------------------------------------------------------------
# Encode side: register the array and describe the transfer
# ---------------------------------------------------------------------------
#
# Keepalive registry for arrays exported via NIXL by the current request. As
# with cuda_ipc, the exported memory must stay alive (and registered) until the
# consumer has read it, so we retain the source array and its NIXL registration
# handle until release. Bounded to one request's worth of exports per side.

_NIXL_EXPORT_REGISTRY: list[Any] = []


def release_nixl_exports() -> None:
    """Deregister and drop every buffer this side exported via NIXL.

    Mirrors :func:`tesseract_core.runtime.cuda_ipc.release_pinned_ipc_exports`:
    driven by both sides at the same points in the request lifecycle (server at
    the start of the next request; client at the end of the current one), since
    the "consumer is done reading" evidence arrives at the same moments.
    """
    if not _NIXL_EXPORT_REGISTRY:
        return
    agent = _get_nixl_agent()
    for reg_handle, _arr in _NIXL_EXPORT_REGISTRY:
        try:
            agent.deregister_memory(reg_handle)
        except Exception:
            pass
    _NIXL_EXPORT_REGISTRY.clear()


def _nixl_tensor_view(arr: Any) -> Any:
    """Wrap a ``__cuda_array_interface__`` array as something NIXL can register.

    NIXL registers memory by ``(addr, len, device_id)`` tuples. Build that tuple
    straight from the CUDA array interface so no framework object is required.
    """
    iface = arr.__cuda_array_interface__
    data_ptr = iface["data"][0]
    shape = tuple(iface["shape"])
    dtype = np.dtype(iface["typestr"])
    nbytes = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
    device = _device_ordinal(arr)
    return (data_ptr, nbytes, device, "")


def _device_ordinal(arr: Any) -> int:
    """Best-effort CUDA device ordinal for a GPU array (mirrors cuda_ipc)."""
    device = 0
    if hasattr(arr, "device"):
        dev = arr.device
        if hasattr(dev, "id"):
            device = dev.id  # CuPy
        elif hasattr(dev, "index") and dev.index is not None:
            device = dev.index  # PyTorch
    return device


def dump_nixl_arraydict(arr: Any) -> ArrayDict:
    """Dump a CUDA array to a JSON dict describing a NIXL READ transfer.

    Registers the array's device memory with this process's NIXL agent and packs
    the agent metadata plus the array's serialized transfer descriptor into the
    wire field. The source array (and its registration) is retained until
    :func:`release_nixl_exports`, so the consumer can read it.
    """
    if not has_cuda_array_interface(arr):
        raise ValueError(
            "nixl encoding requires a CUDA array "
            f"(object with __cuda_array_interface__), got {type(arr).__name__}"
        )
    if not _is_c_contiguous(arr):
        raise ValueError(
            "nixl encoding requires a C-contiguous array; make a contiguous "
            "copy first (e.g. cupy.ascontiguousarray / torch.Tensor.contiguous)."
        )

    iface = arr.__cuda_array_interface__
    shape = tuple(iface["shape"])
    dtype = np.dtype(iface["typestr"])
    device = _device_ordinal(arr)

    agent = _get_nixl_agent()
    reg_descs = agent.get_reg_descs([_nixl_tensor_view(arr)], "VRAM")
    reg_handle = agent.register_memory(reg_descs)
    if reg_handle is None:
        raise RuntimeError("nixl register_memory failed")
    # Retain the source array *and* its registration until release.
    _NIXL_EXPORT_REGISTRY.append((reg_handle, arr))

    xfer_descs = reg_handle.trim()
    meta = agent.get_agent_metadata()
    descs_ser = agent.get_serialized_descs(xfer_descs)

    meta_b64 = pybase64.b64encode_as_string(meta)
    descs_b64 = pybase64.b64encode_as_string(descs_ser)
    return {
        "object_type": "array",
        "shape": list(shape),
        "dtype": dtype.name,
        "data": {
            "buffer": f"{meta_b64}:{descs_b64}:{device}",
            "encoding": "nixl",
        },
    }


# ---------------------------------------------------------------------------
# Decode side: add the remote agent and READ into an owned buffer
# ---------------------------------------------------------------------------


def load_nixl_arraydict(val: ArrayDict) -> IpcDeviceArray:
    """Load a CUDA array from a NIXL transfer descriptor.

    Adds the producer as a remote agent, allocates a fresh device buffer on the
    target device, registers it, posts a matched READ that pulls the producer's
    bytes into it, synchronises, and returns an :class:`IpcDeviceArray` owning
    that buffer -- the same consumer-facing wrapper as ``cuda_ipc``.
    """
    meta_b64, descs_b64, device_str = val["data"]["buffer"].split(":")
    device = int(device_str)
    remote_meta = pybase64.b64decode(meta_b64, validate=True)
    remote_descs_ser = pybase64.b64decode(descs_b64, validate=True)

    dtype = np.dtype(val["dtype"])
    shape = tuple(val["shape"])
    nbytes = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize

    agent = _get_nixl_agent()
    cudart = _get_cudart()

    ret = cudart.cudaSetDevice(device)
    if ret != 0:
        raise RuntimeError(
            f"cudaSetDevice({device}) failed: {_cuda_error_string(cudart, ret)}"
        )
    owned_ptr = ctypes.c_void_p()
    ret = cudart.cudaMalloc(ctypes.byref(owned_ptr), ctypes.c_size_t(nbytes))
    if ret != 0:
        raise RuntimeError(f"cudaMalloc failed: {_cuda_error_string(cudart, ret)}")

    try:
        peer = agent.add_remote_agent(remote_meta)
        peer_name = peer.decode() if isinstance(peer, bytes) else peer
        remote_descs = agent.deserialize_descs(remote_descs_ser)

        # Register our owned buffer as the READ destination.
        local_reg = agent.register_memory(
            agent.get_reg_descs([(owned_ptr.value, nbytes, device, "")], "VRAM")
        )
        if local_reg is None:
            raise RuntimeError("nixl register_memory (local) failed")
        try:
            local_descs = local_reg.trim()
            handle = agent.initialize_xfer(
                "READ", local_descs, remote_descs, peer_name, b"tesseract"
            )
            agent.transfer(handle)
            _wait_for_xfer(agent, handle)
        finally:
            agent.deregister_memory(local_reg)
    except Exception:
        cudart.cudaFree(owned_ptr)
        raise

    return IpcDeviceArray(owned_ptr.value, device, shape, dtype)


def _wait_for_xfer(agent: Any, handle: Any) -> None:
    """Block until a NIXL transfer completes, raising on error."""
    import time

    while True:
        state = agent.check_xfer_state(handle)
        if state == "DONE":
            return
        if state == "ERR":
            raise RuntimeError("nixl transfer failed")
        time.sleep(0.0005)


# ---------------------------------------------------------------------------
# DeviceTransport backend
# ---------------------------------------------------------------------------


class NixlTransport:
    """DeviceTransport backend for the experimental ``json+nixl`` mode.

    Point-to-point, auto-selecting: same-host it rides NIXL's CUDA IPC / shared
    memory backend, cross-host it rides UCX (RDMA where available). Receiver
    driven like ``cuda_ipc`` -- the producer publishes an inert registration and
    the consumer READs it -- so bootstrap and flush are no-ops: the producer's
    agent metadata and per-array descriptor ride in-band in the JSON response.
    """

    name = "nixl"
    reach = "both"

    def bootstrap(self, role: Any, peer_offer: Any) -> None:
        """No-op: agent metadata rides in the response; the READ is one-directional."""

    def register(self, arr: Any, session: Any = None) -> ArrayDict:
        """Register ``arr`` with the NIXL agent and build its transfer descriptor."""
        return dump_nixl_arraydict(arr)

    def descriptor(self, handle: ArrayDict) -> ArrayDict:
        """Return the array dict :meth:`register` already produced."""
        return handle

    def flush(self, session: Any = None) -> None:
        """No-op: NIXL READ is receiver-driven, so there is nothing to post."""

    def receive(self, val: ArrayDict, session: Any = None) -> IpcDeviceArray:
        """READ the exported bytes into a fresh consumer-owned ``IpcDeviceArray``."""
        return load_nixl_arraydict(val)

    def release(self, session: Any = None) -> None:
        """Deregister the buffers this request exported via NIXL."""
        release_nixl_exports()


def _register_nixl_transport() -> None:
    """Register the nixl backend once this module is imported."""
    from tesseract_core.runtime.device_transport import register_transport

    register_transport(NixlTransport())


_register_nixl_transport()
