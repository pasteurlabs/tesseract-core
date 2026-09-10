# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example Tesseract demonstrating ``Ref``: nested models as sidecar JSON files.

``apply`` returns one ``Frame`` per requested time step. Each ``Frame`` is a
user-defined Pydantic model holding two differentiable arrays and a plain
string. Wrapping it in ``Ref[Frame, "name"]`` means that, when the Tesseract is
run with an output path and the ``json+binref`` format, the main payload is
just::

    {"frames": ["frame_000.json", "frame_001.json", "frame_002.json"]}

with each frame written to its own JSON file (whose arrays are in turn binrefs
into the shared ``.bin`` buffer). Run with plain ``json`` output and the frames
are inlined instead, so the same Tesseract still serves over plain HTTP.

Crucially, ``Ref`` is transparent to schema generation: the differentiable
arrays inside ``Frame`` remain visible as ``frames.[0].displacement`` etc., so
``abstract_eval`` and all three gradient endpoints work unchanged.
"""

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32, Float64
from tesseract_core.runtime.experimental import Ref

#
# Schemas
#


class Frame(BaseModel):
    """A single time step. Two differentiable array leaves plus a string leaf."""

    name: str = Field(
        description="Frame identifier, also used as the sidecar filename."
    )
    displacement: Differentiable[Array[(None, 3), Float32]] = Field(
        description="Per-node displacement vectors."
    )
    pressure: Differentiable[Array[(None,), Float64]] = Field(
        description="Per-node scalar pressure."
    )

    def __ref_name__(self) -> str:
        """Name this frame's sidecar file after the frame itself."""
        return self.name


class InputSchema(BaseModel):
    scale: Differentiable[Float32] = Field(
        description="Scalar multiplier applied to every frame.", default=1.0
    )
    n_nodes: int = Field(description="Number of nodes per frame.", default=4, gt=0)
    n_frames: int = Field(description="Number of frames to produce.", default=3, gt=0)


class OutputSchema(BaseModel):
    frames: list[Ref[Frame]] = Field(
        description="One frame per time step, each serialized to its own JSON file."
    )


#
# Helpers
#


def _base_fields(n_nodes: int, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, differentiable-in-`scale` frame contents (before scaling)."""
    node = np.arange(n_nodes, dtype="float64")
    displacement = np.stack([node, node + frame_idx, node - frame_idx], axis=-1)
    pressure = node**2 + frame_idx
    return displacement.astype("float32"), pressure


def _frames(inputs: InputSchema, scale: float) -> list[Frame]:
    frames = []
    for i in range(inputs.n_frames):
        displacement, pressure = _base_fields(inputs.n_nodes, i)
        frames.append(
            Frame(
                name=f"frame_{i:03d}",
                displacement=(scale * displacement).astype("float32"),
                pressure=scale * pressure,
            )
        )
    return frames


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Produce one frame per time step, scaled by `scale`."""
    return OutputSchema(frames=_frames(inputs, float(inputs.scale)))


#
# Optional endpoints
#
# Gradient payloads are flat {path: array} mappings keyed by paths *into* the
# output tree (e.g. "frames.[1].pressure"), so they never traverse a Ref and
# need no ref-specific handling. Every output here is linear in `scale`, so the
# derivative w.r.t. `scale` is just the unscaled field.


def _unit_frames(inputs: InputSchema) -> dict[str, np.ndarray]:
    """d(output)/d(scale) for every differentiable output path."""
    derivatives = {}
    for i, frame in enumerate(_frames(inputs, 1.0)):
        derivatives[f"frames.[{i}].displacement"] = frame.displacement
        derivatives[f"frames.[{i}].pressure"] = frame.pressure
    return derivatives


def jacobian(
    inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]
) -> dict[str, dict[str, np.ndarray]]:
    derivatives = _unit_frames(inputs)
    return {out: {inp: derivatives[out] for inp in jac_inputs} for out in jac_outputs}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    derivatives = _unit_frames(inputs)
    tangent = float(tangent_vector["scale"])
    return {out: derivatives[out] * tangent for out in jvp_outputs}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    derivatives = _unit_frames(inputs)
    total = sum(np.sum(derivatives[out] * cotangent_vector[out]) for out in vjp_outputs)
    return {"scale": np.float32(total)}
