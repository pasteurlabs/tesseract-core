"""Utility functions for data processing: mesh loading, sampling, and report parsing."""

import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Iterable
import open3d as o3d
from dataclasses import dataclass


def load_mesh(stl_path: Path) -> o3d.geometry.TriangleMesh:
    """
    Load an STL mesh from disk and ensure normals are present.
    """
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh



def submesh_in_sphere(
    mesh: o3d.geometry.TriangleMesh,
    center: np.ndarray,
    radius: float,
) -> o3d.geometry.TriangleMesh | None:
    """
    Return a submesh consisting of triangles whose centroids fall inside a sphere.
    """
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    # Triangle centroids
    tri_centroids = verts[tris].mean(axis=1)
    dists = np.linalg.norm(tri_centroids - center[None, :], axis=1)

    mask = dists <= radius
    if not np.any(mask):
        return None

    sub = o3d.geometry.TriangleMesh()
    sub.vertices = mesh.vertices  # reuse original vertices
    sub.triangles = o3d.utility.Vector3iVector(tris[mask])
    sub.compute_vertex_normals()
    return sub


def sample_points(
    mesh: o3d.geometry.TriangleMesh, n_points: int = 1024, method: str = "poisson"
):
    """
    Sample N points from a mesh surface using Poisson-disk (default) or uniform sampling.
    """
    if method == "poisson":
        # More uniform coverage on the surface
        pcd = mesh.sample_points_poisson_disk(n_points, init_factor=5)
    else:
        pcd = mesh.sample_points_uniformly(n_points)
    pts = np.asarray(pcd.points, dtype=np.float32)
    nrm = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None
    return pts, nrm


def sample_points_with_spheres(
    mesh: o3d.geometry.TriangleMesh,
    n_points: int = 1024,
    method: str = "poisson",
    spheres: Iterable[tuple[np.ndarray, float]] = (),
    sphere_fraction: float = 0.5,
):
    """
    Sample points on a mesh, allocating more samples inside given spheres.

    Parameters
    ----------
    mesh : TriangleMesh
        Full mesh to sample from.
    n_points : int
        Total number of points to return.
    method : {"poisson", "uniform"}
        Sampling method.
    spheres : iterable of (center, radius)
        Each center is a (3,) np.ndarray; radius is a float.
    sphere_fraction : float in (0, 1]
        Fraction of total samples to dedicate to all spheres combined.
        The remainder is sampled over the full mesh.
    """
    spheres = list(spheres)
    if not spheres:
        return sample_points(mesh, n_points, method)

    # How many points we want in all spheres combined
    n_roi_total = int(n_points * sphere_fraction)
    n_roi_total = max(0, min(n_roi_total, n_points))

    # Split evenly across all spheres
    n_per_sphere = n_roi_total // len(spheres) if len(spheres) > 0 else 0

    pts_chunks = []
    nrm_chunks = []
    used_roi = 0

    for center, radius in spheres:
        sub = submesh_in_sphere(mesh, np.asarray(center, dtype=np.float32), float(radius))
        if sub is None:
            continue  # nothing of the mesh inside this sphere

        n_i = min(n_per_sphere, n_points - used_roi)
        if n_i <= 0:
            break

        if method == "poisson":
            pcd_i = sub.sample_points_poisson_disk(n_i, init_factor=5)
        else:
            pcd_i = sub.sample_points_uniformly(n_i)

        pts_i = np.asarray(pcd_i.points, dtype=np.float32)
        pts_chunks.append(pts_i)

        if pcd_i.has_normals():
            nrm_i = np.asarray(pcd_i.normals, dtype=np.float32)
            nrm_chunks.append(nrm_i)

        used_roi += len(pts_i)

    # Remaining points sampled over full mesh
    remaining = n_points - used_roi
    if remaining > 0:
        pts_rest, nrm_rest = sample_points(mesh, remaining, method)
        pts_chunks.append(pts_rest)
        if nrm_rest is not None:
            nrm_chunks.append(nrm_rest)

    pts = np.concatenate(pts_chunks, axis=0)
    nrm = np.concatenate(nrm_chunks, axis=0) if nrm_chunks else None

    # In rare cases, we might end up with a few more/less due to rounding;
    # ensure exactly n_points by trimming if needed.
    if len(pts) > n_points:
        pts = pts[:n_points]
        if nrm is not None:
            nrm = nrm[:n_points]

    return pts, nrm
    


def compute_bbox_stats(xyz: np.ndarray) -> Dict[str, np.ndarray | float]:
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    size = mx - mn
    diag = float(np.linalg.norm(size))
    max_side = float(size.max())
    centroid = xyz.mean(axis=0)
    bbox_dict = {
        "min": mn.astype(np.float32),
        "max": mx.astype(np.float32),
        "size": size.astype(np.float32),
        "diag": np.float32(diag),
        "max_side": np.float32(max_side),
        "centroid": centroid.astype(np.float32),
    }
    flattened_values = []
    for value in bbox_dict.values():
        if isinstance(value, np.ndarray):
            flattened_values.extend(value.flatten())
        else:
            flattened_values.append(value)

    stats_values = np.array(flattened_values, dtype=np.float32)
    return bbox_dict, stats_values


def extract_cad_sketch(filename: Path) -> np.ndarray:
    """Extract CAD sketch from the given folder."""
    # Placeholder implementation
    cad_sketch_path = Path(filename)
    if not cad_sketch_path.exists():
        raise FileNotFoundError(f"CAD sketch file {filename} not found")

    cad_sketch_features = {}
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                try:
                    cad_sketch_features[row[1]] = float(row[2])
                except ValueError:
                    continue
    names = list(cad_sketch_features.keys())
    values = list(cad_sketch_features.values())
    return names, values


@dataclass
class SurfaceIntegralReport:
    quantity: str
    units: str
    values: Dict[str, float]
    net: Optional[float]

    @classmethod
    def from_text(cls, text: str) -> "SurfaceIntegralReport":
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        quantity, units = cls._extract_header(lines)
        values, net = cls._extract_data(lines)

        return cls(quantity, units, values, net)

    @classmethod
    def from_file(cls, path: Path | str) -> "SurfaceIntegralReport":
        text = Path(path).read_text()
        return cls.from_text(text)

    # ------------- Internal helpers -------------

    @staticmethod
    def _extract_header(lines):
        for line in lines:
            if "[" in line and "]" in line:
                qty = line[: line.index("[")].strip()
                unit = line[line.index("[") + 1 : line.index("]")]
                return qty, unit
        return "Unknown", "Unknown"

    @staticmethod
    def _extract_data(lines):
        values = {}
        net = None

        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue

            # last element is a number
            try:
                val = float(parts[-1])
            except ValueError:
                continue

            # everything before number = name
            name = " ".join(parts[:-1])

            if name.lower() == "net":
                net = val
            else:
                values[name] = val

        return values, net


def read_experiment_csv_to_metadata(csv_file_path: Path, data_dir: Path, shift_index=1) -> list[dict]:
    """
    Read experiment CSV data and convert each row to metadata dictionary format.
    """    
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        
        for row in reader:
            row = {key.strip(): value.strip() for key, value in row.items()}
            # Skip unsuccessful experiments
            if row.get('Success', '').lower() != 'true':
                continue
                
            # Create metadata dictionary for this experiment
            metadata_dict = {
                "file_series_version": 1.0,
                "dt": 0.0,
                "variations": {
                    key: float(value) for key, value in list(row.items())[1:-1]
                },
                "simulations": [
                    {
                        "time": 0.0,
                        "file": "basic.cas.h5"
                    }
                ]
            }
            
            json_filename = data_dir / f"Experiment_{int(row['Experiment'])+shift_index}/metadata.json.series"
            with open(json_filename, 'w') as json_file:
                json.dump(metadata_dict, json_file, indent=4)
