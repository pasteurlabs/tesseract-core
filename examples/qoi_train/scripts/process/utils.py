import numpy as np


def compute_bbox_stats(xyz: np.ndarray) -> dict[str, np.ndarray | float]:
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