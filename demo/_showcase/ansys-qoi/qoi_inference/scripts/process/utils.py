import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all random number generators.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pca_align(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align principal axes to XYZ using PCA (size preserved).

    Args:
        xyz: Point cloud coordinates (N, 3)

    Returns:
        Tuple of (aligned_points, rotation_matrix) where aligned_points has shape (N, 3)
        and rotation_matrix is (3, 3)
    """
    X = xyz - xyz.mean(axis=0, keepdims=True)
    # covariance (3x3)
    C = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    # sort by descending variance
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order].astype(np.float32)  # columns are principal directions
    Xr = X @ R
    return Xr + xyz.mean(axis=0, keepdims=True), R


def create_train_val_test_split(
    dataset: torch.utils.data.Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple:
    """Split dataset into train, validation, and test subsets.

    Args:
        dataset: PyTorch dataset to split
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, split_info)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    print(
        f"Dataset split: {n_total} total → {n_train} train, {n_val} val, {n_test} test"
    )

    # Generate indices
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    # Create subsets
    from torch.utils.data import Subset

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    split_info = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
    }

    return train_dataset, val_dataset, test_dataset, split_info


def compute_stats_from_samples(samples: list) -> dict:
    """Compute global statistics from a list of raw samples.

    Args:
        samples: List of RawDataSample objects

    Returns:
        Dictionary containing statistics for point_cloud, params, qoi, and num_samples
    """
    all_xyz = []
    all_params = []
    all_qoi = []

    print(f"Computing global statistics from {len(samples)} samples...")
    for sample in samples:
        all_xyz.append(sample.xyz)
        all_params.append(sample.params)
        all_qoi.append(sample.qoi)

    # Compute point cloud statistics
    all_points = np.concatenate(all_xyz, axis=0)  # (total_points, 3)
    point_stats = {
        "mean": np.mean(all_points, axis=0, dtype=np.float32),
        "std": np.std(all_points, axis=0, dtype=np.float32),
        "min": np.min(all_points, axis=0).astype(np.float32),
        "max": np.max(all_points, axis=0).astype(np.float32),
    }

    # Compute parameter statistics (handles empty params gracefully)
    if all_params and len(all_params[0]) > 0:
        all_params = np.array(all_params)  # (N, P)
        param_stats = {
            "mean": np.mean(all_params, axis=0, dtype=np.float32),
            "std": np.std(all_params, axis=0, dtype=np.float32),
            "min": np.min(all_params, axis=0).astype(np.float32),
            "max": np.max(all_params, axis=0).astype(np.float32),
        }
    else:
        param_stats = {
            "mean": np.array([], dtype=np.float32),
            "std": np.array([], dtype=np.float32),
            "min": np.array([], dtype=np.float32),
            "max": np.array([], dtype=np.float32),
        }

    # Compute QoI statistics
    all_qoi = np.array(all_qoi)  # (N, Q)
    qoi_stats = {
        "mean": np.mean(all_qoi, axis=0, dtype=np.float32),
        "std": np.std(all_qoi, axis=0, dtype=np.float32),
        "min": np.min(all_qoi, axis=0).astype(np.float32),
        "max": np.max(all_qoi, axis=0).astype(np.float32),
    }

    print("Global statistics computed successfully!")
    return {
        "point_cloud": point_stats,
        "params": param_stats,
        "qoi": qoi_stats,
        "num_samples": len(samples),
    }


def get_dataset_dimensions(data_loader: torch.utils.data.DataLoader) -> tuple[int, int]:
    """Extract parameter and QoI dimensions from dataset.

    Args:
        data_loader: PyTorch DataLoader to sample from

    Returns:
        Tuple of (p_dim, q_dim) where p_dim is parameter dimension and q_dim is QoI dimension
    """
    sample_batch = next(iter(data_loader))
    p_dim = sample_batch["params"].shape[1]
    q_dim = sample_batch["qoi"].shape[1] if len(sample_batch["qoi"].shape) > 1 else 1
    return p_dim, q_dim


def compute_bbox_stats(
    xyz: np.ndarray,
) -> tuple[dict[str, np.ndarray | float], np.ndarray]:
    """Compute bounding box statistics for a point cloud.

    Args:
        xyz: Point cloud coordinates (N, 3)

    Returns:
        Tuple of (bbox_dict, stats_values) where bbox_dict contains individual statistics
        and stats_values is a flattened array of all statistics
    """
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
