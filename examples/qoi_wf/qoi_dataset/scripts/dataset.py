"""PyTorch dataset for CAD simulation data with configurable feature extraction."""

import numpy as np
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import torch
from torch.utils.data import Dataset
from .process.utils import compute_bbox_stats


def cad_collate(batch):
    """
    Collate function for CAD dataset batching.

    Args:
        batch: List of (xyz, normals, params, qoi) tuples

    Returns:
        Dictionary with batched tensors for 'x', 'xyz', 'params', 'qoi'
    """
    xyz = torch.stack([torch.as_tensor(b[0]) for b in batch], dim=0)

    # Handle optional normals
    first_nrm = batch[0][1]
    if first_nrm is not None:
        nrm = torch.stack([torch.as_tensor(b[1]) for b in batch], dim=0)
        feats = torch.cat([xyz, nrm], dim=-1)  # (B, N, 6)
    else:
        feats = xyz  # (B, N, 3)

    x = feats.permute(0, 2, 1).contiguous()  # (B, C, N)
    xyz = xyz.permute(0, 2, 1).contiguous()  # (B, 3, N)
    params = torch.stack([torch.as_tensor(b[2]) for b in batch], dim=0)
    qoi = torch.stack([torch.as_tensor(b[3]) for b in batch], dim=0)

    return {"x": x, "xyz": xyz, "params": params, "qoi": qoi}


@dataclass
class RawDataSample:
    """Container for a single raw data sample."""
    xyz: np.ndarray  # (N, 3) point coordinates
    normals: Optional[np.ndarray]  # (N, 3) normal vectors or None
    params: np.ndarray  # (P,) parameter values
    qoi: np.ndarray  # (Q,) quantity of interest values
    file_path: Path  # original file path for reference
    source_idx: int = 0  # index in the dataset


class ExpressionEvaluator:
    """Evaluates mathematical expressions on named data arrays."""

    @staticmethod
    def evaluate(names: np.ndarray, values: np.ndarray, expr_config: dict, data_type: str = "feature") -> np.ndarray:
        """
        Compute custom expressions based on named values.

        Args:
            names: Array of feature names
            values: Array of feature values
            expr_config: Expression configuration dictionary
            data_type: Type descriptor for error messages ("param" or "qoi")

        Returns:
            Computed values based on the expression
        """
        data_dict = {name: value for name, value in zip(names, values)}
        expr_type = expr_config.get("type", "select")

        handlers = {
            "select": ExpressionEvaluator._eval_select,
            "ratio": ExpressionEvaluator._eval_ratio,
            "difference": ExpressionEvaluator._eval_difference,
            "custom": ExpressionEvaluator._eval_custom,
        }

        if expr_type not in handlers:
            raise ValueError(f"Unknown expression type: {expr_type}")

        return handlers[expr_type](data_dict, expr_config, data_type)

    @staticmethod
    def _match_patterns(data_dict: dict, patterns: list) -> list:
        """Find values matching given patterns (exact or substring)."""
        matched = []
        for pattern in patterns:
            if pattern in data_dict:
                matched.append(data_dict[pattern])
            else:
                # Substring match
                matches = [val for name, val in data_dict.items() if pattern.lower() in name.lower()]
                matched.extend(matches)
        return matched

    @staticmethod
    def _eval_select(data_dict: dict, config: dict, data_type: str) -> np.ndarray:
        """Select specific features by pattern matching."""
        patterns = config.get("patterns", config.get("expression", []))
        selected = ExpressionEvaluator._match_patterns(data_dict, patterns)

        if not selected:
            raise ValueError(
                f"No {data_type} matches for patterns {patterns}. "
                f"Available: {list(data_dict.keys())}"
            )
        return np.array(selected)

    @staticmethod
    def _eval_ratio(data_dict: dict, config: dict, data_type: str) -> np.ndarray:
        """Compute ratio: numerator / denominator."""
        num_patterns = config.get("numerator", [])
        den_patterns = config.get("denominator", [])

        numerators = ExpressionEvaluator._match_patterns(data_dict, num_patterns)
        denominators = ExpressionEvaluator._match_patterns(data_dict, den_patterns)

        if not numerators or not denominators:
            raise ValueError(
                f"Could not find {data_type} matches for ratio. "
                f"Available: {list(data_dict.keys())}"
            )

        num_avg = np.mean(numerators)
        den_avg = np.mean(denominators)

        if abs(den_avg) < 1e-12:
            print(f"⚠️  Denominator near zero in ratio calculation: {den_avg}")
            return np.array([0.0])

        return np.array([num_avg / den_avg])

    @staticmethod
    def _eval_difference(data_dict: dict, config: dict, data_type: str) -> np.ndarray:
        """Compute difference: minuend - subtrahend."""
        min_patterns = config.get("minuend", [])
        sub_patterns = config.get("subtrahend", [])

        minuends = ExpressionEvaluator._match_patterns(data_dict, min_patterns)
        subtrahends = ExpressionEvaluator._match_patterns(data_dict, sub_patterns)

        if not minuends or not subtrahends:
            raise ValueError(
                f"Could not find {data_type} matches for difference. "
                f"Available: {list(data_dict.keys())}"
            )

        return np.array([np.mean(minuends) - np.mean(subtrahends)])

    @staticmethod
    def _eval_custom(data_dict: dict, config: dict, data_type: str) -> np.ndarray:
        """Evaluate custom mathematical expression."""
        import re

        expression = config.get("expression", "")
        if not expression:
            raise ValueError("Custom expression type requires 'expression' field")

        # Create safe namespace for eval
        safe_dict = {"np": np, "__builtins__": {}}

        # Sanitize variable names and add to namespace
        for name, value in data_dict.items():
            var_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            safe_dict[var_name] = value
            expression = expression.replace(name, var_name)

        try:
            result = eval(expression, safe_dict)
            return np.atleast_1d(np.asarray(result))
        except Exception as e:
            raise ValueError(f"Error evaluating custom {data_type} expression '{expression}': {e}")


class CADDataset(Dataset):
    """
    Dataset for loading CAD simulation data from NPZ files.

    Supports:
    - Optional point cloud normals
    - Multiple parameter sources (BC params, geometry params, point-derived)
    - Custom expressions for parameter and QoI computation
    """

    def __init__(self, files: list[str | Path], config_path: Path):
        """
        Initialize dataset from NPZ files.

        Args:
            files: List of .npz data files
            config_path: Path to YAML configuration file
        """
        self.files = sorted(list(files))
        self.cfg = self._load_config(config_path)

    def _load_config(self, config_path: Path) -> dict:
        """Load YAML configuration file."""
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> RawDataSample:
        """
        Load and process a single data sample.

        Args:
            idx: Index of the sample to load

        Returns:
            RawDataSample with xyz, normals, params, and qoi
        """
        file_path = self.files[idx]
        data = np.load(file_path, allow_pickle=True)

        # Load point cloud data
        xyz = data["points"].astype(np.float32)
        normals = self._load_normals(data)

        # Aggregate parameters from multiple sources
        params, param_names = self._aggregate_params(data, xyz)

        # Apply custom param expressions if configured
        if "param_expressions" in self.cfg and len(param_names) > 0:
            params = self._compute_expressions(param_names, params, self.cfg["param_expressions"], "param")

        # Load or compute QoI
        qoi = self._load_qoi(data, file_path)

        return RawDataSample(
            xyz=xyz,
            normals=normals,
            params=params,
            qoi=qoi,
            file_path=file_path,
            source_idx=idx,
        )

    def _load_normals(self, data: Dict) -> Optional[np.ndarray]:
        """Load normals if configured and available."""
        if self.cfg["model_spec"]["include_normals"] and "normals" in data:
            return data["normals"].astype(np.float32)
        return None

    def _aggregate_params(self, data: Dict, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Combine parameters from multiple configured sources."""
        params_list = []
        names_list = []

        # BC parameters
        if self.cfg["model_spec"]["include_bc_params"] and "bc_params" in data:
            params_list.append(data["bc_params"].astype(np.float32))
            names_list.append(data["bc_param_names"])

        # Geometry parameters
        if self.cfg["model_spec"]["include_geom_params"] and "geom_params" in data:
            params_list.append(data["geom_params"].astype(np.float32))
            names_list.append(data["geom_param_names"])

        # Point cloud derived parameters (bounding box stats)
        if self.cfg["model_spec"]["include_point_derived_params"]:
            bbox_dict, bbox_values = compute_bbox_stats(xyz)
            params_list.append(bbox_values)
            names_list.append(np.asarray(list(bbox_dict.keys()), dtype=object))

        params = np.concatenate(params_list) if params_list else np.array([], dtype=np.float32)
        names = np.concatenate(names_list) if names_list else np.array([], dtype=object)

        return params, names

    def _load_qoi(self, data: Dict, file_path: Path) -> np.ndarray:
        """Load QoI values, applying custom expressions if configured."""
        qoi_config = self.cfg.get("qoi_expressions")

        if qoi_config is not None:
            if "qoi_names" not in data:
                raise ValueError(f"QoI expressions defined but no 'qoi_names' in {file_path}")

            qoi_names = data["qoi_names"]
            qoi_values = data["qoi"].astype(np.float32)

            return self._compute_expressions(qoi_names, qoi_values, qoi_config, "qoi")

        # Default: use raw QoI values
        return data["qoi"].astype(np.float32)

    def _compute_expressions(
        self,
        names: np.ndarray,
        values: np.ndarray,
        expr_configs: dict,
        data_type: str
    ) -> np.ndarray:
        """Evaluate all enabled expressions in the config."""
        results = []

        for expr_name, expr_config in expr_configs.items():
            if not expr_config.get("enabled", True):
                continue

            try:
                result = ExpressionEvaluator.evaluate(names, values, expr_config, data_type)
                results.append(result)
            except Exception as e:
                print(f"❌ Error computing {data_type} expression '{expr_name}': {e}")
                raise

        if results:
            return np.concatenate(results, axis=0)

        # Fallback to original values if no expressions computed
        print(f"⚠️  No {data_type} expressions computed, using original values")
        return values
    

def create_raw_splits(
    dataset: CADDataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
):
    """Create train/val/test splits of raw data samples."""
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
    np.random.seed(seed)
    indices = np.random.permutation(n_total).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    # Create raw sample lists
    train_samples = [dataset[i] for i in train_indices]
    val_samples = [dataset[i] for i in val_indices]
    test_samples = [dataset[i] for i in test_indices]

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

    return train_samples, val_samples, test_samples, split_info


class ScaledCADDataset(Dataset):
    """
    PyTorch dataset for scaled data samples ready for training.
    """
    
    def __init__(self, scaled_samples: list):
        self.samples = scaled_samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        xyz = sample.xyz  # (N, 3)
        normals = sample.normals  # (N, 3) or None
        params = sample.params  # (P,)
        qoi = sample.qoi  # (Q,)
        
        return xyz, normals, params, qoi
    

def create_scaled_datasets(scaled_train, scaled_val, scaled_test):
    """Convert scaled samples to PyTorch datasets."""
    train_dataset = ScaledCADDataset(scaled_train)
    val_dataset = ScaledCADDataset(scaled_val)  
    test_dataset = ScaledCADDataset(scaled_test)
    
    return train_dataset, val_dataset, test_dataset
