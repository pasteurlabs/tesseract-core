import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset
from .process.utils import compute_bbox_stats


def cad_collate(batch):
    # batch is a list of tuples: (xyz, nrm, params, qoi)
    xyz = torch.stack([torch.as_tensor(b[0]) for b in batch], dim=0)
    first_nrm = batch[0][1]
    if first_nrm is not None:
        nrm = torch.stack([torch.as_tensor(b[1]) for b in batch], dim=0)
        feats = torch.cat([xyz, nrm], dim=-1)  # (B, N, 6)
    else:
        nrm = None
        feats = xyz  # (B, N, 3)

    x = feats.permute(0, 2, 1).contiguous()  # (B, C, N)
    xyz = xyz.permute(0, 2, 1).contiguous()  # (B, 3, N)
    params = torch.stack([torch.as_tensor(b[2]) for b in batch], dim=0)
    qoi = torch.stack([torch.as_tensor(b[3]) for b in batch], dim=0)
    
    return {
        "x": x,
        "xyz": xyz,
        "params": params,
        "qoi": qoi,
    }


@dataclass
class RawDataSample:
    """Raw data sample without any preprocessing or scaling."""

    xyz: np.ndarray  # (N, 3) point coordinates
    normals: Optional[np.ndarray]  # (N, 3) normal vectors or None
    params: np.ndarray  # (P,) parameter values
    qoi: np.ndarray  # (Q,) quantity of interest values
    file_path: Path  # original file path for reference
    source_idx: int = 0  # index in the dataset.files list


class CADDataset(Dataset):
    """
    Raw dataset that loads data without any preprocessing, scaling, or augmentation.
    This allows for computing global statistics before applying scaling.
    """

    def __init__(self, files: list[str | Path], config_path: Path):
        self.files = sorted(list(files))  # Sort for consistent ordering
        self.cfg = self._load_config(config_path)

    def _load_config(self, config_path: Path) -> dict:
        """Load configuration from YAML file."""
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _compute_expression(self, names: np.ndarray, values: np.ndarray, expression_config: dict, data_type: str = "feature") -> np.ndarray:
        """
        Compute custom expressions based on names and values (works for both params and QoI).

        Args:
            names: Array of feature names from the data
            values: Array of feature values corresponding to the names
            expression_config: Configuration dict defining the expression
            data_type: Type of data being processed ("param" or "qoi") for logging

        Returns:
            Computed values based on the expression
        """
        # Create a mapping from names to values for easy lookup
        data_dict = {name: value for name, value in zip(names, values)}

        expression_type = expression_config.get("type", "select")

        if expression_type == "ratio":
            # Ratio: numerator / denominator
            numerator_patterns = expression_config.get("numerator", [])
            denominator_patterns = expression_config.get("denominator", [])

            # Find matching names for numerator
            numerator_values = []
            for pattern in numerator_patterns:
                matches = [value for name, value in data_dict.items() if pattern.lower() in name.lower()]
                numerator_values.extend(matches)

            # Find matching names for denominator
            denominator_values = []
            for pattern in denominator_patterns:
                matches = [value for name, value in data_dict.items() if pattern.lower() in name.lower()]
                denominator_values.extend(matches)

            if not numerator_values or not denominator_values:
                raise ValueError(f"Could not find {data_type} matches for ratio expression. "
                               f"Available {data_type} names: {list(data_dict.keys())}")

            # Compute ratio (average numerator / average denominator)
            num_avg = np.mean(numerator_values)
            den_avg = np.mean(denominator_values)

            if abs(den_avg) < 1e-12:
                print(f"⚠️  Warning: Denominator near zero ({den_avg}) in ratio calculation")
                ratio = 0.0
            else:
                ratio = num_avg / den_avg

            #print(f"📊 Computed {data_type} ratio: {num_avg:.6f} / {den_avg:.6f} = {ratio:.6f}")
            return np.array([ratio])

        elif expression_type == "difference":
            # Difference: minuend - subtrahend
            minuend_patterns = expression_config.get("minuend", [])
            subtrahend_patterns = expression_config.get("subtrahend", [])

            # Find matching names
            minuend_values = []
            for pattern in minuend_patterns:
                matches = [value for name, value in data_dict.items() if pattern.lower() in name.lower()]
                minuend_values.extend(matches)

            subtrahend_values = []
            for pattern in subtrahend_patterns:
                matches = [value for name, value in data_dict.items() if pattern.lower() in name.lower()]
                subtrahend_values.extend(matches)

            if not minuend_values or not subtrahend_values:
                raise ValueError(f"Could not find {data_type} matches for difference expression. "
                               f"Available {data_type} names: {list(data_dict.keys())}")

            # Compute difference
            min_avg = np.mean(minuend_values)
            sub_avg = np.mean(subtrahend_values)
            difference = min_avg - sub_avg

            # print(f"📊 Computed {data_type} difference: {min_avg:.6f} - {sub_avg:.6f} = {difference:.6f}")
            return np.array([difference])

        elif expression_type == "select":
            # Select specific features by pattern or exact name matching
            patterns = expression_config.get("patterns", expression_config.get("expression", []))
            selected_values = []

            for pattern in patterns:
                # Try exact match first
                if pattern in data_dict:
                    selected_values.append(data_dict[pattern])
                    # print(f"📊 Selected {data_type} '{pattern}': {data_dict[pattern]:.6f}")
                else:
                    # Try pattern matching
                    matches = [(name, value) for name, value in data_dict.items() if pattern.lower() in name.lower()]
                    if matches:
                        for name, value in matches:
                            selected_values.append(value)
                            # print(f"📊 Selected {data_type} '{name}' (matched pattern '{pattern}'): {value:.6f}")
                    else:
                        print(f"⚠️  Warning: No match found for pattern '{pattern}'")

            if not selected_values:
                raise ValueError(f"Could not find {data_type} matches for selection patterns {patterns}. "
                               f"Available {data_type} names: {list(data_dict.keys())}")

            return np.array(selected_values)

        elif expression_type == "custom":
            # Custom mathematical expression using names as variables
            expression = expression_config.get("expression", "")

            if not expression:
                raise ValueError("Custom expression type requires 'expression' field")

            # Replace name patterns in the expression with actual values
            import re

            # Create a safe namespace for evaluation
            safe_dict = {"np": np, "__builtins__": {}}

            # Add values to the namespace
            for name, value in data_dict.items():
                # Replace spaces and special chars with underscores for valid variable names
                var_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
                safe_dict[var_name] = value

                # Also replace in the expression string
                expression = expression.replace(name, var_name)

            try:
                result = eval(expression, safe_dict)
                if np.isscalar(result):
                    result = np.array([result])
                else:
                    result = np.array(result)

                #print(f"📊 Computed custom {data_type} expression: {result}")
                return result

            except Exception as e:
                raise ValueError(f"Error evaluating custom {data_type} expression '{expression}': {e}")

        else:
            raise ValueError(f"Unknown expression type: {expression_type}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> RawDataSample:
        file_path = self.files[idx]
        data = np.load(file_path, allow_pickle=True)

        xyz = data["points"].astype(np.float32)  # (N, 3)
        normals = (
            data["normals"].astype(np.float32)
            if self.cfg["model_spec"]["include_normals"] and "normals" in data
            else None
        )  # (N, 3) or None
        # Combine different parameter sources if needed
        params_list = []
        params_names = []

        # Add basic parameters if configured
        if self.cfg["model_spec"]["include_bc_params"]:
            if "bc_params" in data:
                basic_params = data["bc_params"].astype(np.float32)
                params_list.append(basic_params)
                params_names.append(data["bc_param_names"])
        if self.cfg["model_spec"]["include_geom_params"]:
            if "geom_params" in data:
                geom_params = data["geom_params"].astype(np.float32)
                params_list.append(geom_params)
                params_names.append(data["geom_param_names"])
        if self.cfg["model_spec"]["include_point_derived_params"]:
            bbox_dict, bbox_values = compute_bbox_stats(xyz)
            params_list.append(bbox_values)
            params_names.append(np.asarray(list(bbox_dict.keys()), dtype=object))

        params = (
            np.concatenate(params_list, axis=0)
            if params_list
            else np.array([], dtype=np.float32)
        )
        params_names = np.concatenate(params_names, axis=0) if params_names else np.array([], dtype=object)

        # Handle param computation - check if custom expressions are defined
        param_config = self.cfg.get("param_expressions", None)

        if param_config is not None and len(params_names) > 0:
            # Custom param expressions are defined
            # print(f"Available param names: {list(params_names)}")

            # Compute all defined expressions
            computed_params = []
            for expr_name, expr_config in param_config.items():
                if expr_config.get("enabled", True):  # Allow enabling/disabling expressions
                    try:
                        expr_result = self._compute_expression(params_names, params, expr_config, data_type="param")
                        computed_params.append(expr_result)
                    except Exception as e:
                        print(f"Error computing param expression '{expr_name}': {e}")
                        raise

            if computed_params:
                params = np.concatenate(computed_params, axis=0)
            else:
                # Keep original params if no expressions computed
                print(f"No param expressions computed, using original params")

        # Handle QoI computation - check if custom expressions are defined
        qoi_config = self.cfg.get("qoi_expressions", None)

        if qoi_config is not None:
            # Custom QoI expressions are defined
            qoi_names = data["qoi_names"] if "qoi_names" in data else None
            qoi_values = data["qoi"].astype(np.float32)

            if qoi_names is None:
                raise ValueError(f"QoI expressions defined but no 'qoi_names' found in data file {file_path}")

            # print(f"Available QoI names: {list(qoi_names)}")

            # Compute all defined expressions
            computed_qois = []
            for expr_name, expr_config in qoi_config.items():
                if expr_config.get("enabled", True):  # Allow enabling/disabling expressions
                    try:
                        expr_result = self._compute_expression(qoi_names, qoi_values, expr_config, data_type="qoi")
                        computed_qois.append(expr_result)
                    except Exception as e:
                        print(f"Error computing QoI expression '{expr_name}': {e}")
                        raise

            if computed_qois:
                qoi = np.concatenate(computed_qois, axis=0)
                #print(f"Final computed QoI shape: {qoi.shape}")
            else:
                # Fallback to original QoI if no expressions computed
                qoi = qoi_values
                print(f"No QoI expressions computed, using original QoI")
        else:
            # Use original QoI values
            qoi = data["qoi"].astype(np.float32)  # (Q,)

        return RawDataSample(
            xyz=xyz, normals=normals, params=params, qoi=qoi, file_path=file_path, source_idx=idx
        )


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
