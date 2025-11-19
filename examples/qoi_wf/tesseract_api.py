# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from torch.utils._pytree import tree_map
from typing_extensions import Self

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

from tesseract_core.runtime.experimental import InputFileReference

#
# Schemata
#

class InputSchema(BaseModel):

    xyz: Array[(None, 3), Float32] = Field(
        description="(N, 3) point coordinates"
    )  

    normals: Array[(None, 3), Float32] = Field(
        description="(N, 3) normal vectors or None",
        default=None
    )

    params : Array[(None, ), Float32] = Field(
        description="parameter values",
    )

    trained_model: InputFileReference

class OutputSchema(BaseModel):

    qoi : Array[(None, ), Float32] = Field(
        description="QoIs",
    )
   

def evaluate(inputs: Any) -> Any:
    xyz = inputs["xyz"]

    # TODO: put code here

    print(inputs["trained_model"])

    f = open(inputs["trained_model"])

    print(f)

    return {
        "qoi" : torch.zeros((10,), dtype=torch.float32)
    }


def apply(inputs: InputSchema) -> OutputSchema:
    # Optional: Insert any pre-processing/setup that doesn't require tracing
    # and is only required when specifically running your apply function
    # and not your differentiable endpoints.
    # For example, you might want to set up a logger or mlflow server.
    # Pre-processing should not modify any input that could impact the
    # differentiable outputs in a nonlinear way (a constant shift
    # should be safe)

    # Convert to pytorch tensors
    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    out = evaluate(tensor_inputs)

    # Optional: Insert any post-processing that doesn't require tracing
    # For example, you might want to save to disk or modify a non-differentiable
    # output. Again, do not modify any differentiable output in a non-linear way.
    return out


# def vector_jacobian_product(
#     inputs: InputSchema,
#     vjp_inputs: set[str],
#     vjp_outputs: set[str],
#     cotangent_vector: dict[str, Any],
# ):
#     # Cast to tuples for consistent ordering in positional function
#     vjp_inputs = tuple(vjp_inputs)
#     # Make ordering of cotangent_vector identical to vjp_inputs
#     cotangent_vector = {key: cotangent_vector[key] for key in vjp_outputs}

#     # convert all numbers and arrays to torch tensors
#     tensor_inputs = tree_map(to_tensor, inputs.model_dump())
#     tensor_cotangent = tree_map(to_tensor, cotangent_vector)

#     # flatten the dictionaries such that they can be accessed by paths
#     pos_inputs = flatten_with_paths(tensor_inputs, vjp_inputs).values()

#     # create a positional function that accepts a list of values
#     filtered_pos_func = filter_func(
#         evaluate, tensor_inputs, vjp_outputs, input_paths=vjp_inputs
#     )

#     _, vjp_func = torch.func.vjp(filtered_pos_func, *pos_inputs)

#     vjp_vals = vjp_func(tensor_cotangent)
#     return dict(zip(vjp_inputs, vjp_vals, strict=True))


to_tensor = lambda x: torch.tensor(x) if isinstance(x, np.generic | np.ndarray) else x


#
# Required endpoints
#



# class HybridPointCloudTreeModel(BaseModel):
#     """Hybrid model: PointNeXt embedder + Random Forest using all parameters."""

#     def __init__(
#         self,
#         name: str = "hybrid_pc_tree",
#         # Model architecture
#         in_dim: int = 6,
#         latent_dim: int = 8,
#         param_fusion: str = "concat",
#         backbone_dim: int = 1024,
#         embedder_type: str = "pointnext",  # "pointnext", "pointnet", or "pointbert"
#         p_dim: int = None,  # Will be auto-detected from dataset if None
#         q_dim: int = None,  # Will be auto-detected from dataset
#         # Embedder parameters
#         embedder_dropout: float = 0.1,
#         fusion_dropout: float = 0.2,
#         use_layer_norm: bool = True,
#         use_residual: bool = False,
#         # PointNet-specific parameters
#         pointnet_hidden_dims: list = None,  # [64, 128, 256] default
#         # PointBERT-specific parameters
#         pointbert_pretrained_path: str = None,  # Path to pre-trained Point-BERT weights
#         pointbert_freeze: bool = True,  # Freeze Point-BERT encoder
#         # Random Forest parameters
#         n_estimators: int = 200,
#         max_depth: int = 15,
#         min_samples_split: int = 2,
#         random_state: int = 42,
#         **tree_kwargs
#     ):
#         super().__init__(name)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Store dimensions - will be auto-detected if None
#         self.latent_dim = latent_dim
#         self.in_dim = in_dim
#         self.p_dim = p_dim  # Can be None, will be set during fit()
#         self.q_dim = q_dim  # Will be set during fit() if None
#         self.param_fusion = param_fusion

#         # Store construction parameters for lazy initialization
#         self._embedder_type = embedder_type.lower()
#         self._backbone_dim = backbone_dim
#         self._embedder_dropout = embedder_dropout
#         self._fusion_dropout = fusion_dropout
#         self._use_layer_norm = use_layer_norm
#         self._use_residual = use_residual
#         self._pointnet_hidden_dims = pointnet_hidden_dims or [64, 128, 256]
#         self._pointbert_pretrained_path = pointbert_pretrained_path
#         self._pointbert_freeze = pointbert_freeze

#         print(f"🔧 Hybrid model configuration:")
#         if p_dim is not None:
#             print(f"   Input parameters: p_dim={p_dim}")
#         else:
#             print(f"   Input parameters will be auto-detected from dataset")
#         if q_dim is not None:
#             print(f"🎯 Output dimension: q_dim={q_dim}")
#         else:
#             print(f"🎯 Output dimension will be auto-detected from dataset")
        
#         # Store tree parameters
#         tree_params = {
#             'n_estimators': n_estimators,
#             'max_depth': max_depth,
#             'min_samples_split': min_samples_split,
#             'random_state': random_state,
#             'n_jobs': -1,
#         }
        
#         # Add valid sklearn RandomForest parameters from tree_kwargs
#         valid_rf_params = {
#             'max_features', 'min_samples_leaf', 'min_weight_fraction_leaf',
#             'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
#             'oob_score', 'warm_start', 'ccp_alpha', 'max_samples'
#         }
        
#         for key, value in tree_kwargs.items():
#             if key in valid_rf_params:
#                 tree_params[key] = value
        
#         self._tree_params = tree_params
        
#         # Components will be initialized in _initialize_components()
#         self.embedder = None
#         self.fusion_head = None
#         self.regularizer = None
#         self.tree_model = None
        
#         # Training state
#         self.embedder_fitted = False
#         self.tree_fitted = False
    
#     def _initialize_components(self, p_dim: int, q_dim: int):
#         """Initialize model components once p_dim and q_dim are known."""
#         if self.embedder is not None:
#             return  # Already initialized

#         self.p_dim = p_dim
#         self.q_dim = q_dim
#         print(f"🎯 Initializing components with p_dim={p_dim}, q_dim={q_dim}")

#         # Create embedder based on type
#         if self._embedder_type == "pointnet":
#             print(f"   Using PointNet embedder (simpler, fewer parameters)")
#             self.embedder = PointNetEmbedder(
#                 in_dim=self.in_dim,
#                 latent_dim=self.latent_dim,
#                 hidden_dims=self._pointnet_hidden_dims,
#                 dropout=self._embedder_dropout,
#                 use_batch_norm=True,
#             ).to(self.device)
#         elif self._embedder_type == "pointnext":
#             print(f"   Using PointNeXt embedder (hierarchical, more capacity)")
#             self.embedder = PointNeXtEmbedder(
#                 in_dim=self.in_dim,
#                 latent_dim=self.latent_dim,
#                 backbone_dim=self._backbone_dim,
#                 dropout=self._embedder_dropout,
#                 use_layer_norm=self._use_layer_norm,
#             ).to(self.device)
#         elif self._embedder_type == "pointbert":
#             print(f"   Using Point-BERT embedder (transformer-based, pre-trained)")
#             if not self._pointbert_pretrained_path:
#                 raise ValueError("Point-BERT requires pretrained_path to be specified")
#             self.embedder = PointBERTEmbedder(
#                 pretrained_path=self._pointbert_pretrained_path,
#                 in_dim=self.in_dim,
#                 latent_dim=self.latent_dim,
#                 freeze=self._pointbert_freeze,
#                 group_size=32,
#                 num_group=64,
#                 trans_dim=384,
#                 depth=12,
#                 num_heads=6,
#                 drop_path_rate=0.1,
#             ).to(self.device)
#         else:
#             raise ValueError(f"Unknown embedder type: {self._embedder_type}. Use 'pointnet', 'pointnext', or 'pointbert'.")

#         # Use the same fusion head as CADQoIModel with generic p_dim
#         self.fusion_head = ParamFusionHead(
#             z_dim=self.latent_dim,
#             p_dim=p_dim,  # Use all parameters
#             q_dim=q_dim,
#             mode=self.param_fusion,
#             dropout=self._fusion_dropout,
#             use_batch_norm=False,
#             use_residual=self._use_residual,
#         ).to(self.device)
        
#         # Add regularizer like in CADQoIModel
#         if self.latent_dim < 32:
#             self.regularizer = nn.Sequential(
#                 nn.Linear(self.latent_dim, self.latent_dim * 2),
#                 nn.LayerNorm(self.latent_dim * 2) if self._use_layer_norm else nn.Identity(),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(self._embedder_dropout),
#                 nn.Linear(self.latent_dim * 2, self.latent_dim),
#             ).to(self.device)
#         else:
#             self.regularizer = nn.Identity()
        
#         # Random Forest for final prediction
#         self.tree_model = RandomForestRegressor(**self._tree_params)
    
#     def fit(self, train_data, val_data=None, training_args=None):
#         """Two-stage training: 1) CADQoI-style embedder, 2) Random Forest."""

#         # Auto-detect p_dim and q_dim from dataset if not provided
#         if self.p_dim is None or self.q_dim is None:
#             detected_p_dim, detected_q_dim = _get_dataset_dimensions(train_data)
#             p_dim = self.p_dim if self.p_dim is not None else detected_p_dim
#             q_dim = self.q_dim if self.q_dim is not None else detected_q_dim
#             print(f"🔍 Auto-detected dimensions: p_dim={p_dim}, q_dim={q_dim}")
#         else:
#             p_dim = self.p_dim
#             q_dim = self.q_dim

#         # Initialize components now that we know p_dim and q_dim
#         self._initialize_components(p_dim, q_dim)
        
#         print(f"Training {self.name} in two stages...")
        
#         # Stage 1: Train embedder using CADQoI-style approach
#         print("Stage 1: Training CADQoI-style embedder...")
#         self._fit_embedder(train_data, val_data, training_args)
        
#         # Stage 2: Extract features and train Random Forest
#         print("Stage 2: Training Random Forest on extracted features...")
#         self._fit_tree(train_data, val_data)
        
#         self.is_fitted = True
#         return self
    
#     def _extract_features(self, data_loader):
#         """Extract features using point cloud + params from batch."""
#         self.embedder.eval()
#         self.fusion_head.eval()

#         all_embedder_features = []
#         all_fusion_features = []
#         all_params = []
#         all_qois = []

#         with torch.no_grad():
#             for batch in data_loader:
#                 x = batch["x"].to(self.device)
#                 xyz = batch["xyz"].to(self.device)
#                 params = batch["params"].to(self.device)  # Use params as defined in batch
#                 qoi = batch["qoi"]

#                 # Extract point cloud features (same as CADQoIModel)
#                 z = self.embedder(x, xyz)  # (B, latent_dim)
#                 z = self.regularizer(z)    # Apply regularization

#                 # Combine latent features with params from batch
#                 if self.param_fusion == "concat":
#                     fusion_input = torch.cat([z, params], dim=-1)  # (B, latent_dim + p_dim)
#                 elif self.param_fusion == "film":
#                     # For FILM, modulate with params
#                     gamma = self.fusion_head.film_gamma(params)
#                     beta = self.fusion_head.film_beta(params)
#                     fusion_input = gamma * z + beta  # (B, latent_dim)
#                 elif self.param_fusion == "gate":
#                     # For gating, gate with params
#                     gate = torch.sigmoid(self.fusion_head.gate(params))
#                     fusion_input = gate * z  # (B, latent_dim)
#                 else:
#                     fusion_input = torch.cat([z, params], dim=-1)  # Default to concat

#                 all_embedder_features.append(z.cpu().numpy())
#                 all_fusion_features.append(fusion_input.cpu().numpy())
#                 all_params.append(params.cpu().numpy())
#                 all_qois.append(qoi.numpy())

#         embedder_features = np.concatenate(all_embedder_features, axis=0)  # (N, latent_dim)
#         fusion_features = np.concatenate(all_fusion_features, axis=0)      # (N, latent_dim + p_dim) or (N, latent_dim)
#         params_features = np.concatenate(all_params, axis=0)               # (N, p_dim)
#         qois = np.concatenate(all_qois, axis=0)                           # (N, qoi_dim)

#         print(f"📊 Extracted features:")
#         print(f"   Embedder features: {embedder_features.shape}")
#         print(f"   Fusion features: {fusion_features.shape}")
#         print(f"   Parameters: {params_features.shape}")

#         # Use fusion features for Random Forest (includes point cloud + parameter interaction)
#         combined_features = fusion_features

#         return combined_features, qois
    
#     def _fit_embedder(self, train_loader, val_loader, training_args):
#         """Train the embedder using params from batch."""

#         # Training parameters
#         epochs = training_args.get('epochs', 50) if training_args else 50
#         lr = training_args.get('lr', 1e-4) if training_args else 1e-4
#         weight_decay = training_args.get('weight_decay', 1e-3) if training_args else 1e-3
#         patience = training_args.get('patience', 20) if training_args else 20
#         gradient_clip_norm = training_args.get('gradient_clip_norm', None) if training_args else None

#         # Create a complete model like CADQoIModel for training
#         model_params = list(self.embedder.parameters()) + list(self.fusion_head.parameters()) + list(self.regularizer.parameters())
#         optimizer = torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

#         best_val_loss = float('inf')
#         no_improve = 0

#         print(f"  Training embedder for {epochs} epochs (using params from batch)...")

#         for epoch in range(epochs):
#             # Training phase
#             self.embedder.train()
#             self.fusion_head.train()

#             train_loss = 0
#             num_batches = 0

#             for batch in train_loader:
#                 x = batch["x"].to(self.device)
#                 xyz = batch["xyz"].to(self.device)
#                 params = batch["params"].to(self.device)  # Use params as defined in batch
#                 qoi = batch["qoi"].to(self.device)

#                 optimizer.zero_grad()

#                 # Forward pass with params from batch
#                 z = self.embedder(x, xyz)
#                 z = self.regularizer(z)
#                 y_pred = self.fusion_head(z, params)  # Use params from batch

#                 # Compute loss
#                 loss = F.mse_loss(y_pred, qoi)

#                 loss.backward()

#                 # Apply gradient clipping if specified
#                 if gradient_clip_norm is not None:
#                     torch.nn.utils.clip_grad_norm_(model_params, gradient_clip_norm)

#                 optimizer.step()

#                 train_loss += loss.item()
#                 num_batches += 1

#             train_loss /= num_batches

#             # Validation phase
#             if val_loader is not None:
#                 self.embedder.eval()
#                 self.fusion_head.eval()

#                 val_loss = 0
#                 num_val_batches = 0

#                 with torch.no_grad():
#                     for batch in val_loader:
#                         x = batch["x"].to(self.device)
#                         xyz = batch["xyz"].to(self.device)
#                         params = batch["params"].to(self.device)  # Use params as defined in batch
#                         qoi = batch["qoi"].to(self.device)

#                         # Forward pass
#                         z = self.embedder(x, xyz)
#                         z = self.regularizer(z)
#                         y_pred = self.fusion_head(z, params)  # Use params from batch

#                         loss = F.mse_loss(y_pred, qoi)
#                         val_loss += loss.item()
#                         num_val_batches += 1

#                 val_loss /= num_val_batches

#                 # Update scheduler
#                 scheduler.step()

#                 # Early stopping
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     no_improve = 0
#                 else:
#                     no_improve += 1

#                 if no_improve >= patience:
#                     print(f"  Early stopping at epoch {epoch+1}")
#                     break

#                 if (epoch + 1) % 10 == 0:
#                     current_lr = scheduler.get_last_lr()[0]
#                     print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={current_lr:.2e}")

#         self.embedder_fitted = True
#         print(f"  Embedder training completed. Best val loss: {best_val_loss:.6f}")
    
    
#     def _fit_tree(self, train_loader, val_loader):
#         """Train Random Forest on extracted features."""
        
#         # Extract features from training data
#         print("  Extracting features from training data...")
#         X_train, y_train = self._extract_features(train_loader)
        
#         print(f"  Feature dimensions: {X_train.shape}")
#         print(f"  Feature type: {'Fusion features' if self.param_fusion == 'concat' else 'Modulated features'}")
        
#         # Train Random Forest
#         print("  Training Random Forest...")
#         self.tree_model.fit(X_train, y_train)
        
#         # Evaluate on training data
#         train_pred = self.tree_model.predict(X_train)
#         train_metrics = compute_metrics(y_train, train_pred)
#         print(f"  Train metrics: {train_metrics}")
        
#         # Evaluate on validation data if available
#         if val_loader is not None:
#             X_val, y_val = self._extract_features(val_loader)
#             val_pred = self.tree_model.predict(X_val)
#             val_metrics = compute_metrics(y_val, val_pred)
#             print(f"  Val metrics: {val_metrics}")
        
#         self.tree_fitted = True
    
#     def predict(self, data_loader) -> np.ndarray:
#         """Make predictions using the hybrid model."""
#         if not self.is_fitted:
#             raise ValueError("Model must be fitted before prediction")
        
#         # Extract features using embedder
#         features, _ = self._extract_features(data_loader)
        
#         # Predict using Random Forest
#         predictions = self.tree_model.predict(features)
        
#         return predictions
    
#     def evaluate(self, data_loader) -> ModelMetrics:
#         """Evaluate the hybrid model."""
#         # Extract features and true labels
#         features, y_true = self._extract_features(data_loader)
        
#         # Make predictions
#         y_pred = self.tree_model.predict(features)
        
#         # Compute metrics
#         return compute_metrics(y_true, y_pred, include_additional=True)
    
#     def save(self, path: Path):
#         """Save the hybrid model."""
#         # Get backbone_dim based on embedder type
#         if self._embedder_type == "pointnext":
#             backbone_dim = self.embedder.backbone.head.weight.shape[0]
#         elif self._embedder_type == "pointbert":
#             backbone_dim = None  # Not used for Point-BERT
#         else:  # pointnet
#             backbone_dim = None  # Not used for PointNet

#         save_dict = {
#             'embedder_state_dict': self.embedder.state_dict(),
#             'fusion_head_state_dict': self.fusion_head.state_dict(),
#             'regularizer_state_dict': self.regularizer.state_dict(),
#             'tree_model': self.tree_model,
#             'latent_dim': self.latent_dim,
#             'in_dim': self.in_dim,
#             'p_dim': self.p_dim,
#             'q_dim': self.q_dim,
#             'param_fusion': self.param_fusion,
#             'embedder_type': self._embedder_type,  # Save embedder type
#             'pointnet_hidden_dims': self._pointnet_hidden_dims,  # Save PointNet config
#             'pointbert_pretrained_path': self._pointbert_pretrained_path,  # Save Point-BERT config
#             'pointbert_freeze': self._pointbert_freeze,
#             'embedder_fitted': self.embedder_fitted,
#             'tree_fitted': self.tree_fitted,
#             'backbone_dim': backbone_dim,
#             'embedder_dropout': self._embedder_dropout,
#             'fusion_dropout': self._fusion_dropout,
#             'use_layer_norm': self._use_layer_norm,
#             'use_residual': self._use_residual,
#         }

#         with open(path, 'wb') as f:
#             pickle.dump(save_dict, f)
    
#     def load(self, path: Path):
#         """Load the hybrid model."""
#         with open(path, 'rb') as f:
#             save_dict = pickle.load(f)

#         # Restore basic parameters first
#         self.latent_dim = save_dict['latent_dim']
#         self.in_dim = save_dict['in_dim']
#         self.p_dim = save_dict['p_dim']
#         self.q_dim = save_dict['q_dim']  # Now we know q_dim
#         self.param_fusion = save_dict['param_fusion']

#         # Store construction parameters
#         self._embedder_type = save_dict.get('embedder_type', 'pointnext')  # Default to pointnext for old models
#         self._pointnet_hidden_dims = save_dict.get('pointnet_hidden_dims', [64, 128, 256])
#         self._pointbert_pretrained_path = save_dict.get('pointbert_pretrained_path', None)
#         self._pointbert_freeze = save_dict.get('pointbert_freeze', True)
#         self._backbone_dim = save_dict.get('backbone_dim', 1024)
#         self._embedder_dropout = save_dict.get('embedder_dropout', 0.1)
#         self._fusion_dropout = save_dict.get('fusion_dropout', 0.2)
#         self._use_layer_norm = save_dict.get('use_layer_norm', True)
#         self._use_residual = save_dict.get('use_residual', False)

#         # Initialize components now that we have all parameters
#         self._initialize_components(self.p_dim, self.q_dim)

#         # Now load the state dictionaries
#         self.embedder.load_state_dict(save_dict['embedder_state_dict'])
#         self.fusion_head.load_state_dict(save_dict['fusion_head_state_dict'])
#         self.regularizer.load_state_dict(save_dict['regularizer_state_dict'])
#         self.tree_model = save_dict['tree_model']

#         # Restore training state
#         self.embedder_fitted = save_dict['embedder_fitted']
#         self.tree_fitted = save_dict['tree_fitted']

#         if self.embedder_fitted and self.tree_fitted:
#             self.is_fitted = True

#         print(f"✅ Loaded hybrid {self._embedder_type} model: {self.name} (p_dim={self.p_dim}, q_dim={self.q_dim})")

