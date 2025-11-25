import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
from abc import ABC, abstractmethod
from .metrics import ModelMetrics, compute_metrics
from .utils import get_dataset_dimensions
from sklearn.ensemble import RandomForestRegressor


class PointNetEmbedder(nn.Module):
    """
    PointNet embedder - Simpler alternative to PointNeXt.

    Better for small datasets due to fewer parameters.
    Architecture: Point-wise MLP → Global MaxPool → Feature MLP

    Reference: Qi et al., "PointNet: Deep Learning on Point Sets
    for 3D Classification and Segmentation", CVPR 2017
    """

    def __init__(
        self,
        in_dim=6,
        latent_dim=8,
        hidden_dims=[64, 128, 256],  # Point-wise MLP dimensions
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm

        # Point-wise feature extraction (shared MLP)
        point_layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            point_layers.extend([
                nn.Conv1d(prev_dim, h_dim, 1),
                nn.BatchNorm1d(h_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ])
            prev_dim = h_dim
        self.point_mlp = nn.Sequential(*point_layers)

        # Global feature dimension after max pooling
        global_dim = hidden_dims[-1]

        # Feature projection MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(global_dim, global_dim // 2),
            nn.BatchNorm1d(global_dim // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(global_dim // 2, latent_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, xyz):
        """
        Forward pass.

        Args:
            x: Point features (B, N, in_dim) - typically [xyz, normals]
            xyz: Point coordinates (B, N, 3) - not used in vanilla PointNet

        Returns:
            Global features (B, latent_dim)
        """
        # Ensure x is in the correct shape: (B, N, in_dim)
        if x.shape[-1] != self.in_dim:
            # If last dim is not in_dim, might need to transpose
            if x.shape[1] == self.in_dim:
                # Already in (B, in_dim, N) format, keep as is
                pass
            else:
                raise ValueError(f"Input shape {x.shape} doesn't match expected in_dim={self.in_dim}")
        else:
            # x: (B, N, in_dim) → (B, in_dim, N) for Conv1d
            x = x.transpose(1, 2).contiguous()

        # Point-wise feature extraction
        x = self.point_mlp(x)  # (B, hidden_dims[-1], N)

        # Global max pooling
        x = torch.max(x, dim=2)[0]  # (B, hidden_dims[-1])

        # Feature projection to latent space
        x = self.feature_mlp(x)  # (B, latent_dim)

        return x
    

class ParamFusionHead(nn.Module):
    """Enhanced fusion head with better regularization."""

    def __init__(
        self,
        z_dim: int,
        p_dim: int,
        q_dim: int,
        mode: str = "concat",
        hidden: int = 512,
        dropout: float = 0.4,
        use_batch_norm: bool = False,
        use_residual: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.use_residual = use_residual
        self.q_dim = q_dim  # Store output dimension
        
        if mode == "concat":
            in_dim = z_dim + p_dim
            
            # Enhanced MLP with better regularization
            layers = []
            
            # First layer
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden) if use_batch_norm else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(dropout),
            ])
            
            # Second layer with residual option
            mid_dim = hidden // 2
            layers.extend([
                nn.Linear(hidden, mid_dim),
                nn.BatchNorm1d(mid_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(dropout * 0.7),  # Reduce dropout towards output
            ])
            
            # Output layer with minimal dropout
            layers.extend([
                nn.Linear(mid_dim, q_dim),
            ])
            
            self.mlp = nn.Sequential(*layers)
            
            # Residual connection (if dimensions match)
            if use_residual and in_dim == q_dim:
                self.residual_proj = nn.Identity()
            elif use_residual:
                self.residual_proj = nn.Linear(in_dim, q_dim)
            else:
                self.residual_proj = None

        elif mode == "film":
            self.film_gamma = nn.Linear(p_dim, z_dim)
            self.film_beta = nn.Linear(p_dim, z_dim)
            self.out = nn.Sequential(
                nn.Linear(z_dim, hidden), nn.ReLU(True), nn.Linear(hidden, q_dim)
            )
        elif mode == "gate":
            self.gate = nn.Linear(p_dim, z_dim)
            self.out = nn.Sequential(
                nn.Linear(z_dim, hidden), nn.ReLU(True), nn.Linear(hidden, q_dim)
            )
        else:
            raise ValueError("Param fusion must be one of {'concat','film','gate'}")

    @property
    def output_dim(self) -> int:
        """Return the output dimension of the fusion head."""
        return self.q_dim

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            concat_input = torch.cat([z, p], dim=-1)
            output = self.mlp(concat_input)
            if self.residual_proj is not None:
                residual = self.residual_proj(concat_input)
                output = output + residual * 0.1  # Scale residual
                return output
            return self.mlp(torch.cat([z, p], dim=-1))
        if self.mode == "film":
            return self.out(self.film_gamma(p) * z + self.film_beta(p))
        if self.mode == "gate":
            return self.out(torch.sigmoid(self.gate(p)) * z)
        raise RuntimeError

class HybridPointCloudTreeModel:
    """Hybrid model: Point embedder + Random Forest using all parameters."""

    def __init__(
        self,
        name: str = "hybrid_pc_tree",
        # Model architecture
        in_dim: int = 6,
        latent_dim: int = 8,
        param_fusion: str = "concat",
        backbone_dim: int = 1024,
        embedder_type: str = "pointnet",  # "pointnext", "pointnet", or "pointbert"
        p_dim: int = None,  # Will be auto-detected from dataset if None
        q_dim: int = None,  # Will be auto-detected from dataset if None
        # Embedder parameters
        embedder_dropout: float = 0.1,
        fusion_dropout: float = 0.2,
        use_layer_norm: bool = True,
        use_residual: bool = False,
        # PointNet-specific parameters
        pointnet_hidden_dims: list = None,  # [64, 128, 256] default
        # PointBERT-specific parameters
        pointbert_pretrained_path: str = None,  # Path to pre-trained Point-BERT weights
        pointbert_freeze: bool = True,  # Freeze Point-BERT encoder
        # Random Forest parameters
        n_estimators: int = 200,
        max_depth: int = 15,
        min_samples_split: int = 2,
        random_state: int = 42,
        **tree_kwargs
    ):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store dimensions - will be auto-detected if None
        self.latent_dim = latent_dim
        self.in_dim = in_dim
        self.p_dim = p_dim  # Can be None, will be set during fit()
        self.q_dim = q_dim  # Will be set during fit() if None
        self.param_fusion = param_fusion

        # Store construction parameters for lazy initialization
        self._embedder_type = embedder_type.lower()
        self._backbone_dim = backbone_dim
        self._embedder_dropout = embedder_dropout
        self._fusion_dropout = fusion_dropout
        self._use_layer_norm = use_layer_norm
        self._use_residual = use_residual
        self._pointnet_hidden_dims = pointnet_hidden_dims or [64, 128, 256]
        self._pointbert_pretrained_path = pointbert_pretrained_path
        self._pointbert_freeze = pointbert_freeze

        print(f"🔧 Hybrid model configuration:")
        if p_dim is not None:
            print(f"   Input parameters: p_dim={p_dim}")
        else:
            print(f"   Input parameters will be auto-detected from dataset")
        if q_dim is not None:
            print(f"🎯 Output dimension: q_dim={q_dim}")
        else:
            print(f"🎯 Output dimension will be auto-detected from dataset")
        
        # Store tree parameters
        tree_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': random_state,
            'n_jobs': -1,
        }
        
        # Add valid sklearn RandomForest parameters from tree_kwargs
        valid_rf_params = {
            'max_features', 'min_samples_leaf', 'min_weight_fraction_leaf',
            'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
            'oob_score', 'warm_start', 'ccp_alpha', 'max_samples'
        }
        
        for key, value in tree_kwargs.items():
            if key in valid_rf_params:
                tree_params[key] = value
        
        self._tree_params = tree_params
        
        # Components will be initialized in _initialize_components()
        self.embedder = None
        self.fusion_head = None
        self.regularizer = None
        self.tree_model = None
        self.scaler = None  # ScalingPipeline instance for reproducibility

        # Training state
        self.embedder_fitted = False
        self.tree_fitted = False
        self.is_fitted = False
    
    def _initialize_components(self, p_dim: int, q_dim: int):
        """Initialize model components once p_dim and q_dim are known."""
        if self.embedder is not None:
            return  # Already initialized

        self.p_dim = p_dim
        self.q_dim = q_dim
        print(f"🎯 Initializing components with p_dim={p_dim}, q_dim={q_dim}")

        # Create embedder based on type
        if self._embedder_type == "pointnet":
            print(f"   Using PointNet embedder (simpler, fewer parameters)")
            self.embedder = PointNetEmbedder(
                in_dim=self.in_dim,
                latent_dim=self.latent_dim,
                hidden_dims=self._pointnet_hidden_dims,
                dropout=self._embedder_dropout,
                use_batch_norm=True,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown embedder type: {self._embedder_type}. Use 'pointnet', 'pointnext', or 'pointbert'.")

        # Use the same fusion head as CADQoIModel with generic p_dim
        self.fusion_head = ParamFusionHead(
            z_dim=self.latent_dim,
            p_dim=p_dim,  # Use all parameters
            q_dim=q_dim,
            mode=self.param_fusion,
            dropout=self._fusion_dropout,
            use_batch_norm=False,
            use_residual=self._use_residual,
        ).to(self.device)
        
        # Add regularizer like in CADQoIModel
        if self.latent_dim < 32:
            self.regularizer = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim * 2),
                nn.LayerNorm(self.latent_dim * 2) if self._use_layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(self._embedder_dropout),
                nn.Linear(self.latent_dim * 2, self.latent_dim),
            ).to(self.device)
        else:
            self.regularizer = nn.Identity()
        
        # Random Forest for final prediction
        self.tree_model = RandomForestRegressor(**self._tree_params)
    
    def fit(self, train_data, val_data=None, training_args=None):
        """Two-stage training: 1) CADQoI-style embedder, 2) Random Forest."""

        # Auto-detect p_dim and q_dim from dataset if not provided
        if self.p_dim is None or self.q_dim is None:
            detected_p_dim, detected_q_dim = get_dataset_dimensions(train_data)
            p_dim = self.p_dim if self.p_dim is not None else detected_p_dim
            q_dim = self.q_dim if self.q_dim is not None else detected_q_dim
            print(f"🔍 Auto-detected dimensions: p_dim={p_dim}, q_dim={q_dim}")
        else:
            p_dim = self.p_dim
            q_dim = self.q_dim

        # Initialize components now that we know p_dim and q_dim
        self._initialize_components(p_dim, q_dim)
        
        print(f"Training {self.name} in two stages...")
        
        # Stage 1: Train embedder using CADQoI-style approach
        print("Stage 1: Training CADQoI-style embedder...")
        self._fit_embedder(train_data, val_data, training_args)
        
        # Stage 2: Extract features and train Random Forest
        print("Stage 2: Training Random Forest on extracted features...")
        self._fit_tree(train_data, val_data)
        
        self.is_fitted = True
        return self
    
    def _extract_features(self, data_loader):
        """Extract features using point cloud + params from batch."""
        self.embedder.eval()
        self.fusion_head.eval()

        all_embedder_features = []
        all_fusion_features = []
        all_params = []
        all_qois = []

        with torch.no_grad():
            for batch in data_loader:
                x = batch["x"].to(self.device)
                xyz = batch["xyz"].to(self.device)
                params = batch["params"].to(self.device)  # Use params as defined in batch
                qoi = batch["qoi"]

                # Extract point cloud features (same as CADQoIModel)
                z = self.embedder(x, xyz)  # (B, latent_dim)
                z = self.regularizer(z)    # Apply regularization

                # Combine latent features with params from batch
                if self.param_fusion == "concat":
                    fusion_input = torch.cat([z, params], dim=-1)  # (B, latent_dim + p_dim)
                elif self.param_fusion == "film":
                    # For FILM, modulate with params
                    gamma = self.fusion_head.film_gamma(params)
                    beta = self.fusion_head.film_beta(params)
                    fusion_input = gamma * z + beta  # (B, latent_dim)
                elif self.param_fusion == "gate":
                    # For gating, gate with params
                    gate = torch.sigmoid(self.fusion_head.gate(params))
                    fusion_input = gate * z  # (B, latent_dim)
                else:
                    fusion_input = torch.cat([z, params], dim=-1)  # Default to concat

                all_embedder_features.append(z.cpu().numpy())
                all_fusion_features.append(fusion_input.cpu().numpy())
                all_params.append(params.cpu().numpy())
                all_qois.append(qoi.numpy())

        embedder_features = np.concatenate(all_embedder_features, axis=0)  # (N, latent_dim)
        fusion_features = np.concatenate(all_fusion_features, axis=0)      # (N, latent_dim + p_dim) or (N, latent_dim)
        params_features = np.concatenate(all_params, axis=0)               # (N, p_dim)
        qois = np.concatenate(all_qois, axis=0)                           # (N, qoi_dim)

        print(f"📊 Extracted features:")
        print(f"   Embedder features: {embedder_features.shape}")
        print(f"   Fusion features: {fusion_features.shape}")
        print(f"   Parameters: {params_features.shape}")

        # Use fusion features for Random Forest (includes point cloud + parameter interaction)
        combined_features = fusion_features

        return combined_features, qois
    
    def _fit_embedder(self, train_loader, val_loader, training_args):
        """Train the embedder using params from batch."""

        # Training parameters
        epochs = training_args.get('epochs', 50) if training_args else 50
        lr = training_args.get('lr', 1e-4) if training_args else 1e-4
        weight_decay = training_args.get('weight_decay', 1e-3) if training_args else 1e-3
        patience = training_args.get('patience', 20) if training_args else 20
        gradient_clip_norm = training_args.get('gradient_clip_norm', None) if training_args else None

        # Create a complete model like CADQoIModel for training
        model_params = list(self.embedder.parameters()) + list(self.fusion_head.parameters()) + list(self.regularizer.parameters())
        optimizer = torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float('inf')
        no_improve = 0

        print(f"  Training embedder for {epochs} epochs (using params from batch)...")

        for epoch in range(epochs):
            # Training phase
            self.embedder.train()
            self.fusion_head.train()

            train_loss = 0
            num_batches = 0

            for batch in train_loader:
                x = batch["x"].to(self.device)
                xyz = batch["xyz"].to(self.device)
                params = batch["params"].to(self.device)  # Use params as defined in batch
                qoi = batch["qoi"].to(self.device)

                optimizer.zero_grad()

                # Forward pass with params from batch
                z = self.embedder(x, xyz)
                z = self.regularizer(z)
                y_pred = self.fusion_head(z, params)  # Use params from batch

                # Compute loss
                loss = F.mse_loss(y_pred, qoi)

                loss.backward()

                # Apply gradient clipping if specified
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model_params, gradient_clip_norm)

                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches

            # Validation phase
            if val_loader is not None:
                self.embedder.eval()
                self.fusion_head.eval()

                val_loss = 0
                num_val_batches = 0

                with torch.no_grad():
                    for batch in val_loader:
                        x = batch["x"].to(self.device)
                        xyz = batch["xyz"].to(self.device)
                        params = batch["params"].to(self.device)  # Use params as defined in batch
                        qoi = batch["qoi"].to(self.device)

                        # Forward pass
                        z = self.embedder(x, xyz)
                        z = self.regularizer(z)
                        y_pred = self.fusion_head(z, params)  # Use params from batch

                        loss = F.mse_loss(y_pred, qoi)
                        val_loss += loss.item()
                        num_val_batches += 1

                val_loss /= num_val_batches

                # Update scheduler
                scheduler.step()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

                if (epoch + 1) % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={current_lr:.2e}")

        self.embedder_fitted = True
        print(f"  Embedder training completed. Best val loss: {best_val_loss:.6f}")
    
    
    def _fit_tree(self, train_loader, val_loader):
        """Train Random Forest on extracted features."""
        
        # Extract features from training data
        print("  Extracting features from training data...")
        X_train, y_train = self._extract_features(train_loader)
        
        print(f"  Feature dimensions: {X_train.shape}")
        print(f"  Feature type: {'Fusion features' if self.param_fusion == 'concat' else 'Modulated features'}")
        
        # Train Random Forest
        print("  Training Random Forest...")
        self.tree_model.fit(X_train, y_train)
        
        # Evaluate on training data
        train_pred = self.tree_model.predict(X_train)
        train_metrics = compute_metrics(y_train, train_pred)
        print(f"  Train metrics: {train_metrics}")
        
        # Evaluate on validation data if available
        if val_loader is not None:
            X_val, y_val = self._extract_features(val_loader)
            val_pred = self.tree_model.predict(X_val)
            val_metrics = compute_metrics(y_val, val_pred)
            print(f"  Val metrics: {val_metrics}")
        
        self.tree_fitted = True
    
    def predict(self, data_loader) -> np.ndarray:
        """Make predictions using the hybrid model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features using embedder
        features, _ = self._extract_features(data_loader)
        
        # Predict using Random Forest
        predictions = self.tree_model.predict(features)
        
        return predictions
    
    def evaluate(self, data_loader) -> ModelMetrics:
        """Evaluate the hybrid model."""
        # Extract features and true labels
        features, y_true = self._extract_features(data_loader)
        
        # Make predictions
        y_pred = self.tree_model.predict(features)
        
        # Compute metrics
        return compute_metrics(y_true, y_pred)
    
    def save(self, path: Path):
        """Save the hybrid model."""
        # Get backbone_dim based on embedder type
        if self._embedder_type == "pointnext":
            backbone_dim = self.embedder.backbone.head.weight.shape[0]
        elif self._embedder_type == "pointbert":
            backbone_dim = None  # Not used for Point-BERT
        else:  # pointnet
            backbone_dim = None  # Not used for PointNet

        save_dict = {
            'embedder_state_dict': self.embedder.state_dict(),
            'fusion_head_state_dict': self.fusion_head.state_dict(),
            'regularizer_state_dict': self.regularizer.state_dict(),
            'tree_model': self.tree_model,
            'scaler': self.scaler,  # Save ScalingPipeline for reproducibility
            'latent_dim': self.latent_dim,
            'in_dim': self.in_dim,
            'p_dim': self.p_dim,
            'q_dim': self.q_dim,
            'param_fusion': self.param_fusion,
            'embedder_type': self._embedder_type,  # Save embedder type
            'pointnet_hidden_dims': self._pointnet_hidden_dims,  # Save PointNet config
            'pointbert_pretrained_path': self._pointbert_pretrained_path,  # Save Point-BERT config
            'pointbert_freeze': self._pointbert_freeze,
            'embedder_fitted': self.embedder_fitted,
            'tree_fitted': self.tree_fitted,
            'backbone_dim': backbone_dim,
            'embedder_dropout': self._embedder_dropout,
            'fusion_dropout': self._fusion_dropout,
            'use_layer_norm': self._use_layer_norm,
            'use_residual': self._use_residual,
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load(self, path: Path):
        """Load the hybrid model."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        # Restore basic parameters first
        self.latent_dim = save_dict['latent_dim']
        self.in_dim = save_dict['in_dim']
        self.p_dim = save_dict['p_dim']
        self.q_dim = save_dict['q_dim']  # Now we know q_dim
        self.param_fusion = save_dict['param_fusion']

        # Store construction parameters
        self._embedder_type = save_dict.get('embedder_type', 'pointnext')  # Default to pointnext for old models
        self._pointnet_hidden_dims = save_dict.get('pointnet_hidden_dims', [64, 128, 256])
        self._pointbert_pretrained_path = save_dict.get('pointbert_pretrained_path', None)
        self._pointbert_freeze = save_dict.get('pointbert_freeze', True)
        self._backbone_dim = save_dict.get('backbone_dim', 1024)
        self._embedder_dropout = save_dict.get('embedder_dropout', 0.1)
        self._fusion_dropout = save_dict.get('fusion_dropout', 0.2)
        self._use_layer_norm = save_dict.get('use_layer_norm', True)
        self._use_residual = save_dict.get('use_residual', False)

        # Initialize components now that we have all parameters
        self._initialize_components(self.p_dim, self.q_dim)

        # Now load the state dictionaries
        self.embedder.load_state_dict(save_dict['embedder_state_dict'])
        self.fusion_head.load_state_dict(save_dict['fusion_head_state_dict'])
        self.regularizer.load_state_dict(save_dict['regularizer_state_dict'])
        self.tree_model = save_dict['tree_model']
        self.scaler = save_dict.get('scaler', None)  # Load scaler (None for backward compatibility)

        # Restore training state
        self.embedder_fitted = save_dict['embedder_fitted']
        self.tree_fitted = save_dict['tree_fitted']

        if self.embedder_fitted and self.tree_fitted:
            self.is_fitted = True

        scaler_status = "with scaler" if self.scaler is not None else "without scaler"
        print(f"✅ Loaded hybrid {self._embedder_type} model: {self.name} (p_dim={self.p_dim}, q_dim={self.q_dim}, {scaler_status})")
