"""
v6.0 Model Architecture
========================
Key improvements over v5.x:
1. DeepHit survival model (handles non-PH data)
2. MCD (Maximum Classifier Discrepancy) domain adaptation
3. Previous session features integration
4. Historical IDH frequency features

Author: Clinical Data Analytics Specialist
Date: 2026-05-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


# ============================================================
# DeepHit Model
# ============================================================
class DeepHitNetwork(nn.Module):
    """
    DeepHit: Deep Neural Network for Survival Analysis.
    
    Unlike Cox PH which assumes proportional hazards, DeepHit:
    1. Models the probability distribution over discrete time intervals
    2. Uses a multi-task loss: likelihood + ranking
    3. Handles non-proportional hazards naturally
    
    Reference: Lee, Changhee, et al. "DeepHit: A Deep Learning Approach to 
    Survival Analysis with Competing Risks." AAAI 2018.
    """
    
    def __init__(
        self,
        num_static_features: int,
        num_dynamic_features: int,
        d_model: int = 64,
        num_time_bins: int = 10,
        dropout: float = 0.1,
        seq_len: int = 1,
    ):
        super(DeepHitNetwork, self).__init__()
        
        self.num_time_bins = num_time_bins
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Dynamic feature encoder
        if seq_len >= 4:
            self.dynamic_encoder = ResNet1DEncoder(
                num_dynamic_features, d_model, dropout
            )
        else:
            self.dynamic_encoder = nn.Sequential(
                nn.Linear(num_dynamic_features, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        
        # Static feature encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(num_static_features, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Shared representation layers
        self.shared_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Output layer: probability for each time bin
        # P(event at time t | survived until t-1)
        self.output_layer = nn.Linear(d_model, num_time_bins)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_static: torch.Tensor, x_dynamic: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_static: (B, num_static_features)
            x_dynamic: (B, seq_len, num_dynamic_features) or (B, 1, num_dynamic_features)
        
        Returns:
            y_hat: (B, num_time_bins) - probability distribution over time bins
        """
        # Encode dynamic features
        if self.seq_len >= 4:
            emb_dyn = self.dynamic_encoder(x_dynamic)
        else:
            emb_dyn = self.dynamic_encoder(x_dynamic.squeeze(1))
        
        # Encode static features
        emb_static = self.static_encoder(x_static)
        
        # Fuse features
        emb_combined = torch.cat([emb_static, emb_dyn], dim=1)
        emb_fused = self.fusion(emb_combined)
        
        # Shared representation
        shared_repr = self.shared_layers(emb_fused)
        
        # Output: probability distribution over time bins
        y_hat = self.output_layer(shared_repr)
        
        # Apply softmax to get valid probability distribution
        y_hat = F.softmax(y_hat, dim=1)
        
        return y_hat


class DeepHitLoss(nn.Module):
    """
    DeepHit Loss Function.
    
    Combines:
    1. Log-likelihood loss: maximize probability of observed events
    2. Ranking loss: ensure correct ordering of risk scores
    """
    
    def __init__(self, alpha: float = 0.5, sigma: float = 0.1, num_time_bins: int = 10):
        """
        Args:
            alpha: weight for log-likelihood vs ranking loss
            sigma: bandwidth for ranking loss kernel
            num_time_bins: number of time bins
        """
        super(DeepHitLoss, self).__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.num_time_bins = num_time_bins
    
    def forward(
        self,
        y_hat: torch.Tensor,
        event: torch.Tensor,
        duration: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DeepHit loss.
        
        Args:
            y_hat: (B, num_time_bins) - predicted probability distribution
            event: (B,) - event indicator (1=event, 0=censored)
            duration: (B,) - survival times
            time_bins: (num_time_bins,) - time bin boundaries
        
        Returns:
            loss: scalar loss value
        """
        # Log-likelihood loss
        log_lik_loss = self._log_likelihood_loss(y_hat, event, duration, time_bins)
        
        # Ranking loss
        rank_loss = self._ranking_loss(y_hat, event, duration, time_bins)
        
        # Combined loss
        loss = self.alpha * log_lik_loss + (1 - self.alpha) * rank_loss
        
        return loss
    
    def _log_likelihood_loss(
        self,
        y_hat: torch.Tensor,
        event: torch.Tensor,
        duration: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log-likelihood loss.
        
        For events: maximize P(T = t_i | x)
        For censored: maximize P(T > t_i | x)
        """
        batch_size = y_hat.size(0)
        eps = 1e-8
        
        # Find time bin index for each sample
        time_bin_idx = torch.searchsorted(time_bins, duration)
        time_bin_idx = torch.clamp(time_bin_idx, 0, self.num_time_bins - 1)
        
        # For events: log P(T = t_i)
        event_mask = event.float()
        event_prob = torch.gather(y_hat, 1, time_bin_idx.unsqueeze(1)).squeeze(1)
        event_loss = -torch.log(event_prob + eps) * event_mask
        
        # For censored: log P(T > t_i) = sum of probabilities after t_i
        censored_mask = 1 - event_mask
        cumulative_prob = torch.zeros(batch_size, device=y_hat.device)
        for i in range(batch_size):
            if censored_mask[i] > 0:
                idx = time_bin_idx[i]
                cumulative_prob[i] = y_hat[i, idx:].sum()
        
        censored_loss = -torch.log(cumulative_prob + eps) * censored_mask
        
        # Total log-likelihood loss
        log_lik_loss = (event_loss + censored_loss).mean()
        
        return log_lik_loss
    
    def _ranking_loss(
        self,
        y_hat: torch.Tensor,
        event: torch.Tensor,
        duration: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ranking loss.
        
        Ensure that patients with earlier events have higher risk scores.
        """
        # Compute risk scores (expected time to event)
        time_indices = torch.arange(self.num_time_bins, device=y_hat.device).float()
        risk_scores = (y_hat * time_indices).sum(dim=1)  # Expected time
        
        # Create pairs: (i, j) where i has event before j
        event_mask = event.float()
        event_indices = torch.where(event_mask > 0)[0]
        
        if len(event_indices) < 2:
            return torch.tensor(0.0, device=y_hat.device)
        
        # Sample pairs for efficiency
        n_pairs = min(1000, len(event_indices) * (len(event_indices) - 1) // 2)
        idx_i = torch.randint(0, len(event_indices), (n_pairs,), device=y_hat.device)
        idx_j = torch.randint(0, len(event_indices), (n_pairs,), device=y_hat.device)
        
        # Ensure i != j
        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]
        
        if len(idx_i) == 0:
            return torch.tensor(0.0, device=y_hat.device)
        
        # Get actual event times
        event_times_i = duration[event_indices[idx_i]]
        event_times_j = duration[event_indices[idx_j]]
        
        # Pairs where i has event before j
        valid_pairs = event_times_i < event_times_j
        
        if valid_pairs.sum() == 0:
            return torch.tensor(0.0, device=y_hat.device)
        
        # Ranking loss: risk_i should be > risk_j (earlier event = higher risk)
        risk_i = risk_scores[event_indices[idx_i]]
        risk_j = risk_scores[event_indices[idx_j]]
        
        # Lower risk score = higher risk (earlier expected event time)
        ranking_diff = risk_i - risk_j
        ranking_loss = torch.exp(-ranking_diff[valid_pairs] / self.sigma).mean()
        
        return ranking_loss


# ============================================================
# MCD (Maximum Classifier Discrepancy) Domain Adaptation
# ============================================================
class FeatureExtractor(nn.Module):
    """Shared feature extractor for MCD."""
    
    def __init__(
        self,
        num_static_features: int,
        num_dynamic_features: int,
        d_model: int = 64,
        dropout: float = 0.1,
        seq_len: int = 1,
    ):
        super(FeatureExtractor, self).__init__()
        
        self.seq_len = seq_len
        
        # Dynamic feature encoder
        if seq_len >= 4:
            self.dynamic_encoder = ResNet1DEncoder(
                num_dynamic_features, d_model, dropout
            )
        else:
            self.dynamic_encoder = nn.Sequential(
                nn.Linear(num_dynamic_features, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        
        # Static feature encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(num_static_features, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x_static: torch.Tensor, x_dynamic: torch.Tensor) -> torch.Tensor:
        if self.seq_len >= 4:
            emb_dyn = self.dynamic_encoder(x_dynamic)
        else:
            emb_dyn = self.dynamic_encoder(x_dynamic.squeeze(1))
        
        emb_static = self.static_encoder(x_static)
        emb_combined = torch.cat([emb_static, emb_dyn], dim=1)
        return self.fusion(emb_combined)


class Classifier1(nn.Module):
    """First classifier for MCD."""
    
    def __init__(self, d_model: int, num_classes: int = 2, dropout: float = 0.1):
        super(Classifier1, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class Classifier2(nn.Module):
    """Second classifier for MCD (different initialization)."""
    
    def __init__(self, d_model: int, num_classes: int = 2, dropout: float = 0.1):
        super(Classifier2, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MCDDomainAdaptation(nn.Module):
    """
    Maximum Classifier Discrepancy (MCD) for Domain Adaptation.
    
    MCD uses two classifiers with different decision boundaries.
    The discrepancy between their predictions on target domain samples
    is used to detect distribution shift.
    
    Training procedure:
    Step 1: Train feature extractor + classifiers on source domain
    Step 2: Maximize discrepancy on target domain (train classifiers)
    Step 3: Minimize discrepancy on target domain (train feature extractor)
    
    Reference: Saito, Kuniaki, et al. "Maximum Classifier Discrepancy for 
    Unsupervised Domain Adaptation." CVPR 2018.
    """
    
    def __init__(
        self,
        num_static_features: int,
        num_dynamic_features: int,
        d_model: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1,
        seq_len: int = 1,
    ):
        super(MCDDomainAdaptation, self).__init__()
        
        self.feature_extractor = FeatureExtractor(
            num_static_features, num_dynamic_features, d_model, dropout, seq_len
        )
        self.classifier1 = Classifier1(d_model, num_classes, dropout)
        self.classifier2 = Classifier2(d_model, num_classes, dropout)
    
    def forward(
        self,
        x_static_src: torch.Tensor,
        x_dynamic_src: torch.Tensor,
        x_static_tgt: Optional[torch.Tensor] = None,
        x_dynamic_tgt: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.
        
        Returns:
            Dictionary containing:
            - pred1_src: predictions from classifier 1 on source
            - pred2_src: predictions from classifier 2 on source
            - pred1_tgt: predictions from classifier 1 on target (if provided)
            - pred2_tgt: predictions from classifier 2 on target (if provided)
            - feat_src: features for source domain
            - feat_tgt: features for target domain (if provided)
        """
        # Extract features
        feat_src = self.feature_extractor(x_static_src, x_dynamic_src)
        pred1_src = self.classifier1(feat_src)
        pred2_src = self.classifier2(feat_src)
        
        result = {
            "pred1_src": pred1_src,
            "pred2_src": pred2_src,
            "feat_src": feat_src,
        }
        
        if x_static_tgt is not None and x_dynamic_tgt is not None:
            feat_tgt = self.feature_extractor(x_static_tgt, x_dynamic_tgt)
            pred1_tgt = self.classifier1(feat_tgt)
            pred2_tgt = self.classifier2(feat_tgt)
            
            result.update({
                "pred1_tgt": pred1_tgt,
                "pred2_tgt": pred2_tgt,
                "feat_tgt": feat_tgt,
            })
        
        return result
    
    def compute_mcd_loss(
        self,
        pred1_tgt: torch.Tensor,
        pred2_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Maximum Classifier Discrepancy loss.
        
        Discrepancy = L1 distance between classifier predictions
        """
        # Softmax to get probabilities
        prob1 = F.softmax(pred1_tgt, dim=1)
        prob2 = F.softmax(pred2_tgt, dim=1)
        
        # L1 distance between predictions
        discrepancy = torch.abs(prob1 - prob2).mean()
        
        return discrepancy
    
    def compute_classification_loss(
        self,
        pred1_src: torch.Tensor,
        pred2_src: torch.Tensor,
        labels_src: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute classification loss for both classifiers on source domain.
        """
        loss1 = F.cross_entropy(pred1_src, labels_src)
        loss2 = F.cross_entropy(pred2_src, labels_src)
        return loss1, loss2


# ============================================================
# ResNet1D Components (reused from v5.x)
# ============================================================
class ResNet1DBlock(nn.Module):
    """1D Residual Block for temporal feature extraction."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super(ResNet1DBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
    
    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class AttentionPool1d(nn.Module):
    """Attention-based pooling for 1D sequences."""
    
    def __init__(self, in_channels):
        super(AttentionPool1d, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1), nn.Softmax(dim=2)
        )
    
    def forward(self, x):
        weights = self.attention(x)
        return (x * weights).sum(dim=2)


class ResNet1DEncoder(nn.Module):
    """1D ResNet Encoder for temporal physiological signals."""
    
    def __init__(self, num_features, d_model, dropout=0.1):
        super(ResNet1DEncoder, self).__init__()
        self.input_proj = nn.Conv1d(num_features, 32, kernel_size=1, bias=False)
        self.bn_input = nn.BatchNorm1d(32)
        
        self.blocks = nn.Sequential(
            ResNet1DBlock(32, 64, kernel_size=3, dropout=dropout),
            ResNet1DBlock(64, 64, kernel_size=3, stride=1, dropout=dropout),
            ResNet1DBlock(64, 128, kernel_size=3, stride=1, dropout=dropout),
            ResNet1DBlock(128, 128, kernel_size=3, stride=1, dropout=dropout),
        )
        
        self.output_proj = nn.Sequential(
            nn.Conv1d(128, d_model, kernel_size=1),
        )
        self.attn_pool = AttentionPool1d(d_model)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.bn_input(x)
        x = self.blocks(x)
        x = self.output_proj(x)
        return self.attn_pool(x)


# ============================================================
# v6.0 Combined Model: DeepHit + MCD
# ============================================================
class V6DeepHitMCD(nn.Module):
    """
    v6.0 Combined Model: DeepHit + MCD Domain Adaptation.
    
    Architecture:
    1. Feature Extractor: Shared encoder for static + dynamic features
    2. DeepHit Head: Survival prediction with discrete time bins
    3. MCD Classifiers: Two classifiers for domain adaptation
    
    Key improvements:
    - Handles non-proportional hazards (DeepHit)
    - Better domain alignment (MCD vs CORAL/MMD)
    - Uses previous session features (fixes data leakage)
    - Includes historical IDH frequency
    """
    
    def __init__(
        self,
        num_static_features: int,
        num_dynamic_features: int,
        d_model: int = 64,
        num_time_bins: int = 10,
        dropout: float = 0.1,
        seq_len: int = 1,
    ):
        super(V6DeepHitMCD, self).__init__()
        
        self.num_time_bins = num_time_bins
        self.seq_len = seq_len
        
        # Shared feature extractor
        self.feature_extractor = FeatureExtractor(
            num_static_features, num_dynamic_features, d_model, dropout, seq_len
        )
        
        # DeepHit survival head
        self.shared_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.survival_head = nn.Linear(d_model, num_time_bins)
        
        # MCD classifiers for domain adaptation
        self.mcd_classifier1 = Classifier1(d_model, num_classes=2, dropout=dropout)
        self.mcd_classifier2 = Classifier2(d_model, num_classes=2, dropout=dropout)
    
    def forward(
        self,
        x_static: torch.Tensor,
        x_dynamic: torch.Tensor,
        x_static_tgt: Optional[torch.Tensor] = None,
        x_dynamic_tgt: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.
        
        Returns:
            Dictionary containing:
            - survival_pred: DeepHit survival predictions
            - mcd_pred1_src: MCD classifier 1 predictions (source)
            - mcd_pred2_src: MCD classifier 2 predictions (source)
            - mcd_pred1_tgt: MCD classifier 1 predictions (target, if provided)
            - mcd_pred2_tgt: MCD classifier 2 predictions (target, if provided)
            - features: extracted features
        """
        # Extract features (source domain)
        features_src = self.feature_extractor(x_static, x_dynamic)
        
        # DeepHit survival prediction
        shared_repr = self.shared_layers(features_src)
        survival_pred = self.survival_head(shared_repr)
        survival_pred = F.softmax(survival_pred, dim=1)
        
        # MCD classifiers (source)
        mcd_pred1_src = self.mcd_classifier1(features_src)
        mcd_pred2_src = self.mcd_classifier2(features_src)
        
        result = {
            "survival_pred": survival_pred,
            "mcd_pred1_src": mcd_pred1_src,
            "mcd_pred2_src": mcd_pred2_src,
            "features_src": features_src,
        }
        
        # Target domain (if provided)
        if x_static_tgt is not None and x_dynamic_tgt is not None:
            features_tgt = self.feature_extractor(x_static_tgt, x_dynamic_tgt)
            mcd_pred1_tgt = self.mcd_classifier1(features_tgt)
            mcd_pred2_tgt = self.mcd_classifier2(features_tgt)
            
            result.update({
                "mcd_pred1_tgt": mcd_pred1_tgt,
                "mcd_pred2_tgt": mcd_pred2_tgt,
                "features_tgt": features_tgt,
            })
        
        return result
    
    def compute_loss(
        self,
        outputs: dict,
        event: torch.Tensor,
        duration: torch.Tensor,
        time_bins: torch.Tensor,
        labels_src: Optional[torch.Tensor] = None,
        mcd_weight: float = 0.5,
        alpha: float = 0.5,
        sigma: float = 0.1,
    ) -> dict:
        """
        Compute combined loss: DeepHit + MCD.
        
        Args:
            outputs: forward() output dictionary
            event: event indicator
            duration: survival times
            time_bins: time bin boundaries
            labels_src: source domain labels for MCD
            mcd_weight: weight for MCD loss
            alpha: DeepHit log-likelihood vs ranking weight
            sigma: ranking loss bandwidth
        
        Returns:
            Dictionary containing individual loss components and total loss
        """
        # DeepHit loss
        deephit_loss_fn = DeepHitLoss(alpha=alpha, sigma=sigma, num_time_bins=time_bins.shape[0]-1)
        survival_loss = deephit_loss_fn(
            outputs["survival_pred"], event, duration, time_bins
        )
        
        # MCD loss (if target domain provided)
        mcd_loss = torch.tensor(0.0, device=survival_loss.device)
        if "mcd_pred1_tgt" in outputs and "mcd_pred2_tgt" in outputs:
            mcd_loss = self._compute_mcd_loss(
                outputs["mcd_pred1_src"],
                outputs["mcd_pred2_src"],
                outputs["mcd_pred1_tgt"],
                outputs["mcd_pred2_tgt"],
                labels_src,
            )
        
        # Total loss
        total_loss = survival_loss + mcd_weight * mcd_loss
        
        return {
            "total_loss": total_loss,
            "survival_loss": survival_loss,
            "mcd_loss": mcd_loss,
        }
    
    def _compute_mcd_loss(
        self,
        pred1_src: torch.Tensor,
        pred2_src: torch.Tensor,
        pred1_tgt: torch.Tensor,
        pred2_tgt: torch.Tensor,
        labels_src: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MCD loss.
        
        Step 1: Classification loss on source (both classifiers)
        Step 2: Maximize discrepancy on target
        """
        # Classification loss on source
        cls_loss1 = F.cross_entropy(pred1_src, labels_src)
        cls_loss2 = F.cross_entropy(pred2_src, labels_src)
        cls_loss = (cls_loss1 + cls_loss2) / 2
        
        # Discrepancy on target
        prob1_tgt = F.softmax(pred1_tgt, dim=1)
        prob2_tgt = F.softmax(pred2_tgt, dim=1)
        discrepancy = torch.abs(prob1_tgt - prob2_tgt).mean()
        
        # Combined MCD loss
        mcd_loss = cls_loss - discrepancy
        
        return mcd_loss
