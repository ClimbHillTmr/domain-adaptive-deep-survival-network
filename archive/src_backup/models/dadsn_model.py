import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


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
        # x: (B, C, L)
        weights = self.attention(x)  # (B, 1, L)
        return (x * weights).sum(dim=2)  # (B, C)


class ResNet1DEncoder(nn.Module):
    """
    1D ResNet Encoder for temporal physiological signals.
    Input: (Batch, Seq_len, Features) -> Output: (Batch, d_model)
    Requires seq_len >= 4 for meaningful temporal convolution.
    """

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
        # x: (Batch, Seq_len, Features) -> (Batch, Features, Seq_len)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.bn_input(x)
        x = self.blocks(x)
        x = self.output_proj(x)
        return self.attn_pool(x)


class DynamicFeatureEncoder(nn.Module):
    """
    Adaptive encoder for dynamic physiological signals.
    
    Automatically selects architecture based on sequence length:
    - seq_len >= 4: ResNet1D for temporal pattern extraction
    - seq_len == 1: MLP for summary statistics processing
    
    This fixes the issue where ResNet1D was being used on single-frame data,
    which wasted the temporal modeling capacity.
    """
    
    def __init__(self, num_features, d_model, seq_len=1, dropout=0.1):
        super(DynamicFeatureEncoder, self).__init__()
        self.seq_len = seq_len
        
        if seq_len >= 4:
            # Use ResNet1D for temporal sequences
            self.encoder_type = "resnet1d"
            self.resnet_encoder = ResNet1DEncoder(num_features, d_model, dropout)
        else:
            # Use MLP for summary statistics (seq_len=1)
            self.encoder_type = "mlp"
            self.mlp_encoder = nn.Sequential(
                nn.Linear(num_features, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
    
    def forward(self, x):
        if self.encoder_type == "resnet1d":
            return self.resnet_encoder(x)
        else:
            # x: (Batch, 1, Features) -> (Batch, Features)
            x = x.squeeze(1)
            return self.mlp_encoder(x)


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for static and dynamic features.
    
    Replaces Cross-Attention when seq_len=1 to avoid attention degradation.
    Uses learnable gates to control information flow from each branch.
    """
    
    def __init__(self, d_model, dropout=0.1):
        super(GatedFusion, self).__init__()
        # Gates compute importance weights for each branch
        self.gate_static = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.gate_dyn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, static_feat, dyn_feat):
        # static_feat: (B, d_model), dyn_feat: (B, d_model)
        combined = torch.cat([static_feat, dyn_feat], dim=1)
        
        # Compute gate weights
        gate_s = self.gate_static(combined)
        gate_d = self.gate_dyn(combined)
        
        # Apply gates with residual connections
        static_out = self.layer_norm(static_feat * gate_s + static_feat)
        dyn_out = self.layer_norm(dyn_feat * gate_d + dyn_feat)
        
        return static_out, dyn_out


class CrossAttentionFusion(nn.Module):
    """Cross-Attention mechanism for fusing static and dynamic features."""

    def __init__(self, d_model, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.query_static = nn.Linear(d_model, d_model)
        self.key_dyn = nn.Linear(d_model, d_model)
        self.value_dyn = nn.Linear(d_model, d_model)

        self.query_dyn = nn.Linear(d_model, d_model)
        self.key_static = nn.Linear(d_model, d_model)
        self.value_static = nn.Linear(d_model, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, static_feat, dyn_feat):
        # static_feat: (B, d_model), dyn_feat: (B, d_model)
        # Add sequence dimension for attention
        static_seq = static_feat.unsqueeze(1)  # (B, 1, d_model)
        dyn_seq = dyn_feat.unsqueeze(1)  # (B, 1, d_model)

        # Static attends to Dynamic
        q_s = self.query_static(static_seq)
        k_d = self.key_dyn(dyn_seq)
        v_d = self.value_dyn(dyn_seq)
        attn_s = torch.softmax(q_s @ k_d.transpose(-2, -1) / self.scale, dim=-1)
        attn_s = self.dropout(attn_s)
        static_enhanced = (attn_s @ v_d).squeeze(1)  # (B, d_model)

        # Dynamic attends to Static
        q_d = self.query_dyn(dyn_seq)
        k_s = self.key_static(static_seq)
        v_s = self.value_static(static_seq)
        attn_d = torch.softmax(q_d @ k_s.transpose(-2, -1) / self.scale, dim=-1)
        attn_d = self.dropout(attn_d)
        dyn_enhanced = (attn_d @ v_s).squeeze(1)  # (B, d_model)

        # Residual connection + layer norm
        static_out = self.layer_norm(static_feat + static_enhanced)
        dyn_out = self.layer_norm(dyn_feat + dyn_enhanced)

        return static_out, dyn_out


class DADSN(nn.Module):
    """
    Domain-Adaptive Deep Survival Network (DA-DSN).

    Architecture:
        - Dynamic Branch: Adaptive encoder (ResNet1D for sequences, MLP for summaries)
        - Static Branch: MLP over baseline features
        - Fusion: Gated fusion (replaces Cross-Attention for seq_len=1)
        - Output: DeepSurv head (Linear -> log-hazard ratio h(t|x))
    
    FIX: Replaced Cross-Attention with Gated Fusion to avoid attention
    degradation when seq_len=1.
    """

    def __init__(
        self, config, num_static_features, num_dynamic_features, use_transformer=True, seq_len=1
    ):
        super(DADSN, self).__init__()

        self.use_transformer = use_transformer
        transformer_cfg = (
            config.model.transformer if hasattr(config, "model") else config.transformer
        )
        d_model = transformer_cfg.d_model
        dropout = transformer_cfg.dropout

        self.d_model = d_model
        self.seq_len = seq_len

        if use_transformer:
            # Use adaptive encoder that selects architecture based on seq_len
            self.dynamic_encoder = DynamicFeatureEncoder(
                num_dynamic_features, d_model, seq_len=seq_len, dropout=dropout
            )
        else:
            self.dynamic_encoder = None

        # MLP for static features
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated fusion (replaces Cross-Attention for seq_len=1)
        # When seq_len=1, Cross-Attention degenerates to bilinear transform
        if seq_len >= 4:
            # Use Cross-Attention for true temporal sequences
            self.fusion_module = CrossAttentionFusion(d_model, dropout)
            self.fusion_type = "cross_attn"
        else:
            # Use Gated Fusion for summary statistics
            self.fusion_module = GatedFusion(d_model, dropout)
            self.fusion_type = "gated"

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.survival_head = nn.Linear(d_model, 1)

    def forward(self, x_static, x_dynamic, return_all_embs=False):
        if self.use_transformer and self.dynamic_encoder is not None:
            emb_dyn = self.dynamic_encoder(x_dynamic)
        else:
            batch_size = x_static.shape[0]
            emb_dyn = torch.zeros(batch_size, self.d_model, device=x_static.device)

        emb_static = self.static_mlp(x_static)

        # Apply fusion module
        if self.fusion_type == "cross_attn":
            emb_static_ca, emb_dyn_ca = self.fusion_module(emb_static, emb_dyn)
        else:
            emb_static_ca, emb_dyn_ca = self.fusion_module(emb_static, emb_dyn)

        emb_fuse = torch.cat([emb_dyn_ca, emb_static_ca], dim=1)
        emb_fuse = self.fusion(emb_fuse)

        log_hazard = self.survival_head(emb_fuse)

        if return_all_embs:
            return log_hazard, emb_fuse, emb_static, emb_dyn
        return log_hazard, emb_fuse


def cox_partial_log_likelihood(log_hazard, event, time):
    """
    Compute Cox Partial Log-Likelihood (negative, for minimization).
    Numerically stable implementation.

    log_hazard: (N,) - predicted log-hazard ratios
    event: (N,) - event indicator (1=event, 0=censored)
    time: (N,) - survival times
    """
    log_hazard = log_hazard.squeeze()
    if log_hazard.dim() == 0 or len(log_hazard) == 0:
        return torch.tensor(0.0, device=log_hazard.device, requires_grad=True)

    sort_idx = torch.argsort(time, descending=True)
    log_hazard_sorted = log_hazard[sort_idx]
    event_sorted = event[sort_idx]

    log_hazard_max = log_hazard_sorted.max()
    cumsum_exp = torch.cumsum(torch.exp(log_hazard_sorted - log_hazard_max), dim=0)
    log_risk = torch.log(cumsum_exp + 1e-10) + log_hazard_max

    log_partial_lik = log_hazard_sorted - log_risk

    mask = event_sorted > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=log_hazard.device, requires_grad=True)

    neg_log_lik = -log_partial_lik[mask].mean()
    return neg_log_lik


def coral_loss(source, target):
    """
    CORAL loss: Frobenius norm of covariance matrix difference.
    Aligns second-order statistics (covariance structure) between domains.

    source: (N, D) - embeddings from source domain
    target: (M, D) - embeddings from target domain
    """
    d = source.size(1)
    ns = max(source.size(0), 2)
    nt = max(target.size(0), 2)

    cs = (source.t() @ source) / (ns - 1)
    ct = (target.t() @ target) / (nt - 1)

    loss = torch.norm(cs - ct, p="fro").pow(2)
    loss = loss / (4 * d * d)
    return loss


def joint_coral_mmd_loss(source, target, coral_weight=0.5, mmd_weight=0.5,
                         kernel_mul=2.0, kernel_num=5):
    """
    Joint CORAL + MMD loss: combines second-order and distribution alignment.
    
    CORAL aligns covariance structure (feature correlations)
    MMD aligns complete distribution (including higher-order statistics)
    
    This is more powerful than either method alone because:
    - CORAL captures linear relationships between features
    - MMD captures non-linear distribution differences
    
    source: (N, D) - embeddings from source domain
    target: (M, D) - embeddings from target domain
    coral_weight: weight for CORAL component
    mmd_weight: weight for MMD component
    """
    # CORAL component
    coral_l = coral_loss(source, target)
    
    # MMD component
    mmd_l = mmd_loss(source, target, kernel_mul, kernel_num)
    
    # Combine with adaptive weights
    joint_loss = coral_weight * coral_l + mmd_weight * mmd_l
    
    return joint_loss


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5):
    """
    Maximum Mean Discrepancy (MMD) loss with RBF kernel.
    Aligns complete distributions (not just covariance like CORAL).

    source: (N, D) - embeddings from source domain
    target: (M, D) - embeddings from target domain
    kernel_mul: multiplier for bandwidth
    kernel_num: number of kernels to use
    """
    batch_size = min(source.size(0), target.size(0))
    source = source[:batch_size]
    target = target[:batch_size]

    # Compute pairwise distances
    XX = source @ source.t()
    YY = target @ target.t()
    XY = source @ target.t()
    YX = target @ source.t()

    # Compute diagonal terms for distance calculation
    X_sqnorm = torch.diag(XX)
    Y_sqnorm = torch.diag(YY)

    # RBF kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
    # Use multiple bandwidths for robustness
    dist_XX = X_sqnorm.unsqueeze(1) - 2 * XX + X_sqnorm.unsqueeze(0)
    dist_YY = Y_sqnorm.unsqueeze(1) - 2 * YY + Y_sqnorm.unsqueeze(0)
    dist_XY = X_sqnorm.unsqueeze(1) - 2 * XY + Y_sqnorm.unsqueeze(0)
    dist_YX = Y_sqnorm.unsqueeze(1) - 2 * YX + X_sqnorm.unsqueeze(0)

    # Ensure non-negative distances
    dist_XX = torch.clamp(dist_XX, min=0.0)
    dist_YY = torch.clamp(dist_YY, min=0.0)
    dist_XY = torch.clamp(dist_XY, min=0.0)
    dist_YX = torch.clamp(dist_YX, min=0.0)

    # Median heuristic for bandwidth
    dist_concat = torch.cat([dist_XX.view(-1), dist_YY.view(-1), dist_XY.view(-1)])
    bandwidth = torch.median(dist_concat)

    mmd = 0.0
    for i in range(kernel_num):
        sigma = bandwidth * (kernel_mul**i)
        sigma = max(sigma.item(), 1e-6)

        K_XX = torch.exp(-dist_XX / (2 * sigma))
        K_YY = torch.exp(-dist_YY / (2 * sigma))
        K_XY = torch.exp(-dist_XY / (2 * sigma))
        K_YX = torch.exp(-dist_YX / (2 * sigma))

        mmd += K_XX.mean() + K_YY.mean() - K_XY.mean() - K_YX.mean()

    return mmd / kernel_num


class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for DANN."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def gradient_reversal(x, alpha=1.0):
    """Apply gradient reversal to input tensor."""
    return GradientReversal.apply(x, alpha)


class DomainClassifier(nn.Module):
    """Domain classifier for DANN."""
    
    def __init__(self, d_model, dropout=0.1):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
    
    def forward(self, x):
        return self.classifier(x)


def dann_loss(source_emb, target_emb, domain_classifier, alpha=1.0):
    """
    DANN (Domain-Adversarial Neural Network) loss.
    
    Uses gradient reversal to learn domain-invariant features.
    The domain classifier tries to distinguish source from target,
    while the feature extractor tries to fool it.
    
    source_emb: (N, D) - embeddings from source domain
    target_emb: (M, D) - embeddings from target domain
    domain_classifier: nn.Module - domain classifier
    alpha: float - gradient reversal strength
    """
    # Combine source and target embeddings
    combined_emb = torch.cat([source_emb, target_emb], dim=0)
    
    # Create domain labels (0=source, 1=target)
    batch_size = combined_emb.size(0)
    source_size = source_emb.size(0)
    domain_labels = torch.zeros(batch_size, 1, device=combined_emb.device)
    domain_labels[source_size:] = 1.0
    
    # Apply gradient reversal
    reversed_emb = gradient_reversal(combined_emb, alpha)
    
    # Domain classification
    domain_pred = domain_classifier(reversed_emb)
    
    # Binary cross entropy loss
    domain_loss = nn.BCEWithLogitsLoss()(domain_pred, domain_labels)
    
    return domain_loss


def joint_mmd_loss(source, target, source_labels=None, target_labels=None, 
                   kernel_mul=2.0, kernel_num=5, conditional_weight=0.5):
    """
    Joint MMD loss: combines marginal and conditional MMD.
    
    Marginal MMD: Aligns P(X_s) and P(X_t)
    Conditional MMD: Aligns P(X_s|Y_s) and P(X_t|Y_t) using pseudo-labels
    
    source: (N, D) - embeddings from source domain
    target: (M, D) - embeddings from target domain
    source_labels: (N,) - risk scores for source (for conditional MMD)
    target_labels: (M,) - risk scores for target (for conditional MMD)
    conditional_weight: weight for conditional MMD component
    """
    # Marginal MMD
    marginal_mmd = mmd_loss(source, target, kernel_mul, kernel_num)
    
    # Conditional MMD (if labels provided)
    if source_labels is not None and target_labels is not None:
        # Discretize labels into bins for conditional alignment
        n_bins = 5
        source_bins = torch.bucketize(
            source_labels, 
            torch.linspace(source_labels.min(), source_labels.max(), n_bins - 1, device=source_labels.device)
        )
        target_bins = torch.bucketize(
            target_labels,
            torch.linspace(target_labels.min(), target_labels.max(), n_bins - 1, device=target_labels.device)
        )
        
        conditional_mmd = 0.0
        n_valid_bins = 0
        
        for bin_idx in range(n_bins):
            source_mask = source_bins == bin_idx
            target_mask = target_bins == bin_idx
            
            if source_mask.sum() > 1 and target_mask.sum() > 1:
                source_bin = source[source_mask]
                target_bin = target[target_mask]
                conditional_mmd += mmd_loss(source_bin, target_bin, kernel_mul, kernel_num)
                n_valid_bins += 1
        
        if n_valid_bins > 0:
            conditional_mmd /= n_valid_bins
        else:
            conditional_mmd = marginal_mmd  # Fallback to marginal
    else:
        conditional_mmd = marginal_mmd
    
    # Combine marginal and conditional MMD
    joint_mmd = (1 - conditional_weight) * marginal_mmd + conditional_weight * conditional_mmd
    
    return joint_mmd
