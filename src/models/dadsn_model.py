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
    1D ResNet Encoder for dynamic physiological signals.
    Input: (Batch, Seq_len, Features) -> Output: (Batch, d_model)
    Works for both seq_len=1 (summary mode) and longer sequences.

    Upgraded:
    - 4 ResNet blocks (was 3)
    - Increased channel dimensions: 32->64->128->128
    - Attention pooling
    """

    def __init__(self, num_features, d_model, dropout=0.1):
        super(ResNet1DEncoder, self).__init__()
        self.input_proj = nn.Conv1d(num_features, 32, kernel_size=1, bias=False)
        self.bn_input = nn.BatchNorm1d(32)

        # Upgraded: 4 blocks with increased channels
        self.blocks = nn.Sequential(
            ResNet1DBlock(32, 64, kernel_size=3, dropout=dropout),
            ResNet1DBlock(64, 64, kernel_size=3, stride=1, dropout=dropout),
            ResNet1DBlock(64, 128, kernel_size=3, stride=1, dropout=dropout),
            ResNet1DBlock(128, 128, kernel_size=3, stride=1, dropout=dropout),
        )

        # Attention pooling
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
        - Dynamic Branch: ResNet1D Encoder over time-series vitals
          OR zero embedding when use_transformer=False (static-only mode)
        - Static Branch: Deeper MLP over baseline features
        - Fusion: Cross-Attention + Concatenate -> FC -> ReLU
        - Output: DeepSurv head (Linear -> log-hazard ratio h(t|x))
    """

    def __init__(
        self, config, num_static_features, num_dynamic_features, use_transformer=True
    ):
        super(DADSN, self).__init__()

        self.use_transformer = use_transformer
        transformer_cfg = (
            config.model.transformer if hasattr(config, "model") else config.transformer
        )
        d_model = transformer_cfg.d_model
        dropout = transformer_cfg.dropout

        self.d_model = d_model

        if use_transformer:
            self.dynamic_encoder = ResNet1DEncoder(
                num_dynamic_features, d_model, dropout
            )
        else:
            self.dynamic_encoder = None

        # Upgraded: Deeper MLP for static features
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Upgraded: Cross-Attention fusion
        self.cross_attn = CrossAttentionFusion(d_model, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.survival_head = nn.Linear(d_model, 1)

    def forward(self, x_static, x_dynamic):
        if self.use_transformer and self.dynamic_encoder is not None:
            emb_dyn = self.dynamic_encoder(x_dynamic)
        else:
            batch_size = x_static.shape[0]
            emb_dyn = torch.zeros(batch_size, self.d_model, device=x_static.device)

        emb_static = self.static_mlp(x_static)

        # Cross-Attention fusion
        emb_static, emb_dyn = self.cross_attn(emb_static, emb_dyn)

        emb_fuse = torch.cat([emb_dyn, emb_static], dim=1)
        emb_fuse = self.fusion(emb_fuse)

        log_hazard = self.survival_head(emb_fuse)

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
