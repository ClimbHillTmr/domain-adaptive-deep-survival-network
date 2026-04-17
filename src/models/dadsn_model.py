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


class DADSN(nn.Module):
    """
    Domain-Adaptive Deep Survival Network (DA-DSN).

    Architecture:
        - Dynamic Branch: Transformer Encoder over time-series vitals
          OR zero embedding when use_transformer=False (static-only mode)
        - Static Branch: MLP over baseline features
        - Fusion: Concatenate -> FC -> ReLU
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
        nhead = transformer_cfg.nhead
        num_layers = transformer_cfg.num_layers
        dropout = transformer_cfg.dropout

        self.d_model = d_model

        if use_transformer:
            self.dynamic_embedding = nn.Linear(num_dynamic_features, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, d_model * 4, dropout, batch_first=False
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            self.summary_encoder = None
        else:
            self.dynamic_embedding = None
            self.pos_encoder = None
            self.transformer_encoder = None
            self.summary_encoder = None

        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, d_model), nn.ReLU(), nn.Dropout(dropout)
        )

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Dropout(dropout)
        )

        self.survival_head = nn.Linear(d_model, 1)

    def forward(self, x_static, x_dynamic):
        if self.use_transformer:
            x_dyn = x_dynamic.permute(1, 0, 2)
            x_dyn = self.dynamic_embedding(x_dyn)
            x_dyn = self.pos_encoder(x_dyn)
            out_dyn = self.transformer_encoder(x_dyn)
            emb_dyn = out_dyn.mean(dim=0)
        else:
            batch_size = x_static.shape[0]
            emb_dyn = torch.zeros(batch_size, self.d_model, device=x_static.device)

        emb_static = self.static_mlp(x_static)

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
