import math
import torch
import torch.nn as nn

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradReverse.apply(x, coeff)

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)

class KANTokenizer(nn.Module):
    def __init__(self, num_features, d_model, basis_dim=8):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(-3.0, 3.0, basis_dim).view(1, 1, basis_dim), requires_grad=False)
        self.log_width = nn.Parameter(torch.zeros(num_features, basis_dim))
        self.basis_weight = nn.Parameter(torch.randn(num_features, basis_dim, d_model) * 0.02)
        self.base_weight = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))

    def forward(self, x):
        diff = x.unsqueeze(-1) - self.centers
        width = torch.exp(self.log_width).unsqueeze(0) + 1e-4
        basis = torch.exp(-0.5 * (diff / width) ** 2)
        spline_term = torch.einsum('bfk,fkd->bfd', basis, self.basis_weight)
        base_term = x.unsqueeze(-1) * self.base_weight.unsqueeze(0)
        return base_term + spline_term + self.bias.unsqueeze(0)

class DomainStratifiedGatedNet(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, domain_hidden, treat_indices, physio_indices, tokenizer_type="linear", kan_basis_dim=8):
        super().__init__()
        self.treat_indices = treat_indices
        self.physio_indices = physio_indices
        self.tokenizer_type = tokenizer_type
        
        if tokenizer_type == "kan":
            self.tokenizer = KANTokenizer(input_dim, d_model, kan_basis_dim)
        else:
            self.feature_weight = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
            self.feature_bias = nn.Parameter(torch.zeros(input_dim, d_model))
            
        self.feature_pos = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
        self.hazard_head = nn.Linear(d_model, 1)
        self.feature_mask = nn.Parameter(torch.ones(d_model) * 0.1)
        
        # CDAN: physio_grl (d_model) + x_treat (len(treat_indices)) + hazard (1)
        domain_input_dim = d_model + len(treat_indices) + 1
        self.domain_head = DomainClassifier(domain_input_dim, domain_hidden, dropout)

    def forward(self, x, grl_coeff=None):
        if self.tokenizer_type == "kan":
            tokens = self.tokenizer(x)
        else:
            tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
            
        tokens = tokens + self.feature_pos.unsqueeze(0)
        
        b_size = x.size(0)
        cls_tokens = self.cls_token.expand(b_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        
        encoded = self.transformer(tokens)
        
        cls_emb = self.norm(encoded[:, 0, :])
        hazard = self.hazard_head(cls_emb)
        
        domain_logits = None
        mask_l1_loss = torch.tensor(0.0, device=x.device)
        
        if grl_coeff is not None:
            physio_idx = [i + 1 for i in self.physio_indices]
            encoded_physio = encoded[:, physio_idx, :]
            
            emb_physio = self.norm(encoded_physio.mean(dim=1))
            
            mask = torch.sigmoid(self.feature_mask)
            mask_l1_loss = torch.mean(mask)
            
            masked_physio = emb_physio * mask
            physio_grl = grad_reverse(masked_physio, grl_coeff)
            
            x_treat = x[:, self.treat_indices]
            
            # CDAN (Risk-Conditioned): use sigmoid to map hazard to 0-1 for stability
            hazard_prob = torch.sigmoid(hazard.detach())
            
            domain_input = torch.cat([physio_grl, x_treat, hazard_prob], dim=-1)
            domain_logits = self.domain_head(domain_input)
            
        return cls_emb, hazard, domain_logits, mask_l1_loss
