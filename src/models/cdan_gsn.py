import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BSplineKANTokenizer(nn.Module):
    """
    True KAN tokenizer using B-spline basis functions (Liu et al. 2024,
    "KAN: Kolmogorov-Arnold Networks").

    Each input feature x_f is mapped to a d_model-dimensional token via:

        φ_f(x_f) = w_base_f * SiLU(x_f)  +  Σ_i  c_{f,i} * B_{i,k}(x_f)

    where B_{i,k} are B-spline basis functions of order k (default: cubic, k=3)
    defined on a uniform grid of G intervals over [grid_lo, grid_hi].

    The number of basis functions per feature is G + k (= grid_size + spline_order).
    After the spline expansion the output is projected to d_model dimensions via a
    learnable weight matrix (per-feature), i.e., the coefficient tensor already has
    shape (num_features, num_basis, d_model) so no extra Linear is needed.

    Args:
        num_features : number of input scalar features
        d_model      : output embedding dimension
        grid_size    : G, number of grid intervals (default 8)
        spline_order : k, B-spline order (3 = cubic)
        grid_range   : (lo, hi) input range for the knot grid (default: -4, 4)
    """

    def __init__(
        self,
        num_features: int,
        d_model: int,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_range: tuple = (-4.0, 4.0),
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.grid_size = grid_size
        self.spline_order = spline_order
        # Number of basis functions: G + k
        self.num_basis = grid_size + spline_order

        # Build extended uniform knot grid: G + 2k + 1 points total.
        # The k extra knots on each side ensure that all x in [grid_lo, grid_hi]
        # have full B-spline support (no boundary truncation).
        grid_lo, grid_hi = grid_range
        h = (grid_hi - grid_lo) / grid_size
        grid = torch.linspace(
            grid_lo - spline_order * h,
            grid_hi + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        # Registered as buffer (not a parameter — grid is fixed).
        self.register_buffer("grid", grid)  # shape: (G + 2k + 1,)

        # Learnable spline coefficients projected to d_model:
        #   c_weight[f, i, d]  →  contribution of basis B_{i,k} for feature f to dim d
        self.spline_weight = nn.Parameter(
            torch.randn(num_features, self.num_basis, d_model) * 0.02
        )
        # Learnable weight for the SiLU residual base term (per feature, per dim)
        self.base_weight = nn.Parameter(
            torch.randn(num_features, d_model) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))

    # ------------------------------------------------------------------
    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis values B_{i,k}(x) for all features and samples.

        Uses the Cox–de Boor recursive definition:
          B_{i,0}(x) = 1  if grid[i] ≤ x < grid[i+1],  else 0
          B_{i,p}(x) = (x − t_i)/(t_{i+p} − t_i) * B_{i,p−1}(x)
                     + (t_{i+p+1} − x)/(t_{i+p+1} − t_{i+1}) * B_{i+1,p−1}(x)

        Args:
            x : (batch_size, num_features)  — already standardised inputs

        Returns:
            basis : (batch_size, num_features, num_basis)
        """
        # Clamp x to the interior of the grid to keep all values in valid range.
        grid_lo = self.grid[self.spline_order].item()
        grid_hi = self.grid[-(self.spline_order + 1)].item()
        x_clamped = x.detach().clamp(grid_lo, grid_hi)
        # Use clamped for basis selection but keep gradient path through x
        x_for_basis = x + (x_clamped - x).detach()  # straight-through for clamping

        # (B, F, 1) to broadcast against the 1-D grid
        xb = x_for_basis.unsqueeze(-1)  # (B, F, 1)
        g = self.grid  # (G + 2k + 1,)

        # Order-0 indicator basis: shape (B, F, G + 2k)
        basis = ((xb >= g[:-1]) & (xb < g[1:])).float()
        # Include right endpoint in the last interval to avoid zero basis at x=grid_hi
        basis[..., -1] = torch.logical_or(basis[..., -1] > 0, (xb[..., 0] >= g[-2])).float()

        # Recursive refinement from order 1 to k
        for p in range(1, self.spline_order + 1):
            # At this iteration the output will have one fewer basis function.
            n_out = basis.shape[-1] - 1  # = G + 2k - p

            # Knot segments accessed at recursion order p:
            #   left denominator  : t_{i+p} − t_i
            #   right denominator : t_{i+p+1} − t_{i+1}
            t_i   = g[:n_out]           # (n_out,)
            t_ip  = g[p: p + n_out]     # (n_out,)
            t_ip1 = g[p + 1: p + 1 + n_out]  # (n_out,)
            t_i1  = g[1: 1 + n_out]    # (n_out,)

            denom_l = t_ip - t_i        # (n_out,) — can be 0 for repeated knots
            denom_r = t_ip1 - t_i1     # (n_out,)

            # Safe division: 0/0 → 0 (standard B-spline convention)
            safe_l = torch.where(denom_l.abs() < 1e-9, torch.ones_like(denom_l), denom_l)
            safe_r = torch.where(denom_r.abs() < 1e-9, torch.ones_like(denom_r), denom_r)

            # Broadcast: (B, F, 1) − (n_out,) → (B, F, n_out)
            alpha_l = (xb - t_i) / safe_l
            alpha_r = (t_ip1 - xb) / safe_r

            # Zero out terms where denominator is 0 (repeated knot)
            alpha_l = torch.where(
                denom_l.abs() < 1e-9, torch.zeros_like(alpha_l), alpha_l
            )
            alpha_r = torch.where(
                denom_r.abs() < 1e-9, torch.zeros_like(alpha_r), alpha_r
            )

            basis = alpha_l * basis[..., :n_out] + alpha_r * basis[..., 1:]

        return basis  # (B, F, num_basis)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x      : (batch_size, num_features)
        Returns:
            tokens : (batch_size, num_features, d_model)
        """
        # 1. B-spline component
        basis = self._b_splines(x)  # (B, F, num_basis)
        spline_out = torch.einsum("bfk,fkd->bfd", basis, self.spline_weight)

        # 2. SiLU residual base term (per-feature linear-in-SiLU projection)
        base_out = F.silu(x).unsqueeze(-1) * self.base_weight.unsqueeze(0)  # (B, F, d_model)

        return base_out + spline_out + self.bias.unsqueeze(0)


# ---------------------------------------------------------------------------
# Legacy alias kept for backward compatibility of checkpoint loading.
# New code should use BSplineKANTokenizer directly.
# ---------------------------------------------------------------------------
KANTokenizer = BSplineKANTokenizer

class DomainStratifiedGatedNet(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        dropout,
        domain_hidden,
        treat_indices,
        physio_indices,
        tokenizer_type="linear",
        kan_basis_dim=8,          # grid_size G for B-spline KAN (backward-compat alias)
        kan_spline_order=3,       # spline order k (cubic by default)
        kan_grid_range=(-4.0, 4.0),
    ):
        super().__init__()
        self.treat_indices = treat_indices
        self.physio_indices = physio_indices
        self.tokenizer_type = tokenizer_type

        if tokenizer_type == "kan":
            # True B-spline KAN tokenizer (Liu et al. 2024)
            self.tokenizer = BSplineKANTokenizer(
                num_features=input_dim,
                d_model=d_model,
                grid_size=kan_basis_dim,
                spline_order=kan_spline_order,
                grid_range=kan_grid_range,
            )
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
