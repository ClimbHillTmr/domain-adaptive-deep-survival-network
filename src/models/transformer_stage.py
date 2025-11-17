import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]

class TransformerStageModel(nn.Module):
    def __init__(self, input_dim, num_classes=4, hidden=64, heads=4, layers=2, dropout=0.1, static_dim=0, fusion='concat'):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden*2, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=max(1, layers // 2))
        self.pe = PositionalEncoding(hidden)
        out_dim = hidden
        if fusion == 'concat' and static_dim > 0:
            out_dim = hidden + static_dim
        self.fusion = fusion
        self.classifier = nn.Sequential(nn.Linear(out_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x_seq, x_static=None):
        x = self.embedding(x_seq)
        x = self.pe(x)
        x = self.norm(x)
        x = self.encoder(x)
        pooled_mean = torch.mean(x, dim=1)
        pooled_max, _ = torch.max(x, dim=1)
        pooled = 0.5 * pooled_mean + 0.5 * pooled_max
        if x_static is not None:
            if self.fusion == 'concat':
                pooled = torch.cat([pooled, x_static], dim=-1)
            elif self.fusion == 'weighted':
                # learnable gate for static vs sequence
                gate = torch.sigmoid(nn.Linear(pooled.size(-1), 1)(pooled))
                pooled = gate * pooled + (1 - gate) * x_static
        logits = self.classifier(pooled)
        return logits

def train_transformer_stage(X_seq, y, X_static=None, epochs=16, lr=6e-4, batch_size=64, seed=42, patience=3):
    device = torch.device('cpu')
    torch.manual_seed(seed)
    y = np.asarray(y)
    num_classes = int(np.max(y) + 1)
    input_dim = X_seq.shape[-1]
    static_dim = 0 if X_static is None else X_static.shape[-1]
    model = TransformerStageModel(input_dim=input_dim, num_classes=num_classes, static_dim=static_dim, hidden=32, heads=2, layers=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    uniq, cnts = np.unique(y, return_counts=True)
    total = y.shape[0]
    cw = np.zeros(num_classes, dtype=float)
    for c in range(num_classes):
        idx = np.where(uniq == c)[0]
        n = cnts[idx][0] if idx.size > 0 else 1
        cw[c] = float(total / (num_classes * n))
    w = torch.tensor(cw, dtype=torch.float32).to(device)
    class FocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        def forward(self, logits, targets):
            ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha)
            pt = torch.exp(-ce)
            loss = (1 - pt) ** self.gamma * ce
            return loss
    criterion = FocalLoss(alpha=w, gamma=2.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32).to(device)
    def to_label(a):
        return torch.tensor(a, dtype=torch.long).to(device)
    def stratified_split(Ys, ratio=0.1):
        idx_all = np.arange(len(Ys))
        val_idx = []
        for c in range(num_classes):
            ci = np.where(Ys == c)[0]
            k = max(1, int(len(ci) * ratio))
            val_idx.extend(ci[:k])
        val_idx = np.array(sorted(set(val_idx)))
        train_mask = np.ones(len(Ys), dtype=bool)
        train_mask[val_idx] = False
        train_idx = idx_all[train_mask]
        return train_idx, val_idx
    def build_balanced_indices(Ys):
        per_class = [np.where(Ys == c)[0] for c in range(num_classes)]
        m = max(len(arr) for arr in per_class)
        out = []
        for c in range(num_classes):
            arr = per_class[c]
            if len(arr) == 0:
                continue
            rep = np.random.choice(arr, size=m, replace=True)
            out.append(rep)
        return np.concatenate(out)
    train_idx, val_idx = stratified_split(y, ratio=0.1)
    X_seq_tr = X_seq[train_idx]
    y_tr = y[train_idx]
    X_st_tr = None if X_static is None else X_static[train_idx]
    X_seq_val = X_seq[val_idx]
    y_val = y[val_idx]
    X_st_val = None if X_static is None else X_static[val_idx]
    best_val = None
    best_state = None
    patience_left = patience
    for _ in range(epochs):
        model.train()
        idx_bal = build_balanced_indices(y_tr)
        for i in range(0, len(idx_bal), batch_size):
            sel = idx_bal[i:i+batch_size]
            xb = to_tensor(X_seq_tr[sel])
            yb = to_label(y_tr[sel])
            xstb = None if X_st_tr is None else to_tensor(X_st_tr[sel])
            opt.zero_grad()
            logits = model(xb, xstb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            xv = to_tensor(X_seq_val)
            yv = to_label(y_val)
            xsv = None if X_st_val is None else to_tensor(X_st_val)
            lv = criterion(model(xv, xsv), yv).item()
        if best_val is None or lv < best_val:
            best_val = lv
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.eval()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model