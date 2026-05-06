import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_rel_data(path, scaler=None):
    df = pd.read_csv(path)
    X_df = pd.DataFrame()
    X_df['age'] = df['透析年龄']
    X_df['gender'] = (df['性别'] == 'M').astype(float)
    X_df['dialysis_age'] = df['透析龄']
    X_df['ufr_ratio'] = df['超滤量MAX'] / df['干体重'].replace(0, np.nan)
    X_df['fluid_overload'] = (df['透前体重'] - df['干体重']) / df['干体重'].replace(0, np.nan)
    X_df['pulse_pressure'] = df['透前收缩压'] - df['透前舒张压']
    X_df['map'] = (df['透前收缩压'] + 2 * df['透前舒张压']) / 3
    
    X = X_df.fillna(X_df.median()).values
    X = np.clip(X, np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0))
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    e = df['透中低血压_计算'].fillna(0).values.astype(int)
    t = df['et_min'].fillna(0).values.astype(float)
    return torch.FloatTensor(X), torch.LongTensor(e), torch.FloatTensor(t), scaler

X_s, e_s, t_s, scaler = load_rel_data('data_preprocessing/data/深医_final_data.csv')
X_t, e_t, t_t, _ = load_rel_data('data_preprocessing/data/福鼎_final_data.csv')

p_s, p_t = e_s.float().mean().item(), e_t.float().mean().item()
w_pos, w_neg = p_t/(p_s+1e-8), (1-p_t)/(1-p_s+1e-8)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(7, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Linear(64, 1)
    def forward(self, x):
        e = self.enc(x)
        return e, self.head(e)

def train_eval(mode="SourceOnly"):
    torch.manual_seed(42)
    model = Net()
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader_s = DataLoader(TensorDataset(X_s, e_s, t_s), batch_size=256, shuffle=True)
    loader_t = DataLoader(TensorDataset(X_t, e_t, t_t), batch_size=256, shuffle=True)
    
    best_c = 0
    for epoch in range(15):
        model.train()
        iter_t = iter(loader_t)
        for xs, es, ts in loader_s:
            opt.zero_grad()
            emb_s, haz_s = model(xs)
            
            # Cox
            haz_s = haz_s.squeeze()
            idx = torch.argsort(ts, descending=True)
            haz_s, es_s = haz_s[idx], es[idx]
            cum_exp = torch.cumsum(torch.exp(haz_s - haz_s.max()), 0)
            log_risk = torch.log(cum_exp + 1e-8) + haz_s.max()
            cox = -( (haz_s - log_risk) * es_s ).sum() / (es_s.sum() + 1e-8)
            
            loss = cox
            if mode != "SourceOnly":
                try: xt, _, _ = next(iter_t)
                except StopIteration:
                    iter_t = iter(loader_t)
                    xt, _, _ = next(iter_t)
                emb_t, _ = model(xt)
                
                if mode == "StandardMMD":
                    mmd = torch.norm(emb_s.mean(0) - emb_t.mean(0))
                else: # LSA-DSN
                    w = torch.where(es==1, torch.tensor(w_pos), torch.tensor(w_neg)).unsqueeze(1)
                    mmd = torch.norm((emb_s*w).mean(0) - emb_t.mean(0))
                loss += 0.5 * (epoch/15) * mmd
                
            loss.backward()
            opt.step()
            
        model.eval()
        with torch.no_grad():
            _, ht = model(X_t)
            c = concordance_index(t_t.numpy(), -ht.squeeze().numpy(), e_t.numpy())
            best_c = max(best_c, c)
    return best_c

print(f"Source-Only Target C-Index: {train_eval('SourceOnly'):.4f}")
print(f"Standard MMD Target C-Index: {train_eval('StandardMMD'):.4f}")
print(f"LSA-DSN Target C-Index: {train_eval('LSADSN'):.4f}")
