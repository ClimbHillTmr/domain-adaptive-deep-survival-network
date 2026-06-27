import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

# Copy of feature names from original v11_15 script
FEATURE_NAMES = [
    # Demographics & Static
    "性别", "透析龄占比", "超负荷",
    # History Means
    "历史平均超滤率_mean", "历史平均超滤量MAX", "历史平均透前体重", 
    "历史平均透前收缩压", "历史平均透前舒张压", "历史平均透中低血压_计算",
    # Context / Treatments
    "抗凝剂类型_code", "透析方式_code", "瘘管类型_code", "瘘管位置_code",
    "透析液钙浓度", "透析液电导率", 
    # Pre-dialysis Status
    "透前体重-干体重", "透前呼吸频率", "透前体温", "透前收缩压", "透前舒张压", "透前动脉压",
    # Pre-dialysis computed hemodynamics
    "脉压差", "平均动脉压"
]

def encode_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Encode categorical features as integers"""
    if col in df.columns:
        df[col + "_code"] = df[col].astype('category').cat.codes
    return df

def build_feature_tables(
    source_path: str,
    target_path: str,
    remove_features: Optional[List[str]] = None,
    return_dataframes: bool = False,
):
    """
    Load data from files and build feature matrices.
    Returns:
        feat_source, feat_target, e_source, t_source, e_target, t_target, feature_names
    """
    df_s = pd.read_csv(source_path)
    df_t = pd.read_csv(target_path)

    # Encode categoricals
    cats = ["抗凝剂类型", "透析方式", "瘘管类型", "瘘管位置"]
    for c in cats:
        df_s = encode_categorical(df_s, c)
        df_t = encode_categorical(df_t, c)

    # Build feature lists dynamically based on actual columns
    base_cols = [c for c in FEATURE_NAMES if c in df_s.columns and c in df_t.columns]
    
    # Additional history features
    hist_cols = [c for c in df_s.columns if c.startswith("history_")]
    final_cols = base_cols + hist_cols
    
    if remove_features:
        final_cols = [c for c in final_cols if c not in remove_features]

    # Ensure no NaNs by filling with 0
    df_s[final_cols] = df_s[final_cols].fillna(0)
    df_t[final_cols] = df_t[final_cols].fillna(0)
    
    df_s = df_s.dropna(subset=['et_min', 'events'])
    df_t = df_t.dropna(subset=['et_min', 'events'])
    
    # Filter et_min <= 0 to avoid errors in time-dependent metrics
    df_s = df_s[df_s['et_min'] > 0].copy()
    df_t = df_t[df_t['et_min'] > 0].copy()

    # Standardize
    for c in final_cols:
        mean_val = df_s[c].mean()
        std_val = df_s[c].std() + 1e-8
        df_s[c] = (df_s[c] - mean_val) / std_val
        df_t[c] = (df_t[c] - mean_val) / std_val

    x_s = df_s[final_cols].values.astype(np.float32)
    x_t = df_t[final_cols].values.astype(np.float32)

    e_s = df_s["events"].values.astype(np.float32)
    t_s = df_s["et_min"].values.astype(np.float32)
    
    e_t = df_t["events"].values.astype(np.float32)
    t_t = df_t["et_min"].values.astype(np.float32)

    if return_dataframes:
        return x_s, x_t, e_s, t_s, e_t, t_t, final_cols, df_s.copy(), df_t.copy()

    return x_s, x_t, e_s, t_s, e_t, t_t, final_cols

def prepare_dataloaders(
    source_path: str, 
    target_path: str, 
    batch_size: int, 
    seed: int = 42,
    target_adapt_ratio: float = 0.2,
    target_val_ratio: float = 0.2,
    remove_features: Optional[List[str]] = None
):
    """
    End-to-end data preparation returning loaders and evaluation sets.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.evaluate.metrics import compute_ipcw_weights
    
    x_s, x_t, e_s, t_s, e_t, t_t, feature_names, df_s, df_t = build_feature_tables(
        source_path,
        target_path,
        remove_features,
        return_dataframes=True,
    )

    # Split target data
    target_indices = np.arange(len(x_t))
    x_adapt_pool, x_test, e_adapt_pool, e_test, t_adapt_pool, t_test, idx_adapt_pool, idx_test = train_test_split(
        x_t, e_t, t_t, target_indices, test_size=1 - target_adapt_ratio, random_state=seed, stratify=e_t
    )
    
    x_train, x_val, e_train, e_val, t_train, t_val, idx_train, idx_val = train_test_split(
        x_adapt_pool, e_adapt_pool, t_adapt_pool, idx_adapt_pool, test_size=target_val_ratio, random_state=seed, stratify=e_adapt_pool
    )

    # Compute IPCW
    w_s = compute_ipcw_weights(t_s, e_s)
    w_train = compute_ipcw_weights(t_train, e_train)

    # Create loaders
    ds_s = TensorDataset(torch.tensor(x_s), torch.tensor(e_s), torch.tensor(t_s), torch.tensor(w_s))
    ds_t = TensorDataset(torch.tensor(x_train), torch.tensor(e_train), torch.tensor(t_train), torch.tensor(w_train))

    source_loader = DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(ds_t, batch_size=batch_size, shuffle=True, drop_last=False)

    return {
        "source_loader": source_loader,
        "target_loader": target_loader,
        "x_val": x_val, "e_val": e_val, "t_val": t_val,
        "x_test": x_test, "e_test": e_test, "t_test": t_test,
        "x_train": x_train,
        "t_train": t_train, "e_train": e_train,
        "df_source": df_s,
        "df_target": df_t,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "feature_names": feature_names,
        "input_dim": x_s.shape[1]
    }
