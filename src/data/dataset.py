import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
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

CATEGORICAL_COLS = ["抗凝剂类型", "透析方式", "瘘管类型", "瘘管位置"]
FEATURE_ALLOWLIST_PATH = "experiments/audit/feature_allowlist.csv"


def _safe_str_series(series: pd.Series) -> pd.Series:
    return series.fillna("__MISSING__").astype(str).str.strip().replace("", "__MISSING__")


def encode_categoricals_from_source(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Fit categorical mappings on the source cohort and apply the same mappings to target.
    Target-only categories are encoded as an explicit unknown level instead of being
    assigned center-specific integer meanings.
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    mappings: Dict[str, Dict[str, int]] = {}
    df_source = df_source.copy()
    df_target = df_target.copy()

    for col in categorical_cols:
        if col not in df_source.columns or col not in df_target.columns:
            continue

        source_values = _safe_str_series(df_source[col])
        categories = sorted(source_values.unique().tolist())
        mapping = {value: idx for idx, value in enumerate(categories)}
        unknown_code = len(mapping)

        df_source[col + "_code"] = source_values.map(mapping).fillna(unknown_code).astype(int)
        df_target[col + "_code"] = (
            _safe_str_series(df_target[col]).map(mapping).fillna(unknown_code).astype(int)
        )
        mappings[col] = mapping

    return df_source, df_target, mappings

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

    for label, frame in [("source", df_s), ("target", df_t)]:
        zero_time_non_events = (frame["events"].eq(0) & frame["et_min"].le(0)).sum()
        if zero_time_non_events:
            raise ValueError(
                f"{label} data contains {zero_time_non_events} non-events with et_min <= 0. "
                "This is the new binary-IDH data contract, not a valid Cox follow-up time."
            )

    # Encode categoricals using source-fitted mappings to avoid cross-center code drift.
    df_s, df_t, category_mappings = encode_categoricals_from_source(df_s, df_t)

    if not os.path.exists(FEATURE_ALLOWLIST_PATH):
        raise FileNotFoundError(
            f"Missing feature allowlist: {FEATURE_ALLOWLIST_PATH}. "
            "Create it before training to freeze prediction-time features."
        )
    allowlist = pd.read_csv(FEATURE_ALLOWLIST_PATH)
    final_cols = allowlist.loc[allowlist["allowed"].astype(str).str.lower() == "yes", "feature"].tolist()
    final_cols = [c for c in final_cols if c in df_s.columns and c in df_t.columns]
    
    if remove_features:
        final_cols = [c for c in final_cols if c not in remove_features]

    if not final_cols:
        raise ValueError("No usable features after applying feature allowlist.")

    df_s = df_s.dropna(subset=['et_min', 'events'])
    df_t = df_t.dropna(subset=['et_min', 'events'])
    
    # Filter et_min <= 0 to avoid errors in time-dependent metrics
    df_s = df_s[df_s['et_min'] > 0].copy()
    df_t = df_t[df_t['et_min'] > 0].copy()

    # This helper is used by audits. Training uses the split-aware transform below.
    # Do not use target rows to estimate preprocessing parameters here.
    for c in final_cols:
        df_s[c] = pd.to_numeric(df_s[c], errors="coerce")
        df_t[c] = pd.to_numeric(df_t[c], errors="coerce")
        median = df_s[c].median()
        df_s[c] = df_s[c].fillna(median)
        df_t[c] = df_t[c].fillna(median)
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
        return (
            x_s,
            x_t,
            e_s,
            t_s,
            e_t,
            t_t,
            final_cols,
            df_s.copy(),
            df_t.copy(),
            category_mappings,
        )

    return x_s, x_t, e_s, t_s, e_t, t_t, final_cols


def _load_and_filter_cohort(path: str) -> pd.DataFrame:
    """Load one cohort and retain rows with a valid positive follow-up time."""
    frame = pd.read_csv(path).dropna(subset=["et_min", "events"]).copy()
    zero_time_non_events = (frame["events"].eq(0) & frame["et_min"].le(0)).sum()
    if zero_time_non_events:
        raise ValueError(
            f"{path} uses zero event times for {zero_time_non_events} non-events. "
            "The legacy Cox pipeline is disabled for the binary-IDH data contract."
        )
    frame = frame[frame["et_min"] > 0].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No valid rows remain after outcome filtering: {path}")
    return frame


def _allowed_features(df_source: pd.DataFrame, df_target: pd.DataFrame, remove_features=None):
    if not os.path.exists(FEATURE_ALLOWLIST_PATH):
        raise FileNotFoundError(f"Missing feature allowlist: {FEATURE_ALLOWLIST_PATH}")
    allowlist = pd.read_csv(FEATURE_ALLOWLIST_PATH)
    features = allowlist.loc[
        allowlist["allowed"].astype(str).str.lower() == "yes", "feature"
    ].tolist()
    features = [name for name in features if name in df_source and name in df_target]
    if remove_features:
        features = [name for name in features if name not in remove_features]
    if not features:
        raise ValueError("No usable features after applying feature allowlist.")
    return features


def _apply_source_fitted_categoricals(
    source_train: pd.DataFrame,
    *frames: pd.DataFrame,
) -> Tuple[pd.DataFrame, ...]:
    """Fit categorical codes on source training patients and apply unchanged."""
    mappings = {}
    for column in CATEGORICAL_COLS:
        if column not in source_train.columns:
            continue
        values = _safe_str_series(source_train[column])
        mappings[column] = {value: index for index, value in enumerate(sorted(values.unique()))}

    transformed = []
    for frame in (source_train, *frames):
        out = frame.copy()
        for column, mapping in mappings.items():
            if column in out.columns:
                out[column + "_code"] = _safe_str_series(out[column]).map(mapping).fillna(len(mapping)).astype(int)
        transformed.append(out)
    return (*transformed, mappings)


def _fit_source_train_transform(
    source_train: pd.DataFrame,
    source_val: pd.DataFrame,
    target: pd.DataFrame,
    remove_features=None,
):
    """Fit imputation and scaling on source training patients only."""
    source_train, source_val, target, mappings = _apply_source_fitted_categoricals(
        source_train, source_val, target
    )
    features = _allowed_features(source_train, target, remove_features)
    medians = {}
    means = {}
    scales = {}
    for column in features:
        for frame in (source_train, source_val, target):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        median = float(source_train[column].median())
        if not np.isfinite(median):
            median = 0.0
        source_train[column] = source_train[column].fillna(median)
        source_val[column] = source_val[column].fillna(median)
        target[column] = target[column].fillna(median)
        mean = float(source_train[column].mean())
        scale = float(source_train[column].std())
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        for frame in (source_train, source_val, target):
            frame[column] = (frame[column] - mean) / scale
        medians[column], means[column], scales[column] = median, mean, scale
    return source_train, source_val, target, features, mappings, {
        "imputation": "source_train_median",
        "medians": medians,
        "means": means,
        "scales": scales,
    }


def _stratify_or_none(labels: np.ndarray) -> Optional[np.ndarray]:
    values, counts = np.unique(labels, return_counts=True)
    if len(values) < 2 or np.any(counts < 2):
        return None
    return labels


def _patient_level_target_split(
    df_target: pd.DataFrame,
    e_target: np.ndarray,
    target_adapt_ratio: float,
    target_val_ratio: float,
    seed: int,
    patient_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if patient_col not in df_target.columns:
        raise ValueError(
            f"Patient-level split requested but patient column '{patient_col}' is missing."
        )

    patient_frame = pd.DataFrame(
        {
            "patient": df_target[patient_col].astype(str).values,
            "event": e_target.astype(int),
        }
    )
    patient_event = patient_frame.groupby("patient", sort=False)["event"].max()
    patients = patient_event.index.to_numpy()
    patient_labels = patient_event.to_numpy()

    if len(patients) < 5:
        raise ValueError("Too few target patients for patient-level train/val/test split.")

    adapt_patients, test_patients = train_test_split(
        patients,
        test_size=1 - target_adapt_ratio,
        random_state=seed,
        stratify=_stratify_or_none(patient_labels),
    )

    adapt_labels = patient_event.loc[adapt_patients].to_numpy()
    train_patients, val_patients = train_test_split(
        adapt_patients,
        test_size=target_val_ratio,
        random_state=seed,
        stratify=_stratify_or_none(adapt_labels),
    )

    target_patients = df_target[patient_col].astype(str)
    idx_train = np.flatnonzero(target_patients.isin(train_patients).to_numpy())
    idx_val = np.flatnonzero(target_patients.isin(val_patients).to_numpy())
    idx_test = np.flatnonzero(target_patients.isin(test_patients).to_numpy())

    return idx_train, idx_val, idx_test


def _patient_level_source_split(
    x_s: np.ndarray,
    e_s: np.ndarray,
    t_s: np.ndarray,
    df_source: pd.DataFrame,
    source_val_ratio: float,
    seed: int,
    patient_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hold out a patient-level source validation set for source pretrain early stopping.

    Returns (x_train, e_train, t_train, x_val, e_val, t_val) where val sessions
    belong to patients never seen in the source training fold.
    """
    if patient_col not in df_source.columns:
        # Fallback: random session-level split (no patient info)
        idx = np.arange(len(x_s))
        idx_train, idx_val = train_test_split(
            idx, test_size=source_val_ratio, random_state=seed
        )
        return x_s[idx_train], e_s[idx_train], t_s[idx_train], x_s[idx_val], e_s[idx_val], t_s[idx_val]

    patient_frame = pd.DataFrame(
        {"patient": df_source[patient_col].astype(str).values, "event": e_s.astype(int)}
    )
    patient_event = patient_frame.groupby("patient", sort=False)["event"].max()
    patients = patient_event.index.to_numpy()
    patient_labels = patient_event.to_numpy()

    train_patients, val_patients = train_test_split(
        patients,
        test_size=source_val_ratio,
        random_state=seed,
        stratify=_stratify_or_none(patient_labels),
    )

    src_patients = df_source[patient_col].astype(str).values
    idx_train = np.flatnonzero(np.isin(src_patients, train_patients))
    idx_val   = np.flatnonzero(np.isin(src_patients, val_patients))

    return (
        x_s[idx_train], e_s[idx_train], t_s[idx_train],
        x_s[idx_val],   e_s[idx_val],   t_s[idx_val],
    )


def prepare_dataloaders(
    source_path: str,
    target_path: str,
    batch_size: int,
    seed: int = 42,
    target_adapt_ratio: float = 0.2,
    target_val_ratio: float = 0.2,
    source_val_ratio: float = 0.10,   # fraction of SOURCE patients held out for pretrain ES
    target_train_fraction: float = 1.0,
    remove_features: Optional[List[str]] = None,
    patient_col: str = "患者id",
    split_strategy: str = "patient",
):
    """
    End-to-end data preparation returning loaders and evaluation sets.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.evaluate.metrics import compute_ipcw_weights
    
    df_s_raw = _load_and_filter_cohort(source_path)
    df_t_raw = _load_and_filter_cohort(target_path)
    e_s_raw = df_s_raw["events"].to_numpy(dtype=np.float32)
    e_t_raw = df_t_raw["events"].to_numpy(dtype=np.float32)

    if split_strategy == "patient":
        idx_train, idx_val, idx_test = _patient_level_target_split(
            df_t_raw,
            e_t_raw,
            target_adapt_ratio,
            target_val_ratio,
            seed,
            patient_col,
        )
    elif split_strategy == "session":
        target_indices = np.arange(len(df_t_raw))
        idx_adapt_pool, idx_test = train_test_split(
            target_indices,
            test_size=1 - target_adapt_ratio,
            random_state=seed,
            stratify=_stratify_or_none(e_t_raw),
        )
        idx_train, idx_val = train_test_split(
            idx_adapt_pool,
            test_size=target_val_ratio,
            random_state=seed,
            stratify=_stratify_or_none(e_t_raw[idx_adapt_pool]),
        )
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")

    if not 0 < target_train_fraction <= 1:
        raise ValueError("target_train_fraction must be in (0, 1].")
    if target_train_fraction < 1:
        train_patient_ids = df_t_raw.iloc[idx_train][patient_col].astype(str).unique()
        train_labels = (
            df_t_raw.iloc[idx_train]
            .assign(_patient=df_t_raw.iloc[idx_train][patient_col].astype(str).values)
            .groupby("_patient")["events"].max()
            .reindex(train_patient_ids)
            .to_numpy()
        )
        keep_patients, _ = train_test_split(
            train_patient_ids,
            train_size=target_train_fraction,
            random_state=seed,
            stratify=_stratify_or_none(train_labels),
        )
        idx_train = idx_train[np.isin(df_t_raw.iloc[idx_train][patient_col].astype(str), keep_patients)]

    # Split source patients before fitting any encoding, imputation, or scaling.
    source_patients = df_s_raw[patient_col].astype(str).to_numpy()
    patient_event = pd.DataFrame({"patient": source_patients, "event": e_s_raw}).groupby("patient")["event"].max()
    train_patients, val_patients = train_test_split(
        patient_event.index.to_numpy(),
        test_size=source_val_ratio,
        random_state=seed,
        stratify=_stratify_or_none(patient_event.to_numpy()),
    )
    source_train_raw = df_s_raw.loc[np.isin(source_patients, train_patients)].copy()
    source_val_raw = df_s_raw.loc[np.isin(source_patients, val_patients)].copy()
    df_s, df_s_val, df_t, feature_names, category_mappings, preprocessing = _fit_source_train_transform(
        source_train_raw, source_val_raw, df_t_raw, remove_features
    )

    x_s_train = df_s[feature_names].to_numpy(dtype=np.float32)
    e_s_train = df_s["events"].to_numpy(dtype=np.float32)
    t_s_train = df_s["et_min"].to_numpy(dtype=np.float32)
    x_s_val = df_s_val[feature_names].to_numpy(dtype=np.float32)
    e_s_val = df_s_val["events"].to_numpy(dtype=np.float32)
    t_s_val = df_s_val["et_min"].to_numpy(dtype=np.float32)
    x_t = df_t[feature_names].to_numpy(dtype=np.float32)
    e_t = df_t["events"].to_numpy(dtype=np.float32)
    t_t = df_t["et_min"].to_numpy(dtype=np.float32)

    x_train, e_train, t_train = x_t[idx_train], e_t[idx_train], t_t[idx_train]
    x_val, e_val, t_val = x_t[idx_val], e_t[idx_val], t_t[idx_val]
    x_test, e_test, t_test = x_t[idx_test], e_t[idx_test], t_t[idx_test]

    # Compute IPCW (source IPCW on source train only; target IPCW on target train only)
    w_s_train = compute_ipcw_weights(t_s_train, e_s_train)
    w_train   = compute_ipcw_weights(t_train, e_train)

    # Create loaders (source loader uses training patients only)
    ds_s = TensorDataset(
        torch.tensor(x_s_train), torch.tensor(e_s_train),
        torch.tensor(t_s_train), torch.tensor(w_s_train),
    )
    ds_t = TensorDataset(
        torch.tensor(x_train), torch.tensor(e_train),
        torch.tensor(t_train), torch.tensor(w_train),
    )

    source_loader = DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(ds_t, batch_size=batch_size, shuffle=True, drop_last=False)

    return {
        "source_loader": source_loader,
        "target_loader": target_loader,
        # Target validation (drives DA-phase early stopping)
        "x_val": x_val, "e_val": e_val, "t_val": t_val,
        # Source validation (drives SOURCE pretrain early stopping — no target leakage)
        "x_source_val": x_s_val, "e_source_val": e_s_val, "t_source_val": t_s_val,
        "x_test": x_test, "e_test": e_test, "t_test": t_test,
        "x_train": x_train,
        "t_train": t_train, "e_train": e_train,
        # Full source arrays (for metrics that need all source data, e.g. IPCW baseline)
        "x_source_all": np.concatenate([x_s_train, x_s_val]),
        "e_source_all": np.concatenate([e_s_train, e_s_val]),
        "t_source_all": np.concatenate([t_s_train, t_s_val]),
        "df_source": df_s,
        "df_target": df_t,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "patient_ids_train": df_t.iloc[idx_train][patient_col].astype(str).to_numpy(),
        "patient_ids_val":   df_t.iloc[idx_val][patient_col].astype(str).to_numpy(),
        "patient_ids_test":  df_t.iloc[idx_test][patient_col].astype(str).to_numpy(),
        "category_mappings": category_mappings,
        "preprocessing": preprocessing,
        "target_train_fraction": target_train_fraction,
        "split_strategy": split_strategy,
        "patient_col": patient_col,
        "feature_names": feature_names,
        "input_dim": x_s_train.shape[1],
    }
