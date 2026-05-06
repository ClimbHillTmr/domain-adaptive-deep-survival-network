import pandas as pd
import numpy as np
import ast
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Clinical observation window: only use first OBS_WINDOW minutes of data for prediction
# Events occurring within this window are excluded (cannot predict what already happened)
OBS_WINDOW = 60  # minutes


def parse_list_col(val):
    """Parse stringified list from CSV."""
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    elif isinstance(val, list):
        return val
    return []


def pad_sequence(seq, max_len, padding_value=0.0):
    """Pad sequence to max_len."""
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [padding_value] * (max_len - len(seq))


class DialysisDataLoader:
    def __init__(self, config):
        self.cfg = config
        self.static_imputer = None
        self.static_scaler = StandardScaler()
        self.dynamic_imputer = None
        self.dynamic_scaler = StandardScaler()
        self.label_encoders = {}
        self._fitted = False

    def _process_static(self, df, is_training):
        static_cols = self.cfg.columns.static_cols
        available_cols = [c for c in static_cols if c in df.columns]
        X_static = df[available_cols].copy()

        # CRITICAL: Winsorize 超滤量MAX to P99 to eliminate extreme outliers
        # Normal ultrafiltration volume: 500-5000 mL per session
        # Values >10000 are likely data entry errors (extra zeros)
        if "超滤量MAX" in X_static.columns:
            p99 = X_static["超滤量MAX"].quantile(0.99)
            X_static["超滤量MAX"] = X_static["超滤量MAX"].clip(upper=p99)

        # P2: Feature Engineering - Construct clinically meaningful derived features
        # These features capture physiological relationships known to predict IDH

        # 1. Pulse Pressure (脉压差) = SBP - DBP
        # Higher pulse pressure indicates arterial stiffness, a risk factor for IDH
        if "透前收缩压" in X_static.columns and "透前舒张压" in X_static.columns:
            X_static["脉压差"] = X_static["透前收缩压"] - X_static["透前舒张压"]

        # 2. Mean Arterial Pressure (平均动脉压) = (SBP + 2*DBP) / 3
        # Better indicator of perfusion pressure than SBP alone
        if "透前收缩压" in X_static.columns and "透前舒张压" in X_static.columns:
            X_static["平均动脉压"] = (
                X_static["透前收缩压"] + 2 * X_static["透前舒张压"]
            ) / 3

        # 3. Pre-dialysis weight - dry weight difference (超负荷)
        # Positive values indicate fluid overload, a key IDH risk factor
        if "透前体重" in X_static.columns and "干体重" in X_static.columns:
            X_static["超负荷"] = X_static["透前体重"] - X_static["干体重"]

        # 4. Ultrafiltration volume / dry weight ratio (超滤比)
        # Normalized ultrafiltration target, more meaningful than absolute volume
        if "超滤量MAX" in X_static.columns and "干体重" in X_static.columns:
            X_static["超滤比"] = X_static["超滤量MAX"] / (
                X_static["干体重"].replace(0, np.nan)
            )
            X_static["超滤比"] = X_static["超滤比"].fillna(0)

        # 5. Age-dialysis age interaction (透析龄/年龄比)
        # Proportion of life spent on dialysis
        if "透析年龄" in X_static.columns and "透析龄" in X_static.columns:
            X_static["透析龄占比"] = X_static["透析龄"] / (
                X_static["透析年龄"].replace(0, np.nan)
            )
            X_static["透析龄占比"] = X_static["透析龄占比"].fillna(0)

        # 6. Body Mass Index proxy (BMI_proxy) = dry_weight / (height^2)
        # Assuming average height 1.65m if not available
        if "干体重" in X_static.columns:
            avg_height = 1.65  # meters
            X_static["BMI_proxy"] = X_static["干体重"] / (avg_height ** 2)

        # 7. Cardiac Stress Index (心脏负荷指数) = pulse_pressure * heart_rate
        # Combines arterial stiffness and cardiac workload
        if "透前收缩压" in X_static.columns and "透前舒张压" in X_static.columns:
            pulse_pressure = X_static["透前收缩压"] - X_static["透前舒张压"]
            # Use average heart rate ~72 bpm if not available
            X_static["心脏负荷指数"] = pulse_pressure * 72 / 1000  # Normalized

        # 8. Fluid Overload Ratio (容量超负荷比) = (pre_weight - dry_weight) / dry_weight
        # Relative fluid overload, more clinically meaningful than absolute difference
        if "透前体重" in X_static.columns and "干体重" in X_static.columns:
            X_static["容量超负荷比"] = (
                (X_static["透前体重"] - X_static["干体重"]) /
                X_static["干体重"].replace(0, np.nan)
            )
            X_static["容量超负荷比"] = X_static["容量超负荷比"].fillna(0)

        # 9. Age-Weight Interaction (年龄体重交互)
        # Older patients with higher weight may have different risk profiles
        if "透析年龄" in X_static.columns and "干体重" in X_static.columns:
            X_static["年龄体重交互"] = (
                X_static["透析年龄"] * X_static["干体重"]
            ) / 1000  # Normalized

        # 10. Hypertension-Age Interaction (高血压年龄交互)
        # Hypertension risk increases with age
        if "高血压诊断" in X_static.columns and "透析年龄" in X_static.columns:
            X_static["高血压年龄交互"] = (
                X_static["高血压诊断"] * X_static["透析年龄"]
            ) / 100  # Normalized

        cat_cols = ["性别", "高血压诊断"]
        for col in cat_cols:
            if col in X_static.columns:
                if is_training:
                    le = LabelEncoder()
                    X_static[col] = le.fit_transform(X_static[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        X_static[col] = (
                            X_static[col]
                            .astype(str)
                            .map(
                                lambda x: (
                                    le.transform([x])[0] if x in le.classes_ else 0
                                )
                            )
                        )

        X_static_vals = X_static.values
        if len(available_cols) > 0:
            if is_training:
                self.static_imputer = SimpleImputer(strategy="mean")
                X_static_vals = self.static_imputer.fit_transform(X_static_vals)
                self.static_scaler = StandardScaler()
                X_static_vals = self.static_scaler.fit_transform(X_static_vals)
                self._fitted = True
            else:
                if not self._fitted:
                    raise RuntimeError(
                        "Loader not fitted. Call load_data with is_training=True first."
                    )
                X_static_vals = self.static_imputer.transform(X_static_vals)
                X_static_vals = self.static_scaler.transform(X_static_vals)
        else:
            X_static_vals = np.zeros((len(df), 1))

        return X_static_vals.astype(np.float32)

    def _process_dynamic(self, df, is_training):
        seq_cols = self.cfg.columns.seq_cols
        dynamic_cols = self.cfg.columns.dynamic_cols
        max_len = self.cfg.model.transformer.max_len
        n_samples = len(df)
        X_dynamic = None

        # Truncate sequence to observation window (first OBS_WINDOW minutes)
        # Sampling rate: max_len covers 4 hours = 240 minutes
        # OBS_WINDOW=60 minutes => keep first OBS_WINDOW/240 * max_len frames
        obs_frames = int(OBS_WINDOW / 240.0 * max_len)
        obs_frames = max(obs_frames, 1)  # at least 1 frame

        sample_col = seq_cols[0] if seq_cols else None
        if sample_col and sample_col in df.columns:
            sample_val = df[sample_col].iloc[0] if len(df) > 0 else None
            if isinstance(sample_val, (list, str)) and sample_val.startswith("["):
                n_features = len(seq_cols)
                X_dynamic = np.zeros((n_samples, obs_frames, n_features))
                for i, col in enumerate(seq_cols):
                    if col in df.columns:
                        parsed = df[col].apply(parse_list_col)
                        # Truncate to observation window, then pad if shorter
                        padded = parsed.apply(
                            lambda x: pad_sequence(x[:obs_frames], obs_frames)
                        )
                        X_dynamic[:, :, i] = np.vstack(padded)

        if X_dynamic is None and dynamic_cols:
            available_dyn = [c for c in dynamic_cols if c in df.columns]
            if available_dyn:
                n_features = len(available_dyn)
                X_dynamic = np.zeros((n_samples, 1, n_features))
                X_dynamic[:, 0, :] = df[available_dyn].values

                # CRITICAL: Normalize dynamic features (same as static)
                if is_training:
                    self.dynamic_imputer = SimpleImputer(strategy="mean")
                    X_dynamic_flat = self.dynamic_imputer.fit_transform(
                        X_dynamic.reshape(n_samples, -1)
                    )
                    self.dynamic_scaler = StandardScaler()
                    X_dynamic_flat = self.dynamic_scaler.fit_transform(X_dynamic_flat)
                    X_dynamic = X_dynamic_flat.reshape(n_samples, 1, n_features)
                    self._fitted = True
                else:
                    if not self._fitted:
                        raise RuntimeError(
                            "Loader not fitted. Call load_data with is_training=True first."
                        )
                    if self.dynamic_imputer is not None:
                        X_dynamic_flat = self.dynamic_imputer.transform(
                            X_dynamic.reshape(n_samples, -1)
                        )
                        X_dynamic_flat = self.dynamic_scaler.transform(X_dynamic_flat)
                        X_dynamic = X_dynamic_flat.reshape(n_samples, 1, n_features)
            else:
                X_dynamic = np.zeros((n_samples, 1, 1))

        if X_dynamic is None:
            X_dynamic = np.zeros((n_samples, 1, 1))

        return X_dynamic.astype(np.float32)

    def _process_targets(self, df):
        targets = {}
        if self.cfg.columns.targets.event_col in df.columns:
            targets["event"] = (
                df[self.cfg.columns.targets.event_col].fillna(0).values.astype(int)
            )

        if self.cfg.columns.targets.duration_col in df.columns:
            targets["duration"] = (
                df[self.cfg.columns.targets.duration_col].fillna(0).values.astype(float)
            )

        if self.cfg.columns.targets.label_col in df.columns:
            targets["label"] = (
                df[self.cfg.columns.targets.label_col].fillna(0).values.astype(int)
            )

        # CRITICAL: Filter out events that occurred within the observation window.
        # We can only predict future events for patients who survived OBS_WINDOW minutes.
        # This prevents data leakage: the model must not learn from events that already
        # happened during the observation period used as input.
        valid_idx = (targets["duration"] > 0) | (targets["event"] == 0)
        if "event" in targets and "duration" in targets:
            early_event_mask = (targets["event"] == 1) & (
                targets["duration"] <= OBS_WINDOW
            )
            valid_idx = valid_idx & ~early_event_mask

        if "event" in targets:
            targets["event"] = targets["event"][valid_idx]
        if "duration" in targets:
            targets["duration"] = targets["duration"][valid_idx]
        if "label" in targets:
            targets["label"] = targets["label"][valid_idx]

        return targets, valid_idx

    def load_data(
        self, csv_path: str, is_training: bool = True
    ) -> Dict[str, np.ndarray]:
        df = pd.read_csv(csv_path)

        X_static_vals = self._process_static(df, is_training)
        X_dynamic = self._process_dynamic(df, is_training)
        targets, valid_idx = self._process_targets(df)

        X_static_vals = X_static_vals[valid_idx]
        X_dynamic = X_dynamic[valid_idx]

        return {
            "static": X_static_vals,
            "dynamic": X_dynamic,
            "targets": targets,
            "df_raw": df,
        }


def get_loader(config):
    return DialysisDataLoader(config)


def load_and_build_features(
    csv_path, boundaries, columns, missing_threshold, gap_threshold
):
    from omegaconf import OmegaConf

    conf = OmegaConf.create(
        {"columns": columns, "model": {"transformer": {"max_len": 48}}}
    )
    loader = DialysisDataLoader(conf)
    data = loader.load_data(csv_path)
    return data, data["targets"]["label"]


def load_seq_static_survival(
    csv_path, boundaries, columns, missing_threshold, gap_threshold
):
    from omegaconf import OmegaConf

    conf = OmegaConf.create(
        {"columns": columns, "model": {"transformer": {"max_len": 48}}}
    )
    loader = DialysisDataLoader(conf)
    data = loader.load_data(csv_path)
    return (
        data["dynamic"],
        data["static"],
        data["targets"]["label"],
        data["targets"]["duration"],
        data["targets"]["event"],
    )
