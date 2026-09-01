"""v6.0 — 数据装载与预处理 (P1 数据资产的消费层)。

从 data/v6_20260830 冻结资产构建训练/验证/测试所需的张量视图。
单一损失与模型同代际，杜绝 v4/v5 的代际混杂。

本模块仅做「装载 + 标准化 + 离散生存目标构造」，不含训练。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/v6_20260830"

# 19 项锁定特征（顺序与 preprocessing.json 一致）
FEATURES = [
    "年龄", "性别", "透析龄年", "透前收缩压", "透前舒张压", "透前体重",
    "透前体重-干体重", "历史平均透前体重", "历史平均透前收缩压",
    "历史平均透前舒张压", "history_prior_session_count",
    "history_IDH_240_interval_1_rate", "history_IDH_240_interval_2_rate",
    "history_IDH_240_interval_3_rate", "history_IDH_240_interval_4_rate",
    "history_IH_240_interval_1_rate", "history_IH_240_interval_2_rate",
    "history_IH_240_interval_3_rate", "history_IH_240_interval_4_rate",
]
LOG1P = ["history_prior_session_count"]
TREATMENT = ["透前体重-干体重"]
HORIZONS = [60.0, 120.0, 180.0, 240.0]


def load_preprocessing() -> dict:
    path = DATA_ROOT / "preprocessing.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_split() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / "split_manifest.csv", dtype={"患者id": str})


def load_budget() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / "budget_subsets.csv", dtype={"患者id": str})


def load_cohorts(
    *,
    apply_preprocess: bool = True,
    preprocessor: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载 source/target cohort 并打标准化。

    target 的 history 特征在冻结时已置 NaN；source 保留完整历史。
    standardization 用冻结的 medians + population_sd（v6 protocol）。
    """
    source = pd.read_csv(DATA_ROOT / "source_cohort.csv", dtype={"患者id": str})
    target = pd.read_csv(DATA_ROOT / "target_cohort.csv", dtype={"患者id": str})
    if apply_preprocess:
        if preprocessor is None:
            preprocessor = load_preprocessing()
        source = _standardize(source, preprocessor)
        target = _standardize(target, preprocessor)
    return source, target


def _standardize(frame: pd.DataFrame, pre: dict) -> pd.DataFrame:
    out = frame.copy()
    medians = pre["medians"]
    scales = pre["scales_population_sd"]
    for feature in FEATURES:
        raw = pd.to_numeric(out[feature], errors="coerce")
        if feature in medians:
            raw = raw.fillna(medians[feature])
        out[feature] = (raw - medians[feature]) / scales[feature]
    return out


def discrete_survival_targets(events: pd.Series, intervals: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """转 4 间隔离散生存目标 y_ik 与观测 mask o_ik。

    **v6 单一真值来源**：训练/评估 loop 一律从本函数取目标，禁止在 runner 内联
    重写。这是防止 v4/v5「双实现漂移」（pos_weight vs patient-balanced）的硬约定。

    对每条 session：
      - `observed_240=1`（240 min 内发生事件，interval ∈ {1,2,3,4}）
        → interval-1 前的格为可观测 0，事件格为 1，其后格不可观测
      - `observed_240=0`（240 min 内未发生/右删失）→ 全程观测，目标全 0
    返回 (n_sessions, 4) 的 y（float 0/1）与 o（int 0/1 观测 mask）。
    """
    n = len(events)
    y = np.zeros((n, 4), dtype=float)
    o = np.zeros((n, 4), dtype=int)
    ev = np.asarray(events, dtype=int)
    iv = np.asarray(intervals, dtype=int).clip(1, 4)
    iv = np.where(ev == 1, iv, 4)  # 删失者观测到最后一个格
    for i in range(4):
        # 该格在随访窗口内才可观测（i+1 >= 事件间隔或事件未发生全程）
        obs = (ev == 0) | (iv >= (i + 1))
        # 事件在该格发生（ev=1 且 interval == i+1）
        y_target = (ev == 1) & (iv == (i + 1))
        o[:, i] = obs.astype(int)
        y[:, i] = y_target.astype(float)
    return y, o


def session_patient_counts(view: pd.DataFrame) -> np.ndarray:
    """返回每条 session 所属患者的 session 总数 n_i（patient-balanced NLL 权重用）。"""
    sizes = view.groupby("患者id").size()
    return view["患者id"].map(sizes).to_numpy(dtype=float)


def session_batch(
    cohorts: tuple[pd.DataFrame, pd.DataFrame],
    split: pd.DataFrame,
    budget: pd.DataFrame | None,
    role: str,
    *,
    center: str = "source",
    budget_frac: float | None = None,
    subset_repeat: int = 1,
) -> pd.DataFrame:
    """返回指定 center+role(+budget) 的 session 视图，含特征/目标/患者id。

    center='source'：仅 source_train（预训练）。
    center='target'：role ∈ {target_update, target_calibration, target_test}。
      budget_frac 非空时，把 target_update 限定为预算选中的患者。
    """
    source, target = cohorts
    base = source if center == "source" else target
    # 合并 split_role 到 base，避免对 split 原索引做掩码导致的错位
    merged = base.merge(split[["session_id", "split_role"]], on="session_id", how="left", validate="one_to_one")
    keep_sessions = merged.loc[merged["split_role"].eq(role), "session_id"]

    if center == "target" and budget_frac is not None and role == "target_update":
        if budget is None:
            budget = load_budget()
        sel = set(budget.loc[
            budget["budget"].eq(float(budget_frac)) & budget["subset_repeat"].eq(int(subset_repeat)),
            "患者id",
        ].astype(str))
        keep_patients = merged["患者id"].isin(sel) & merged["split_role"].eq(role)
        keep_sessions = merged.loc[keep_patients, "session_id"]

    view = base[base["session_id"].isin(set(keep_sessions))].copy()
    return view.reset_index(drop=True)