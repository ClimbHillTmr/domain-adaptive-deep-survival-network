"""v6.0 — 统计推断 harness (P4)：patient-cluster bootstrap + Holm-2 主族 + 效应量。

输入：experiments/v6_20260830/{family}__{endpoint}__{budget}__{seed}/test_predictions.csv
  （由 run_v6_train.py 产出，含 patient_id / {endpoint}_survival_time /
    {endpoint}_event_observed_240 / risk_60..risk_240）

推断（对齐 RESEARCH_PLAN_V6 第 4 章）：
  1. 在 target 患者级生成一次 patient-cluster bootstrap 索引（固定 INF_SEED，全文件共用，
     保证对比两臂在同一患者抽样上配对）。
  2. 每个 run 对每 replicate 计算 4 时点 IPCW-AUC 均值 alpha_not_estimable 用 flag 标记。
  3. 主族（Holm-2，仅 2 条）：
       H10: b=10%, IDH  supervised_update − target_only ΔmeanAUC > 0
       H20: b=10%, IH   supervised_update − target_only ΔmeanAUC > 0
     跨 5 seed 池化 Δ_bootstrap，报告：
       Δ 点估计(=5 seed 均值)、双侧 95% percentile CI、单侧 p（+1 伪计数，下限 2/(N+1)）、
       Holm 调整 p、判定。
  4. 支持性证据（descriptive，不校正）：b∈{0.25,0.50,1.00} 的 supervised−target_only Δ + CI，
     以及 source_only 参考 meanAUC + CI。
  5. IPCW G(t) 估计性诊断：frac_bootstrap_not_estimable + 全测集 G(240)/G(240-)。

**种子合成约定**（需稿件层确认，见方案 4.1 未定项）：Δ 为同种子配对、跨 seed 池化。

输出：experiments/v6_20260830/inference/
  main_holm2.csv            ：主族判定表
  supportive_descriptive.csv：支持性证据（Δ+CI）
  g_estimability.csv        ：G(t) 诊断
  bootstrap_indices.npy     ：可复现的患者级 bootstrap 索引（跨 seed 固化）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate.discrete_survival_metrics import (
    evaluate_discrete_survival,
    fit_reverse_kaplan_meier,
    patient_cluster_bootstrap_indices,
)

OUT_ROOT = ROOT / "experiments/v6_20260830"
INF_DIR = OUT_ROOT / "inference"

FAMILIES = ("target_only", "source_only", "supervised_update")
ENDPOINTS = ("idh", "ih")
BUDGETS = (0.10, 0.25, 0.50, 1.00)
SEEDS = (401, 402, 433, 202, 107)
HORIZONS = [60.0, 120.0, 180.0, 240.0]
MIN_G = 1e-8
INF_SEED = 20260831  # 固定：患者级 bootstrap 索引的可复现种子（全文件共用）
MAIN_BUDGET = 0.10
MAIN_PAIR = ("supervised_update", "target_only")
HORN = {h: int(h) for h in HORIZONS}  # horizon -> risk column label 60/120/180/240

# ---- 进程内共享（fork 后 copy-on-write），由 multiprocessing initializer 设置 ----
_G = {}


def _auc_replicate(repl_idx) -> tuple[float, int]:
    """计算一个 bootstrap replica 的 4 时点 mean_auc；异常(不可估)返回 (nan,1)。"""
    times, events, risk, horizons = _G["times"], _G["events"], _G["risk"], _G["horizons"]
    idx = _G["idx_list"][repl_idx]
    try:
        per = np.array([
            _gcd_auc(times, events, risk, idx, h) for h in horizons
        ])
        return float(np.mean(per)), 0
    except ValueError:
        return float("nan"), 1


def _gcd_auc(times, events, risk, idx, horizon) -> float:
    """对抽样行计算单个时点 IPCW-AUC；不可估时抛 ValueError。"""
    from src.evaluate.discrete_survival_metrics import ipcw_cumulative_dynamic_auc
    return ipcw_cumulative_dynamic_auc(
        times[idx], events[idx], risk[idx, _G["hcol"][horizon]], float(horizon), min_g=MIN_G
    )


def _init_worker(times, events, risk, idx_list):
    _G.update(times=times, events=events, risk=risk, idx_list=idx_list,
              horizons=HORIZONS, hcol={h: list(HORIZONS).index(h) for h in HORIZONS})


def _run_run_boot_stats(run_dir: Path, endpoint: str, idx_list: list[np.ndarray],
                        n_workers: int) -> tuple[float, np.ndarray, float, int]:
    """加载一个 run 的预测，返回 (point_mean_auc, boot_mean_auc[n_repl], frac_not_estimable, n_valid)。"""
    df = pd.read_csv(run_dir / "test_predictions.csv", dtype={"patient_id": str})
    times = df[f"{endpoint}_survival_time"].to_numpy(dtype=float)
    events = df[f"{endpoint}_event_observed_240"].to_numpy(dtype=int)
    risk = df[[f"risk_{int(h)}" for h in HORIZONS]].to_numpy(dtype=float)
    if np.any(np.diff(risk, axis=1) < -1e-12):
        raise AssertionError(f"Non-decreasing risk violated: {run_dir.name}")
    point = float(evaluate_discrete_survival(times, events, risk, horizons=np.array(HORIZONS))["mean_auc"])

    ctx = get_context("fork")
    with ctx.Pool(processes=n_workers, initializer=_init_worker,
                  initargs=(times, events, risk, idx_list)) as pool:
        results = pool.map(_auc_replicate, range(len(idx_list)), chunksize=16)
    boots = np.array([r[0] for r in results], dtype=float)
    invalid = np.array([r[1] for r in results], dtype=int)
    n_invalid = int(invalid.sum())
    frac = n_invalid / max(len(idx_list), 1)
    # 保留全长（不可估 replica 为 NaN）以维持跨臂的 by-position 配对；
    # 有效计数供诊断，过滤在对比组装时按位置进行。
    n_valid = int(np.isfinite(boots).sum())
    return point, boots, frac, n_valid


def _load_run(run_dir: Path, endpoint: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(run_dir / "test_predictions.csv", dtype={"patient_id": str})
    return (
        df["patient_id"].to_numpy(dtype=str),
        df[f"{endpoint}_survival_time"].to_numpy(dtype=float),
        df[f"{endpoint}_event_observed_240"].to_numpy(dtype=int),
    )


def _pool_contrast(eps: str, budgets: tuple[float, ...], pair: tuple[str, str],
                   cache: dict, idx_list: list[np.ndarray], n_workers: int) -> dict:
    """对 (endpoint, budget) 计算 pair 对比的池化 bootstrap 指标。cache 存 per-file 原始 boots/point。"""
    out = {}
    for b in budgets:
        sup_boots, tgt_boots, sup_point, tgt_point = [], [], 0.0, 0.0
        n_valid_sup = n_valid_tgt = 0
        frac_sup, frac_tgt = 0.0, 0.0
        for s in SEEDS:
            key_sup = (pair[0], eps, b, s)
            if key_sup not in cache:
                p, boots, frac, nv = _run_run_boot_stats(
                    OUT_ROOT / f"{pair[0]}__{eps}__{b:.2f}__seed{s}", eps, idx_list, n_workers)
                cache[key_sup] = (p, boots, frac, nv)
            p, boots, frac, nv = cache[key_sup]
            sup_boots.append(boots); sup_point += p / len(SEEDS); frac_sup += frac / len(SEEDS); n_valid_sup += nv
            key_tgt = (pair[1], eps, b, s)
            if key_tgt not in cache:
                p2, boots2, frac2, nv2 = _run_run_boot_stats(
                    OUT_ROOT / f"{pair[1]}__{eps}__{b:.2f}__seed{s}", eps, idx_list, n_workers)
                cache[key_tgt] = (p2, boots2, frac2, nv2)
            p2, boots2, frac2, nv2 = cache[key_tgt]
            tgt_boots.append(boots2); tgt_point += p2 / len(SEEDS); frac_tgt += frac2 / len(SEEDS); n_valid_tgt += nv2

        # 同 seed 配对：sup/tgt 各 replica 全长并按位置对齐，任一侧不可估则剔除该 Δ
        deltas = []
        dropped = 0
        for s in range(len(SEEDS)):
            sb, tb = sup_boots[s], tgt_boots[s]
            min_len = min(len(sb), len(tb))
            d = sb[:min_len] - tb[:min_len]
            finite = np.isfinite(d)
            deltas.append(d[finite])
            dropped += int((~finite).sum())
        delta_pool = np.concatenate(deltas) if deltas else np.array([])
        n_seed = len(SEEDS)
        delta = float(sup_point - tgt_point)
        if delta_pool.size:
            ci_lo, ci_hi = np.percentile(delta_pool, [2.5, 97.5])
            n_le0 = int(np.sum(delta_pool <= 0))
            p_raw = (n_le0 + 1) / (delta_pool.size + 1)
            p_raw = max(p_raw, 2.0 / (delta_pool.size + 1))  # +1 伪计数下限
        else:
            ci_lo = ci_hi = float("nan"); p_raw = float("nan")
        out[b] = {
            "delta": delta, "ci_low": float(ci_lo), "ci_high": float(ci_hi),
            "raw_p": float(p_raw),
            "mean_sup": round(sup_point, 6), "mean_tgt": round(tgt_point, 6),
            "frac_not_estimable": round(max(frac_sup, frac_tgt), 6),
            "n_valid_pair_sup": n_valid_sup, "n_valid_pair_tgt": n_valid_tgt,
            "n_dropped_replica": dropped, "n_pooled": int(delta_pool.size),
        }
    return out


def _single_mean_boot(endpoint: str, budget: float, idx_list, n_workers: int, cache: dict):
    """source_only 独占：5-seed 池化 point meanAUC + bootstrap CI（无对比），descriptive。"""
    key = ("source_only", endpoint, budget, "ref")
    if key not in cache:
        boots_all = []
        point = 0.0
        frac = 0.0
        for sd in SEEDS:
            rd = OUT_ROOT / f"source_only__{endpoint}__{budget:.2f}__seed{sd}"
            p_, b_, f_, _ = _run_run_boot_stats(rd, endpoint, idx_list, n_workers)
            boots_all.append(b_); point += p_ / len(SEEDS); frac += f_ / len(SEEDS)
        boots = np.concatenate(boots_all)
        cache[key] = (point, boots, frac)
    point, boots, frac = cache[key]
    if boots.size:
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    else:
        ci_lo = ci_hi = float("nan")
    return {"mean": point, "ci_low": float(ci_lo), "ci_high": float(ci_hi),
            "frac_not_estimable": frac, "n_pooled": int(boots.size)}


def _holm2(p_by_key: dict[str, float]) -> dict[str, float]:
    m = len(p_by_key)
    order = sorted(p_by_key, key=lambda k: p_by_key[k])
    adj = {}
    running = 0.0
    for rank, k in enumerate(order, start=1):
        v = min(1.0, (m - rank + 1) * p_by_key[k])
        running = max(running, v)
        adj[k] = min(1.0, running)
    return adj


def _g_estimability(endpoint: str) -> dict:
    run_dir = OUT_ROOT / f"target_only__{endpoint}__0.10__seed401"
    _, times, events = _load_run(run_dir, endpoint)
    cens = fit_reverse_kaplan_meier(times, events)
    return {
        "endpoint": endpoint,
        "g_240": float(cens.g(240.0)),
        "g_240_left": float(cens.g_left(240.0)),
        "n_test_sessions": int(len(times)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=INF_SEED)
    args = parser.parse_args(argv)

    INF_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 用第一个 run 的患者 id 序列生成患者级 bootstrap 索引（全文件共用，保证配对）
    ref_run = OUT_ROOT / "target_only__idh__0.10__seed401"
    ref_pid, _, _ = _load_run(ref_run, "idh")
    rng = np.random.default_rng(args.seed)
    idx_list = patient_cluster_bootstrap_indices(
        ref_pid, n_bootstrap=args.reps, seed=args.seed)
    np.save(INF_DIR / "bootstrap_indices.npy", np.array([x.astype(np.int32) for x in idx_list], dtype=object), allow_pickle=True)

    cache: dict = {}

    # ---- 主族：Holm-2，仅 b=10% 两 endpoint ----
    main = _pool_contrast("idh", (MAIN_BUDGET,), MAIN_PAIR, cache, idx_list, args.workers)[MAIN_BUDGET]
    main_ih = _pool_contrast("ih", (MAIN_BUDGET,), MAIN_PAIR, cache, idx_list, args.workers)[MAIN_BUDGET]
    raw = {"idh": main["raw_p"], "ih": main_ih["raw_p"]}
    adj = _holm2(raw)

    main_rows = []
    for eps, m in (("idh", main), ("ih", main_ih)):
        main_rows.append({
            "hypothesis": "H10" if eps == "idh" else "H20",
            "endpoint": eps, "budget": MAIN_BUDGET, "n_seeds": len(SEEDS), "n_pooled": m["n_pooled"],
            "mean_auc_supervised": m["mean_sup"], "mean_auc_target_only": m["mean_tgt"],
            "delta": round(m["delta"], 6),
            "ci_low": round(m["ci_low"], 6), "ci_high": round(m["ci_high"], 6),
            "ci_width": round(m["ci_high"] - m["ci_low"], 6),
            "raw_p": round(float(m["raw_p"]), 6), "holm_adj_p": round(adj[eps], 6),
            "significant_alpha_0.05": adj[eps] < 0.05,
            "frac_bootstrap_not_estimable": m["frac_not_estimable"],
            "n_dropped_replica": m["n_dropped_replica"],
        })
    pd.DataFrame(main_rows).to_csv(INF_DIR / "main_holm2.csv", index=False, encoding="utf-8")

    # ---- 支持性证据（descriptive，不校正）：b∈{0.25,0.50,1.00} ----
    supp = _pool_contrast("idh", (0.25, 0.50, 1.00), MAIN_PAIR, cache, idx_list, args.workers)
    supp_ih = _pool_contrast("ih", (0.25, 0.50, 1.00), MAIN_PAIR, cache, idx_list, args.workers)
    supp_rows = []
    for eps, tbl in (("idh", supp), ("ih", supp_ih)):
        for b, m in tbl.items():
            supp_rows.append({
                "endpoint": eps, "budget": b, "contrast": "supervised_update - target_only",
                "delta": round(m["delta"], 6), "ci_low": round(m["ci_low"], 6),
                "ci_high": round(m["ci_high"], 6), "ci_width": round(m["ci_high"] - m["ci_low"], 6),
                "mean_supervised": m["mean_sup"], "mean_target_only": m["mean_tgt"],
                "frac_bootstrap_not_estimable": m["frac_not_estimable"],
                "corrected": "descriptive (not multiplicity-corrected)",
            })
    # source_only 参考（无对比）
    for eps in ENDPOINTS:
        ref = _single_mean_boot(eps, 1.00, idx_list, args.workers, cache)
        supp_rows.append({
            "endpoint": eps, "budget": 1.00, "contrast": "source_only (reference)",
            "delta": float("nan"), "ci_low": round(ref["ci_low"], 6), "ci_high": round(ref["ci_high"], 6),
            "ci_width": round(ref["ci_high"] - ref["ci_low"], 6),
            "mean_supervised": round(ref["mean"], 6), "mean_target_only": float("nan"),
            "frac_bootstrap_not_estimable": ref["frac_not_estimable"],
            "corrected": "descriptive (not multiplicity-corrected)",
        })
    pd.DataFrame(supp_rows).to_csv(INF_DIR / "supportive_descriptive.csv", index=False, encoding="utf-8")

    # ---- G(t) 估计性诊断 ----
    pd.DataFrame([_g_estimability("idh"), _g_estimability("ih")]) \
        .to_csv(INF_DIR / "g_estimability.csv", index=False, encoding="utf-8")

    summary = {
        "n_bootstrap": args.reps, "bootstrap_seed": args.seed,
        "holm2_adjusted": {k: round(v, 6) for k, v in adj.items()},
        "main": {eps: {k: m[k] for k in ("delta", "ci_low", "ci_high", "raw_p")} for eps, m in (("idh", main), ("ih", main_ih))},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (INF_DIR / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "n_bootstrap"}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())