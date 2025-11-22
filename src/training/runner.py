import os
import json
import numpy as np
from datetime import datetime
from src.labels.stage_builder import build_stage_label
from src.models.lightgbm_stage import LightGBMStageClassifier
from src.models.transformer_stage import train_transformer_stage
from src.models.wrappers import TorchStageWrapper
from src.evaluation.classification import (
    compute_basic_metrics,
    per_stage_recall,
    brier_score,
    stage_misalignment,
    confusion,
)
from src.calibration.stage_calibration import (
    calibrate_per_stage,
    optimize_thresholds_fbeta,
    bootstrap_ci_per_stage,
    plot_calibration_curves,
)
from src.survival.stage_mapper import map_survival_to_stage_probs


def _coral_align(Xs, Xt, eps=1e-6):
    import numpy as np
    from sklearn.covariance import LedoitWolf

    Xs = np.asarray(Xs, dtype=np.float64)
    Xt = np.asarray(Xt, dtype=np.float64)
    mu_s = np.mean(Xs, axis=0)
    mu_t = np.mean(Xt, axis=0)
    Xs_c = Xs - mu_s
    Xt_c = Xt - mu_t
    try:
        Cs = LedoitWolf().fit(Xs_c).covariance_ + eps * np.eye(Xs.shape[1])
        Ct = LedoitWolf().fit(Xt_c).covariance_ + eps * np.eye(Xt.shape[1])
    except Exception:
        Cs = np.cov(Xs_c.T) + eps * np.eye(Xs.shape[1])
        Ct = np.cov(Xt_c.T) + eps * np.eye(Xt.shape[1])
    ws, Vs = np.linalg.eigh(Cs)
    wt, Vt = np.linalg.eigh(Ct)
    sqrtCs = Vs @ np.diag(np.sqrt(np.maximum(ws, eps))) @ Vs.T
    invSqrtCt = Vt @ np.diag(1.0 / np.sqrt(np.maximum(wt, eps))) @ Vt.T
    T = invSqrtCt @ sqrtCs
    Xt_a = Xt_c @ T + mu_s
    return Xt_a


def run_once(
    X,
    y_stage,
    save_dir="./runs",
    model_type="lgbm",
    X_seq=None,
    X_static=None,
    survival=False,
    boundaries=(30, 90),
    durations=None,
    events=None,
    calibrate=True,
    calibration_method="sigmoid",
    cv_folds=3,
    seeds=None,
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(save_dir, ts)
    os.makedirs(out, exist_ok=True)

    if model_type == "lgbm":
        unique, counts = np.unique(y_stage, return_counts=True)
        total = y_stage.shape[0]
        cw = {
            int(c): float(total / (len(unique) * n))
            for c, n in zip(unique.tolist(), counts.tolist())
        }
        clf = LightGBMStageClassifier(
            class_weight=cw,
            calibrate=calibrate,
            calibration_method=calibration_method,
            cv_folds=cv_folds,
            seeds=(seeds or [42, 17, 99]),
        )
        clf.fit(X, y_stage)
        proba = clf.predict_proba(X)
        preds = clf.predict(X)
    else:
        model = train_transformer_stage(X_seq, y_stage, X_static=X_static)
        wrapper = TorchStageWrapper(model)
        proba = wrapper.predict_proba(X_seq, X_static)
        preds = wrapper.predict(X_seq, X_static)

    # 形状修复：确保 proba 与 y_stage 样本数一致
    n_samples = y_stage.shape[0]
    if proba.ndim == 2:
        if proba.shape[0] != n_samples and proba.shape[1] == n_samples:
            proba = proba.T
    # 重新计算 preds（仅在Transformer路径下，lgbm内部已处理权重）
    if model_type != "lgbm":
        preds = np.argmax(proba, axis=1)

    # 对齐长度
    y_len = y_stage.shape[0]
    p_len = preds.shape[0]
    min_len = min(y_len, p_len)
    if y_len != p_len:
        y_stage = y_stage[:min_len]
        preds = preds[:min_len]
        if proba is not None and proba.shape[0] != min_len:
            proba = proba[:min_len]
    metrics = {}
    metrics.update(compute_basic_metrics(y_stage, preds))
    metrics.update(per_stage_recall(y_stage, preds))
    metrics.update(stage_misalignment(y_stage, preds))
    metrics["brier"] = brier_score(y_stage, proba)

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 保存阶段概率与置信区间
    lower, upper = bootstrap_ci_per_stage(proba)
    np.savetxt(
        os.path.join(out, "stage_probabilities.csv"), proba, delimiter=",", fmt="%.6f"
    )
    np.savetxt(
        os.path.join(out, "stage_prob_ci_lower.csv"), lower, delimiter=",", fmt="%.6f"
    )
    np.savetxt(
        os.path.join(out, "stage_prob_ci_upper.csv"), upper, delimiter=",", fmt="%.6f"
    )

    # 混淆矩阵
    cm = confusion(y_stage, preds)
    np.savetxt(os.path.join(out, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")

    # 校准曲线
    plot_calibration_curves(proba, y_stage, os.path.join(out, "calibration"))

    try:
        from src.calibration.stage_calibration import (
            calibrate_per_stage,
            apply_thresholds_predict,
            optimize_thresholds_fbeta_per_class,
        )

        proba_cal = calibrate_per_stage(proba, y_stage)
        betas = [2.5, 2.0, 1.5, 1.2]
        th, fb = optimize_thresholds_fbeta_per_class(proba_cal, y_stage, betas)
        with open(os.path.join(out, "thresholds.json"), "w") as f:
            json.dump(
                {
                    "beta": betas,
                    "thresholds": list(map(float, th)),
                    "f_beta": list(map(float, fb)),
                },
                f,
                indent=2,
            )
        from src.calibration.stage_calibration import (
            optimize_thresholds_with_distribution,
        )

        target_props = [float(np.mean(y_stage == c)) for c in range(proba_cal.shape[1])]
        th_dist = optimize_thresholds_with_distribution(
            proba_cal,
            y_stage,
            target_props=target_props,
            beta=2.0,
            min_thresholds=[0.15, 0.08, 0.12, 0.12],
        )
        preds_th = apply_thresholds_predict(
            proba_cal,
            th_dist,
            priority=[1, 0, 2, 3],
            weights=[1.12, 1.08, 1.0, 0.98],
            min_thresholds=[0.15, 0.08, 0.12, 0.12],
            conf_min=0.30,
        )
        metrics_th = {}
        metrics_th.update(compute_basic_metrics(y_stage, preds_th))
        metrics_th.update(per_stage_recall(y_stage, preds_th))
        metrics_th.update(stage_misalignment(y_stage, preds_th))
        metrics_th["brier"] = brier_score(y_stage, proba_cal)
        with open(os.path.join(out, "metrics_thresholded.json"), "w") as f:
            json.dump(metrics_th, f, indent=2)
    except Exception:
        pass

    # 生存辅助（仅当传入durations/events且survival=True）
    if survival and durations is not None and events is not None:
        try:
            from src.survival.cox_adapter import CoxAdapter
            import pandas as pd

            df_cov = pd.DataFrame(X_static if X_static is not None else X)
            cox = CoxAdapter()
            cox.fit(df_cov, durations, events)
            times = np.array([boundaries[0], boundaries[1], float(np.max(durations))])
            sf = cox.survival_at(df_cov, times)
            # 取第一样本的生存函数作为示例映射
            # 实际可对每个样本映射并与分类阶段做一致性评估
            soft_probs = map_survival_to_stage_probs(
                sf.iloc[:, 0], boundaries=boundaries
            )
            # HR/CI输出
            exp = cox.explain()
            import json

            with open(os.path.join(out, "coefficients.csv"), "w") as f:
                f.write("name,coef,hr,ci_lower,ci_upper,p\n")
                for k in exp["coefficients"].keys():
                    f.write(
                        f"{k},{exp['coefficients'][k]},{exp['hazard_ratio'][k]},{exp['ci_lower'][k]},{exp['ci_upper'][k]},{exp['p_values'][k]}\n"
                    )
            try:
                import numpy as np

                sp = np.array(soft_probs, dtype=float).reshape(1, -1)
                np.savetxt(
                    os.path.join(out, "stage_soft_vs_pred.csv"),
                    np.concatenate([sp, proba[:1]], axis=1),
                    delimiter=",",
                    fmt="%.6f",
                )
            except Exception:
                pass
        except Exception:
            pass

    return out


def run_train_external(
    X_train,
    y_train,
    X_ext=None,
    y_ext=None,
    save_dir="./runs",
    model_type="lgbm",
    boundaries=(30, 90),
    X_seq_train=None,
    X_static_train=None,
    X_seq_ext=None,
    X_static_ext=None,
    survival=False,
    durations=None,
    events=None,
    calibrate=True,
    calibration_method="sigmoid",
    temp=1.0,
    clip_min=0.8,
    clip_max=1.5,
    scaler="robust",
    cv_folds=3,
    seeds=None,
    ext_min_thresholds=None,
    coral=True,
    floor_eps=None,
    mix_alpha=0.0,
):
    import os, json, numpy as np
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(save_dir, ts)
    os.makedirs(out, exist_ok=True)

    if model_type == "lgbm":
        unique, counts = np.unique(y_train, return_counts=True)
        total = y_train.shape[0]
        cw = {
            int(c): float(total / (len(unique) * n))
            for c, n in zip(unique.tolist(), counts.tolist())
        }
        if scaler == "quantile":
            from sklearn.preprocessing import QuantileTransformer

            scaler = QuantileTransformer(
                output_distribution="normal", subsample=100000, random_state=42
            )
        else:
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        clf = LightGBMStageClassifier(
            class_weight=cw,
            calibrate=calibrate,
            calibration_method=calibration_method,
            cv_folds=cv_folds,
            seeds=(seeds or [42, 17, 99]),
        )
        clf.fit(X_train_s, y_train)
        proba_tr = clf.predict_proba(X_train_s)
        preds_tr = clf.predict(X_train_s)
        # 内部指标
        min_len = min(len(y_train), len(preds_tr))
        y_train = y_train[:min_len]
        preds_tr = preds_tr[:min_len]
        metrics_int = {}
        metrics_int.update(compute_basic_metrics(y_train, preds_tr))
        metrics_int.update(per_stage_recall(y_train, preds_tr))
        metrics_int.update(stage_misalignment(y_train, preds_tr))
        metrics_int["brier"] = brier_score(y_train, proba_tr[:min_len])
        with open(os.path.join(out, "internal_metrics.json"), "w") as f:
            json.dump(metrics_int, f, indent=2)
        # 外部评估
        if X_ext is not None and y_ext is not None:
            X_ext_s = scaler.transform(X_ext)
            X_ext_a = _coral_align(X_train_s, X_ext_s) if coral else X_ext_s
            proba_ext = clf.predict_proba(X_ext_a)
            preds_ext = clf.predict(X_ext_a)
            min_len2 = min(len(y_ext), len(preds_ext))
            y_ext = y_ext[:min_len2]
            preds_ext = preds_ext[:min_len2]
            metrics_ext = {}
            metrics_ext.update(compute_basic_metrics(y_ext, preds_ext))
            metrics_ext.update(per_stage_recall(y_ext, preds_ext))
            metrics_ext.update(stage_misalignment(y_ext, preds_ext))
            metrics_ext["brier"] = brier_score(y_ext, proba_ext[:min_len2])
            with open(os.path.join(out, "external_metrics.json"), "w") as f:
                json.dump(metrics_ext, f, indent=2)
            try:
                from src.calibration.stage_calibration import (
                    calibrate_per_stage,
                    apply_thresholds_predict,
                    optimize_thresholds_fbeta_per_class,
                    optimize_thresholds_with_distribution,
                )

                train_props = [
                    float(np.mean(y_train == c)) for c in range(proba_ext.shape[1])
                ]
                ext_props = [
                    float(np.mean(y_ext == c)) for c in range(proba_ext.shape[1])
                ]
                w = np.array(
                    [
                        (train_props[c] + 1e-6) / (ext_props[c] + 1e-6)
                        for c in range(proba_ext.shape[1])
                    ],
                    dtype=float,
                )
                w = np.clip(w, float(clip_min), float(clip_max))
                proba_ext_adj = proba_ext[:min_len2] * w.reshape(1, -1)
                denom = np.sum(proba_ext_adj, axis=1, keepdims=True)
                denom = np.where(denom == 0, 1.0, denom)
                proba_ext_adj = proba_ext_adj / denom
                if float(temp) > 0 and float(temp) != 1.0:
                    Pts = np.power(np.clip(proba_ext_adj, 1e-8, 1.0), 1.0 / float(temp))
                    denom2 = np.sum(Pts, axis=1, keepdims=True)
                    denom2 = np.where(denom2 == 0, 1.0, denom2)
                    proba_ext_adj = Pts / denom2
                if floor_eps is not None:
                    try:
                        epsv = np.array(
                            [float(x) for x in floor_eps], dtype=float
                        ).reshape(1, -1)
                        proba_ext_adj = np.maximum(proba_ext_adj, epsv)
                        denom3 = np.sum(proba_ext_adj, axis=1, keepdims=True)
                        denom3 = np.where(denom3 == 0, 1.0, denom3)
                        proba_ext_adj = proba_ext_adj / denom3
                    except Exception:
                        pass
                if float(mix_alpha) > 0.0:
                    pri = np.array(
                        [float(np.mean(y_ext == c)) for c in range(proba_ext.shape[1])],
                        dtype=float,
                    ).reshape(1, -1)
                    pri = pri / max(np.sum(pri), 1e-8)
                    proba_ext_adj = (1.0 - float(mix_alpha)) * proba_ext_adj + float(
                        mix_alpha
                    ) * pri
                    denom4 = np.sum(proba_ext_adj, axis=1, keepdims=True)
                    denom4 = np.where(denom4 == 0, 1.0, denom4)
                    proba_ext_adj = proba_ext_adj / denom4
                proba_ext_cal = calibrate_per_stage(proba_ext_adj, y_ext)
                betas_ext = [2.0, 1.5, 2.4, 2.0]
                th_ext, fb_ext = optimize_thresholds_fbeta_per_class(
                    proba_ext_cal, y_ext, betas_ext
                )
                with open(os.path.join(out, "thresholds_external.json"), "w") as f:
                    json.dump(
                        {
                            "beta": betas_ext,
                            "thresholds": list(map(float, th_ext)),
                            "f_beta": list(map(float, fb_ext)),
                        },
                        f,
                        indent=2,
                    )
                target_props_ext = [
                    float(np.mean(y_ext == c)) for c in range(proba_ext_cal.shape[1])
                ]
                target_props_ext = [max(p, 0.01) for p in target_props_ext]
                min_th = None
                if ext_min_thresholds is not None:
                    try:
                        min_th = [float(x) for x in ext_min_thresholds]
                    except Exception:
                        min_th = None
                min_th = min_th or [0.12, 0.10, 0.12, 0.12]
                th_dist_ext = optimize_thresholds_with_distribution(
                    proba_ext_cal,
                    y_ext,
                    target_props=target_props_ext,
                    beta=2.0,
                    min_thresholds=min_th,
                )
                preds_ext_th = apply_thresholds_predict(
                    proba_ext_cal,
                    th_dist_ext,
                    priority=[2, 1, 3, 0],
                    weights=[1.05, 1.02, 1.18, 1.12],
                    min_thresholds=min_th,
                    conf_min=0.30,
                )
                metrics_ext_th = {}
                metrics_ext_th.update(compute_basic_metrics(y_ext, preds_ext_th))
                metrics_ext_th.update(per_stage_recall(y_ext, preds_ext_th))
                metrics_ext_th.update(stage_misalignment(y_ext, preds_ext_th))
                metrics_ext_th["brier"] = brier_score(y_ext, proba_ext_cal)
                with open(
                    os.path.join(out, "external_metrics_thresholded.json"), "w"
                ) as f:
                    json.dump(metrics_ext_th, f, indent=2)
                plot_calibration_curves(
                    proba_ext_cal, y_ext, os.path.join(out, "calibration_external")
                )
            except Exception:
                pass
    else:
        from sklearn.preprocessing import RobustScaler

        X_static_train_s = None
        if X_static_train is not None:
            scaler = RobustScaler()
            X_static_train_s = scaler.fit_transform(X_static_train)
        model = train_transformer_stage(X_seq_train, y_train, X_static=X_static_train_s)
        wrapper = TorchStageWrapper(model)
        proba_tr = wrapper.predict_proba(
            X_seq_train,
            X_static_train_s if X_static_train_s is not None else X_static_train,
        )
        preds_tr = np.argmax(proba_tr, axis=1)
        min_len = min(len(y_train), len(preds_tr))
        y_train = y_train[:min_len]
        preds_tr = preds_tr[:min_len]
        metrics_int = {}
        metrics_int.update(compute_basic_metrics(y_train, preds_tr))
        metrics_int.update(per_stage_recall(y_train, preds_tr))
        metrics_int.update(stage_misalignment(y_train, preds_tr))
        metrics_int["brier"] = brier_score(y_train, proba_tr[:min_len])
        with open(os.path.join(out, "internal_metrics.json"), "w") as f:
            json.dump(metrics_int, f, indent=2)
        try:
            th_tr, fb_tr = optimize_thresholds_fbeta(
                proba_tr[:min_len], y_train, beta=2.0
            )
            with open(os.path.join(out, "thresholds_internal.json"), "w") as f:
                json.dump(
                    {"beta": 2.0, "thresholds": th_tr, "f_beta": fb_tr}, f, indent=2
                )
            plot_calibration_curves(
                proba_tr[:min_len], y_train, os.path.join(out, "calibration_internal")
            )
        except Exception:
            pass
        if X_seq_ext is not None and X_static_ext is not None and y_ext is not None:
            X_static_ext_a = None
            if X_static_train is not None and X_static_ext is not None:
                X_static_ext_s = scaler.transform(X_static_ext)
                X_static_ext_a = (
                    _coral_align(X_static_train_s, X_static_ext_s)
                    if coral
                    else X_static_ext_s
                )
            X_seq_ext_a = X_seq_ext
            try:
                if coral and X_seq_train is not None and X_seq_ext is not None:
                    D = int(X_seq_train.shape[-1])
                    Xs_flat = np.reshape(X_seq_train, (-1, D))
                    Xt_flat = np.reshape(X_seq_ext, (-1, D))
                    Xt_aligned_flat = _coral_align(Xs_flat, Xt_flat)
                    X_seq_ext_a = np.reshape(Xt_aligned_flat, X_seq_ext.shape)
            except Exception:
                X_seq_ext_a = X_seq_ext
            proba_ext = wrapper.predict_proba(
                X_seq_ext_a,
                X_static_ext_a if X_static_train is not None else None,
            )
            preds_ext = np.argmax(proba_ext, axis=1)
            min_len2 = min(len(y_ext), len(preds_ext))
            y_ext = y_ext[:min_len2]
            preds_ext = preds_ext[:min_len2]
            metrics_ext = {}
            metrics_ext.update(compute_basic_metrics(y_ext, preds_ext))
            metrics_ext.update(per_stage_recall(y_ext, preds_ext))
            metrics_ext.update(stage_misalignment(y_ext, preds_ext))
            metrics_ext["brier"] = brier_score(y_ext, proba_ext[:min_len2])
            with open(os.path.join(out, "external_metrics.json"), "w") as f:
                json.dump(metrics_ext, f, indent=2)
            try:
                from src.calibration.stage_calibration import (
                    calibrate_per_stage,
                    optimize_thresholds_fbeta_per_class,
                    optimize_thresholds_with_distribution,
                    apply_thresholds_predict,
                )

                train_props = (
                    [float(np.mean(y_train == c)) for c in range(proba_ext.shape[1])]
                    if y_train is not None
                    else [0.25] * proba_ext.shape[1]
                )
                ext_props = [
                    float(np.mean(y_ext == c)) for c in range(proba_ext.shape[1])
                ]
                w = np.array(
                    [
                        (train_props[c] + 1e-6) / (ext_props[c] + 1e-6)
                        for c in range(proba_ext.shape[1])
                    ],
                    dtype=float,
                )
                w = np.clip(w, 0.8, 1.5)
                P_adj = proba_ext[:min_len2] * w.reshape(1, -1)
                denom = np.sum(P_adj, axis=1, keepdims=True)
                denom = np.where(denom == 0, 1.0, denom)
                P_adj = P_adj / denom
                P_cal = calibrate_per_stage(P_adj, y_ext)
                betas_ext = [2.5, 2.0, 1.5, 1.2]
                th_ext, fb_ext = optimize_thresholds_fbeta_per_class(
                    P_cal, y_ext, betas_ext
                )
                with open(os.path.join(out, "thresholds_external.json"), "w") as f:
                    json.dump(
                        {
                            "beta": betas_ext,
                            "thresholds": list(map(float, th_ext)),
                            "f_beta": list(map(float, fb_ext)),
                        },
                        f,
                        indent=2,
                    )
                target_props_ext = [
                    float(np.mean(y_ext == c)) for c in range(P_cal.shape[1])
                ]
                target_props_ext = [max(p, 0.01) for p in target_props_ext]
                th_dist_ext = optimize_thresholds_with_distribution(
                    P_cal,
                    y_ext,
                    target_props=target_props_ext,
                    beta=2.0,
                    min_thresholds=[0.12, 0.05, 0.12, 0.12],
                )
                preds_ext_th = apply_thresholds_predict(
                    P_cal,
                    th_dist_ext,
                    priority=[1, 0, 2, 3],
                    weights=[1.15, 1.08, 1.0, 0.98],
                    min_thresholds=[0.12, 0.05, 0.12, 0.12],
                    conf_min=0.25,
                )
                metrics_ext_th = {}
                metrics_ext_th.update(compute_basic_metrics(y_ext, preds_ext_th))
                metrics_ext_th.update(per_stage_recall(y_ext, preds_ext_th))
                metrics_ext_th.update(stage_misalignment(y_ext, preds_ext_th))
                metrics_ext_th["brier"] = brier_score(y_ext, P_cal)
                with open(
                    os.path.join(out, "external_metrics_thresholded.json"), "w"
                ) as f:
                    json.dump(metrics_ext_th, f, indent=2)
                plot_calibration_curves(
                    P_cal, y_ext, os.path.join(out, "calibration_external")
                )
            except Exception:
                pass
        # 生存辅助（仅当传入durations/events且survival=True）
        if (
            survival
            and durations is not None
            and events is not None
            and X_static_train is not None
        ):
            try:
                from src.survival.cox_adapter import CoxAdapter
                import pandas as pd

                df_cov = pd.DataFrame(X_static_train)
                cox = CoxAdapter()
                cox.fit(df_cov, durations, events)
                times = np.array(
                    [boundaries[0], boundaries[1], float(np.max(durations))]
                )
                sf = cox.survival_at(df_cov, times)
                soft_probs = map_survival_to_stage_probs(
                    sf.iloc[:, 0], boundaries=boundaries
                )
                exp = cox.explain()
                with open(os.path.join(out, "coefficients.csv"), "w") as f:
                    f.write("name,coef,hr,ci_lower,ci_upper,p\n")
                    for k in exp["coefficients"].keys():
                        f.write(
                            f"{k},{exp['coefficients'][k]},{exp['hazard_ratio'][k]},{exp['ci_lower'][k]},{exp['ci_upper'][k]},{exp['p_values'][k]}\n"
                        )
                try:
                    sp = np.array(soft_probs, dtype=float).reshape(1, -1)
                    np.savetxt(
                        os.path.join(out, "stage_soft_vs_pred.csv"),
                        np.concatenate([sp, proba_tr[:1]], axis=1),
                        delimiter=",",
                        fmt="%.6f",
                    )
                except Exception:
                    pass
            except Exception:
                pass

    return out
