import numpy as np
import matplotlib.pyplot as plt

def calibrate_per_stage(proba, y, method='isotonic'):
    import numpy as np
    y = np.asarray(y)
    P = np.asarray(proba, dtype=float)
    C = P.shape[1]
    if method == 'isotonic':
        try:
            from sklearn.isotonic import IsotonicRegression
            Pc = np.zeros_like(P)
            for c in range(C):
                ir = IsotonicRegression(out_of_bounds='clip')
                ir.fit(P[:, c], (y == c).astype(float))
                Pc[:, c] = ir.transform(P[:, c])
            denom = np.sum(Pc, axis=1, keepdims=True)
            denom = np.where(denom == 0, 1.0, denom)
            Pc = Pc / denom
            return Pc
        except Exception:
            return P
    return P

def optimize_thresholds_fbeta(proba, y, beta=2.0):
    y = np.asarray(y)
    C = proba.shape[1]
    thresholds = np.zeros(C, dtype=float)
    scores = np.zeros(C, dtype=float)
    for c in range(C):
        best_thr, best_score = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 19):
            pred_c = (proba[:, c] >= thr).astype(int)
            tp = np.sum((y == c) & (pred_c == 1))
            fp = np.sum((y != c) & (pred_c == 1))
            fn = np.sum((y == c) & (pred_c == 0))
            if tp == 0 and fp == 0 and fn == 0:
                score = 0.0
            else:
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                b2 = beta * beta
                denom = max(b2 * precision + recall, 1e-8)
                score = (1 + b2) * precision * recall / denom
            if score > best_score:
                best_score = score
                best_thr = thr
        thresholds[c] = best_thr
        scores[c] = best_score
    return thresholds, scores

def optimize_thresholds_fbeta_per_class(proba, y, betas):
    import numpy as np
    y = np.asarray(y)
    C = proba.shape[1]
    thresholds = np.zeros(C, dtype=float)
    scores = np.zeros(C, dtype=float)
    for c in range(C):
        b2 = float(betas[c]) ** 2
        best_thr, best_score = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 19):
            pred_c = (proba[:, c] >= thr).astype(int)
            tp = np.sum((y == c) & (pred_c == 1))
            fp = np.sum((y != c) & (pred_c == 1))
            fn = np.sum((y == c) & (pred_c == 0))
            if tp == 0 and fp == 0 and fn == 0:
                score = 0.0
            else:
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                denom = max(b2 * precision + recall, 1e-8)
                score = (1 + b2) * precision * recall / denom
            if score > best_score:
                best_score = score
                best_thr = thr
        thresholds[c] = best_thr
        scores[c] = best_score
    return thresholds, scores

def bootstrap_ci_per_stage(proba, n_boot=200, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = proba.shape[0]
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(proba[idx].mean(axis=0))
    means = np.stack(means, axis=0)
    lower = np.quantile(means, alpha / 2, axis=0)
    upper = np.quantile(means, 1 - alpha / 2, axis=0)
    return lower, upper

def plot_calibration_curves(proba, y, out_path_prefix, bins=10):
    y = np.asarray(y)
    C = proba.shape[1]
    for c in range(C):
        p = proba[:, c]
        bin_edges = np.linspace(0, 1, bins + 1)
        obs = []
        mids = []
        for i in range(bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (p >= lo) & (p < hi)
            if np.sum(mask) == 0:
                obs.append(0.0)
            else:
                obs.append(np.mean((y[mask] == c).astype(float)))
            mids.append((lo + hi) / 2.0)
        plt.figure(figsize=(5,4))
        plt.plot(mids, obs, marker='o', label='Observed')
        plt.plot([0,1],[0,1],'--', label='Perfect')
        plt.xlabel('Predicted probability')
        plt.ylabel('Observed frequency')
        plt.title(f'Calibration class {c}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_path_prefix}_class{c}.png", dpi=200)
        plt.close()

def apply_thresholds_predict(proba, thresholds, priority=None, weights=None, min_thresholds=None, conf_min=None):
    import numpy as np
    P = np.asarray(proba, dtype=float)
    C = P.shape[1]
    thr = np.asarray(thresholds).reshape(1, C)
    if min_thresholds is not None:
        mthr = np.asarray(min_thresholds).reshape(1, C)
        thr = np.maximum(thr, mthr)
    mask = (P >= thr).astype(int)
    any_hit = np.sum(mask, axis=1)
    w = np.ones((1, C)) if weights is None else np.asarray(weights, dtype=float).reshape(1, C)
    scores = P * w
    preds = np.argmax(scores, axis=1)
    if conf_min is not None:
        maxp = np.max(P, axis=1)
        low_conf = maxp < float(conf_min)
        if np.any(low_conf):
            idx = np.where(low_conf)[0]
            preds[idx] = np.argmax(P[idx], axis=1)
    one = (any_hit == 1)
    if np.any(one):
        idx = np.where(one)[0]
        preds[idx] = np.argmax(mask[idx], axis=1)
    multi = (any_hit > 1)
    if np.any(multi) and priority is not None:
        idx = np.where(multi)[0]
        pr = np.array(priority, dtype=int)
        for i in idx:
            for c in pr:
                if mask[i, c] == 1:
                    preds[i] = c
                    break
    return preds

def optimize_thresholds_with_distribution(proba, y, target_props=None, beta=2.0, min_thresholds=None):
    import numpy as np
    y = np.asarray(y)
    P = np.asarray(proba, dtype=float)
    n, C = P.shape
    if target_props is None:
        props = np.zeros(C, dtype=float)
        for c in range(C):
            props[c] = float(np.mean(y == c))
    else:
        props = np.asarray(target_props, dtype=float)
        props = props / max(np.sum(props), 1e-8)
    thr_out = np.zeros(C, dtype=float)
    for c in range(C):
        candidates = np.linspace(0.05, 0.95, 19)
        best_thr, best_obj = 0.5, 1e9
        b2 = (float(beta) ** 2)
        for t in candidates:
            t_eff = max(float(t), float(min_thresholds[c])) if min_thresholds is not None else float(t)
            pred_c = (P[:, c] >= t_eff).astype(int)
            prop_pred = float(np.mean(pred_c))
            tp = np.sum((y == c) & (pred_c == 1))
            fp = np.sum((y != c) & (pred_c == 1))
            fn = np.sum((y == c) & (pred_c == 0))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            fbeta = (1 + b2) * precision * recall / max(b2 * precision + recall, 1e-8)
            obj = (prop_pred - props[c]) ** 2 + (1.0 - fbeta)
            if obj < best_obj:
                best_obj = obj
                best_thr = t_eff
        thr_out[c] = best_thr
    return thr_out