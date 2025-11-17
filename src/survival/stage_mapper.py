import numpy as np

def map_survival_to_stage_probs(S_t, boundaries=(30,90)):
    # S_t: survival function values at specific minutes per individual
    # 返回按边界分箱的阶段概率（未发生/早期/中期/晚期）
    b1, b2 = boundaries
    # 简化：阶段概率由区间生存函数差分近似
    # p(早期) ~ 1 - S(b1)
    # p(中期) ~ S(b1) - S(b2)
    # p(晚期) ~ S(b2) - S(T_end)
    # p(未发生) ~ S(T_end)
    T_end = S_t.index[-1]
    vals = S_t.values
    def val_at(t):
        if t in S_t.index:
            return float(S_t.loc[t])
        # 最近邻插值
        idx = np.searchsorted(S_t.index.values, t)
        idx = np.clip(idx, 1, len(S_t.index)-1)
        return float(vals[idx])
    Sb1 = val_at(b1)
    Sb2 = val_at(b2)
    Send = val_at(T_end)
    p_early = max(0.0, 1.0 - Sb1)
    p_mid = max(0.0, Sb1 - Sb2)
    p_late = max(0.0, Sb2 - Send)
    p_none = max(0.0, Send)
    p = np.array([p_none, p_early, p_mid, p_late], dtype=float)
    s = p.sum()
    if s > 0:
        p /= s
    return p