import numpy as np

def diebold_mariano(e1, e2, h=1):
    e1 = np.array(e1)
    e2 = np.array(e2)
    d = e1**2 - e2**2
    d_mean = np.mean(d)
    # Newey-West variance estimate
    T = len(d)
    gamma0 = np.var(d, ddof=1)
    var_d = gamma0
    for lag in range(1, h):
        gamma = np.cov(d[:-lag], d[lag:])[0,1]
        var_d += 2 * (1 - lag / (h+1)) * gamma
    dm_stat = d_mean / np.sqrt(var_d / T + 1e-8)
    return float(dm_stat)