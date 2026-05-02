import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

try:
    from sksurv.metrics import concordance_index_censored as _sksurv_c_index
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False


def concordance_index_censored(event, time, risk):
    """
    Concordance index for right-censored data.
    Uses scikit-survival (O(n log n)) if available, falls back to pure Python O(n²).

    event: (N,) boolean array, True = event occurred
    time: (N,) float array, survival times
    risk: (N,) float array, predicted risk scores (higher = more risk)
    """
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float)

    n = len(time)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if HAS_SKSURV:
        try:
            c_index, concordant, discordant, tied_risk, comparable = _sksurv_c_index(
                event, time, risk
            )
            return c_index, concordant, discordant, tied_risk, comparable
        except Exception:
            pass

    # Fallback: pure Python O(n²) implementation
    concordant = 0
    discordant = 0
    tied_risk = 0
    comparable = 0

    for i in range(n):
        for j in range(i + 1, n):
            if event[i] and event[j]:
                if (time[i] < time[j] and risk[i] > risk[j]) or \
                   (time[j] < time[i] and risk[j] > risk[i]):
                    concordant += 1
                elif risk[i] == risk[j]:
                    tied_risk += 1
                else:
                    discordant += 1
                comparable += 1

            elif event[i] and not event[j]:
                if time[i] <= time[j]:
                    comparable += 1
                    if risk[i] > risk[j]:
                        concordant += 1
                    elif risk[i] < risk[j]:
                        discordant += 1
                    else:
                        tied_risk += 1

            elif not event[i] and event[j]:
                if time[j] <= time[i]:
                    comparable += 1
                    if risk[j] > risk[i]:
                        concordant += 1
                    elif risk[j] < risk[i]:
                        discordant += 1
                    else:
                        tied_risk += 1

    if comparable == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    c_index = (concordant + 0.5 * tied_risk) / comparable

    return c_index, concordant, discordant, tied_risk, comparable


def integrated_brier_score(event, time, risk, times=None):
    """
    Compute Integrated Brier Score (IBS) for survival data.

    event: (N,) boolean, event indicator
    time: (N,) float, survival times
    risk: (N,) float, predicted risk scores
    times: optional, time points at which to evaluate
    """
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float).flatten()

    if times is None:
        times = np.percentile(time[event], np.linspace(10, 90, 10))
        times = times[times > 0]

    n = len(time)
    ibs_values = []

    for t in times:
        at_risk = time >= t
        n_at_risk = at_risk.sum()

        if n_at_risk == 0:
            continue

        y_true = event.astype(float)
        y_pred = risk

        se = (y_pred - y_true) ** 2

        ibs_t = se[at_risk].mean()
        ibs_values.append(ibs_t)

    if len(ibs_values) == 0:
        return np.nan

    ibs = np.mean(ibs_values)
    return ibs
