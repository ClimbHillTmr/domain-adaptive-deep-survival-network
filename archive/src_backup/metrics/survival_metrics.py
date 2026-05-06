import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

try:
    from sksurv.metrics import concordance_index_censored as _sksurv_c_index
    from sksurv.metrics import brier_score as _sksurv_brier_score
    from sksurv.metrics import integrated_brier_score as _sksurv_integrated_brier_score
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
        except ValueError as e:
            import logging
            logging.getLogger(__name__).warning(
                f"scikit-survival C-index failed (invalid input): {e}, using fallback"
            )
        except TypeError as e:
            import logging
            logging.getLogger(__name__).warning(
                f"scikit-survival C-index failed (type error): {e}, using fallback"
            )

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


def _risk_to_survival_probability(risk, time, eval_times):
    """
    Convert risk scores to survival probabilities at specific time points.
    
    Uses the relationship: S(t|x) = S0(t)^exp(risk)
    where S0(t) is the baseline survival function estimated from data.
    
    Parameters:
    -----------
    risk : (N,) float array, predicted log-hazard ratios
    time : (N,) float array, observed survival times
    eval_times : array, time points at which to evaluate survival probability
    
    Returns:
    --------
    surv_probs : (N, len(eval_times)) array of survival probabilities
    """
    from sksurv.nonparametric import kaplan_meier_estimator
    
    # Estimate baseline survival function using Kaplan-Meier
    unique_times, survival_probs = kaplan_meier_estimator(
        np.ones(len(time), dtype=bool),  # All events for baseline estimation
        time
    )
    
    # For each sample, compute S(t|x) = S0(t)^exp(risk - mean_risk)
    # Center risk scores to avoid numerical issues
    risk_centered = risk - np.mean(risk)
    
    surv_probs_matrix = np.zeros((len(risk), len(eval_times)))
    
    for i, t in enumerate(eval_times):
        # Find baseline survival probability at time t
        mask = unique_times >= t
        if mask.sum() == 0:
            s0_t = 0.0
        else:
            s0_t = survival_probs[mask][0]
        
        # Apply Cox model: S(t|x) = S0(t)^exp(risk_centered)
        # Clip to avoid numerical issues
        exp_risk = np.exp(np.clip(risk_centered, -10, 10))
        surv_probs_matrix[:, i] = np.clip(s0_t ** exp_risk, 0.0, 1.0)
    
    return surv_probs_matrix


def integrated_brier_score(event, time, risk, times=None):
    """
    Compute Integrated Brier Score (IBS) for survival data.
    
    Uses scikit-survival's integrated_brier_score which handles
    IPCW weighting and time integration automatically.
    
    Parameters:
    -----------
    event : (N,) boolean, event indicator
    time : (N,) float, survival times
    risk : (N,) float, predicted risk scores (log-hazard ratios)
    times : optional, time points at which to evaluate
    
    Returns:
    --------
    ibs : float, integrated Brier score (lower is better, range [0, 1])
    """
    if not HAS_SKSURV:
        import logging
        logging.getLogger(__name__).error(
            "scikit-survival is required for proper IBS computation. "
            "Install with: pip install scikit-survival"
        )
        return np.nan
    
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float).flatten()
    
    # Define evaluation times - ensure they're within data range
    if times is None:
        event_times = time[event]
        if len(event_times) == 0:
            return np.nan
        times = np.percentile(event_times, [25, 50, 75])
        times = times[times > 0]
    
    # Ensure times are within the range of observed times
    # sksurv requires: times.min() >= time.min() AND times.max() < time.max()
    time_min = time.min()
    time_max = time.max()
    original_len = len(times)
    times = times[(times >= time_min) & (times < time_max)]
    times = np.unique(times)  # Remove duplicates
    
    if len(times) < 2:
        import logging
        logging.getLogger(__name__).warning(
            f"IBS: too few valid time points after filtering ({len(times)}, was {original_len})"
        )
        return np.nan
    
    # Convert risk scores to survival probabilities USING FILTERED TIMES
    # This is critical: surv_probs columns must match filtered times
    try:
        surv_probs = _risk_to_survival_probability(risk, time, times)
    except (ValueError, RuntimeError) as e:
        import logging
        logging.getLogger(__name__).warning(
            f"Survival probability conversion failed: {e}"
        )
        return np.nan
    
    # Verify shapes match
    if surv_probs.shape[1] != len(times):
        import logging
        logging.getLogger(__name__).warning(
            f"IBS: shape mismatch - surv_probs has {surv_probs.shape[1]} cols, "
            f"but times has {len(times)} elements"
        )
        return np.nan
    
    # Compute IBS using scikit-survival
    try:
        survival_struct = np.array(
            [(event[j], time[j]) for j in range(len(time))],
            dtype=[('event', '?'), ('time', '<f8')]
        )
        
        ibs = _sksurv_integrated_brier_score(
            survival_struct,  # training data for censoring estimation
            survival_struct,  # test data
            surv_probs,       # (n_samples, n_times)
            times             # time points
        )
        return float(ibs)
        
    except (ValueError, TypeError) as e:
        import logging
        logging.getLogger(__name__).warning(
            f"IBS computation failed: {e}"
        )
        return np.nan
