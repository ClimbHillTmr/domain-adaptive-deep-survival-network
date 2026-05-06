"""
Calibration curves and Decision Curve Analysis (DCA) for clinical utility evaluation.
"""

import numpy as np
from scipy import stats
import logging

log = logging.getLogger(__name__)


def calculate_calibration_probs(risk_scores, events, n_bins=10):
    """
    Calculate calibration data for probability calibration plot.
    
    Parameters:
    -----------
    risk_scores : array-like
        Predicted risk scores (probabilities or continuous scores)
    events : array-like
        Binary event indicators (1=event, 0=censored)
    n_bins : int
        Number of bins for calibration
        
    Returns:
    --------
    bin_centers, observed_probs, predicted_probs, bin_counts
    """
    risk_scores = np.asarray(risk_scores).flatten()
    events = np.asarray(events).astype(float)
    
    # Normalize risk scores to [0, 1] range
    if risk_scores.max() > 1.0 or risk_scores.min() < 0.0:
        risk_min = risk_scores.min()
        risk_max = risk_scores.max()
        if risk_max - risk_min > 0:
            risk_scores = (risk_scores - risk_min) / (risk_max - risk_min)
        else:
            risk_scores = np.zeros_like(risk_scores)
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    observed_probs = []
    predicted_probs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (risk_scores >= bin_edges[i]) & (risk_scores < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (risk_scores >= bin_edges[i]) & (risk_scores <= bin_edges[i + 1])
        
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            observed_probs.append(events[mask].mean())
            predicted_probs.append(risk_scores[mask].mean())
            bin_counts.append(mask.sum())
    
    return (
        np.array(bin_centers),
        np.array(observed_probs),
        np.array(predicted_probs),
        np.array(bin_counts),
    )


def calculate_ece(risk_scores, events, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the weighted average difference between predicted
    and observed probabilities across bins.
    """
    bin_centers, observed_probs, predicted_probs, bin_counts = calculate_calibration_probs(
        risk_scores, events, n_bins
    )
    
    if len(bin_counts) == 0:
        return np.nan
    
    total = bin_counts.sum()
    ece = np.sum((bin_counts / total) * np.abs(observed_probs - predicted_probs))
    return ece


def calculate_brier_score(risk_scores, events):
    """Calculate Brier score for binary outcomes."""
    risk_scores = np.asarray(risk_scores).flatten()
    events = np.asarray(events).astype(float)
    
    # Normalize to [0, 1]
    if risk_scores.max() > 1.0 or risk_scores.min() < 0.0:
        risk_min = risk_scores.min()
        risk_max = risk_scores.max()
        if risk_max - risk_min > 0:
            risk_scores = (risk_scores - risk_min) / (risk_max - risk_min)
        else:
            risk_scores = np.zeros_like(risk_scores)
    
    return np.mean((risk_scores - events) ** 2)


def decision_curve_analysis(risk_scores, events, threshold_range=None, n_thresholds=100):
    """
    Decision Curve Analysis (DCA) for clinical utility evaluation.
    
    Parameters:
    -----------
    risk_scores : array-like
        Predicted risk scores
    events : array-like
        Binary event indicators
    threshold_range : tuple, optional
        (min_threshold, max_threshold), defaults to (0, 1)
    n_thresholds : int
        Number of threshold points to evaluate
        
    Returns:
    --------
    thresholds, net_benefit_model, net_benefit_all, net_benefit_none
    """
    risk_scores = np.asarray(risk_scores).flatten()
    events = np.asarray(events).astype(float)
    n = len(events)
    
    # Normalize risk scores
    if risk_scores.max() > 1.0 or risk_scores.min() < 0.0:
        risk_min = risk_scores.min()
        risk_max = risk_scores.max()
        if risk_max - risk_min > 0:
            risk_scores = (risk_scores - risk_min) / (risk_max - risk_min)
        else:
            risk_scores = np.zeros_like(risk_scores)
    
    if threshold_range is None:
        threshold_range = (0.01, 0.99)
    
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = []
    
    event_rate = events.mean()
    
    for pt in thresholds:
        # Model strategy: treat if predicted risk >= threshold
        treated = (risk_scores >= pt).astype(float)
        tp = (treated * events).sum()
        fp = (treated * (1 - events)).sum()
        
        # Net benefit = TP/n - FP/n * (pt / (1 - pt))
        nb_model = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefit_model.append(nb_model)
        
        # Treat all strategy
        nb_all = event_rate - (1 - event_rate) * (pt / (1 - pt))
        net_benefit_all.append(nb_all)
        
        # Treat none strategy
        net_benefit_none.append(0.0)
    
    return (
        thresholds,
        np.array(net_benefit_model),
        np.array(net_benefit_all),
        np.array(net_benefit_none),
    )


def calculate_nri(
    risk_scores_old, risk_scores_new, events, category_bounds=None
):
    """
    Calculate Net Reclassification Improvement (NRI).
    
    Parameters:
    -----------
    risk_scores_old : array-like
        Risk scores from old model
    risk_scores_new : array-like
        Risk scores from new model
    events : array-like
        Binary event indicators
    category_bounds : list, optional
        Risk category boundaries, e.g., [0, 0.1, 0.2, 0.5, 1.0]
        
    Returns:
    --------
    nri, nri_events, nri_non_events
    """
    if category_bounds is None:
        category_bounds = [0, 0.1, 0.2, 0.5, 1.0]
    
    events = np.asarray(events).astype(bool)
    n_categories = len(category_bounds) - 1
    
    def categorize(risk_scores):
        return np.digitize(risk_scores, category_bounds[1:-1])
    
    old_cats = categorize(risk_scores_old)
    new_cats = categorize(risk_scores_new)
    
    # Events: improvement = moved up
    event_mask = events
    if event_mask.sum() > 0:
        improved_events = (new_cats[event_mask] > old_cats[event_mask]).sum()
        worsened_events = (new_cats[event_mask] < old_cats[event_mask]).sum()
        nri_events = (improved_events - worsened_events) / event_mask.sum()
    else:
        nri_events = 0.0
    
    # Non-events: improvement = moved down
    nonevent_mask = ~events
    if nonevent_mask.sum() > 0:
        improved_nonevents = (new_cats[nonevent_mask] < old_cats[nonevent_mask]).sum()
        worsened_nonevents = (new_cats[nonevent_mask] > old_cats[nonevent_mask]).sum()
        nri_nonevents = (improved_nonevents - worsened_nonevents) / nonevent_mask.sum()
    else:
        nri_nonevents = 0.0
    
    nri = nri_events + nri_nonevents
    return nri, nri_events, nri_nonevents
