import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def compute_prior_adjusted_risk(risk_scores_source, source_prevalence, target_prevalence):
    """
    Adjust predicted risk scores from source domain to target domain
    using prior probability adjustment (Prior-based Calibration).
    Formula: P(y=1|x,T) = P(y=1|x,S) * [P(y=1|T)/P(y=1|S)] * [P(x|S)/P(x|T)]
    Assuming P(x|y,S) ≈ P(x|y,T) after Domain Adaptation.
    """
    # Convert hazards to approximate probabilities using sigmoid for calibration
    # Note: Cox model outputs log-hazard ratio, we approximate P(event) for short-term
    p_source = 1 / (1 + np.exp(-risk_scores_source))
    
    odds_source = p_source / (1 - p_source + 1e-7)
    prior_odds_ratio = (target_prevalence / (1 - target_prevalence)) / (source_prevalence / (1 - source_prevalence))
    
    odds_target = odds_source * prior_odds_ratio
    p_target = odds_target / (1 + odds_target)
    
    # Convert back to risk scores for evaluation metrics that expect hazards
    adjusted_risk_scores = np.log(p_target / (1 - p_target + 1e-7))
    
    return p_target, adjusted_risk_scores

def plot_calibration_curve(y_true, y_prob, n_bins=10, name="Model", ax=None):
    """Plot calibration curve (reliability diagram)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
    # Bin predictions
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    ax.plot(prob_pred, prob_true, "s-", label="%s" % (name,))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Mean predicted value")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="lower right")
    
    # Calculate ECE (Expected Calibration Error)
    ece = np.sum(bin_total[nonzero] * np.abs(prob_true - prob_pred)) / np.sum(bin_total[nonzero])
    
    return ax, ece
