import numpy as np
import matplotlib.pyplot as plt

def calculate_net_benefit(y_true, y_prob, thresholds):
    """
    Calculate Net Benefit for Decision Curve Analysis.
    Net Benefit = (True Positives / N) - (False Positives / N) * (threshold / (1 - threshold))
    """
    net_benefits = []
    
    n_samples = len(y_true)
    
    for pt in thresholds:
        # Binary prediction based on threshold
        y_pred = (y_prob >= pt).astype(int)
        
        # Calculate true positives and false positives
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        if pt == 1.0:
            nb = 0.0
        else:
            nb = (tp / n_samples) - (fp / n_samples) * (pt / (1 - pt))
            
        net_benefits.append(nb)
        
    return np.array(net_benefits)

def plot_dca(y_true, models_dict, thresholds=None, ax=None):
    """
    Plot Decision Curve Analysis.
    models_dict: dict of {name: y_prob}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
        
    n_samples = len(y_true)
    prevalence = np.sum(y_true) / n_samples
    
    # 1. Treat All strategy
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, nb_all, color='gray', linestyle='--', label='Treat All')
    
    # 2. Treat None strategy (Net Benefit is exactly 0)
    ax.plot(thresholds, np.zeros_like(thresholds), color='black', label='Treat None')
    
    # 3. Models
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, (name, y_prob) in enumerate(models_dict.items()):
        nb_model = calculate_net_benefit(y_true, y_prob, thresholds)
        # Ensure we don't plot below 0 for better visualization
        # nb_model = np.maximum(nb_model, -0.05)
        ax.plot(thresholds, nb_model, color=colors[i % len(colors)], linewidth=2, label=name)
        
    ax.set_ylim([-0.05, max(prevalence * 1.2, 0.2)])
    ax.set_xlim([0, 1.0])
    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title('Decision Curve Analysis (DCA)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return ax
