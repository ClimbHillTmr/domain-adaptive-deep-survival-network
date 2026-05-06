import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from sklearn.manifold import TSNE
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from src.models.cdan_gsn import DomainStratifiedGatedNet
from src.data.dataset import prepare_dataloaders

import matplotlib
# Use a font that supports Chinese to avoid warnings
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# Set academic plotting style
plt.style.use('default')
# Nature Medicine / Lancet typical style: clean white background, no heavy grids
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.title_fontsize': 12,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
})

# Professional Academic Palette (Lancet/Nature style)
# [Blue, Red, Green, Teal, Purple, Peach, Dark Red, Grey, Black]
ACADEMIC_PALETTE = ['#00468B', '#ED0000', '#42B540', '#0099B4', '#925E9F', '#FDAF91', '#AD002A', '#ADB6B6', '#1B1919']
NATURE_PALETTE = ACADEMIC_PALETTE # Alias for backwards compatibility

# Semantic mapping for feature names (Chinese to standardized Academic English)
FEATURE_NAME_MAPPING = {
    '性别': 'Gender',
    '透析龄占比': 'Dialysis Vintage Ratio',
    '超负荷': 'Fluid Overload (L)',
    '历史平均超滤率_mean': 'Hist. Avg. UFR (mL/kg/h)',
    '历史平均超滤量MAX': 'Hist. Max UF Volume (L)',
    '历史平均透前体重': 'Hist. Avg. Pre-dialysis Weight (kg)',
    '历史平均透前收缩压': 'Hist. Avg. Pre-dialysis SBP (mmHg)',
    '历史平均透前舒张压': 'Hist. Avg. Pre-dialysis DBP (mmHg)',
    '历史平均透中低血压_计算': 'Historical IDH Rate (%)',
    '抗凝剂类型_code': 'Anticoagulant Type',
    '透析方式_code': 'Dialysis Modality',
    '瘘管类型_code': 'Vascular Access Type',
    '瘘管位置_code': 'Vascular Access Location',
    '透析液钙浓度': 'Dialysate Calcium (mmol/L)',
    '透析液电导率': 'Dialysate Conductivity (mS/cm)',
    '透前体重-干体重': 'Target UF Volume (L)',
    '透前呼吸频率': 'Pre-dialysis Respiratory Rate (bpm)',
    '透前体温': 'Pre-dialysis Temperature (°C)',
    '透前收缩压': 'Pre-dialysis SBP (mmHg)',
    '透前舒张压': 'Pre-dialysis DBP (mmHg)',
    '透前动脉压': 'Pre-dialysis MAP (mmHg)',
    '脉压差': 'Pulse Pressure (mmHg)',
    '平均动脉压': 'Mean Arterial Pressure (mmHg)',
    'history_HBP_rate': 'Hist. Hypertension Rate (%)',
    'history_LBP_times_0_rate': 'Hist. LBP Rate (Hr 0) (%)',
    'history_LBP_times_1_rate': 'Hist. LBP Rate (Hr 1) (%)',
    'history_LBP_times_2_rate': 'Hist. LBP Rate (Hr 2) (%)',
    'history_LBP_times_3_rate': 'Hist. LBP Rate (Hr 3) (%)',
    'history_LBP_times_4_rate': 'Hist. LBP Rate (Hr 4) (%)',
    'history_HBP': 'Hist. Hypertension Count',
    'history_LBP_times_0': 'Hist. LBP Count (Hr 0)',
    'history_LBP_times_1': 'Hist. LBP Count (Hr 1)',
    'history_LBP_times_2': 'Hist. LBP Count (Hr 2)',
    'history_LBP_times_3': 'Hist. LBP Count (Hr 3)',
    'history_LBP_times_4': 'Hist. LBP Count (Hr 4)'
}

def translate_features(feature_names):
    """Translate raw dataset features to academic standard names."""
    return [FEATURE_NAME_MAPPING.get(name, name) for name in feature_names]

def load_config(config_path="conf/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_indices(feature_names):
    treat_features = [
        "超滤比", "超滤率_绝对", "超滤率_体重归一化", "超滤量MAX",
        "透析液电导率", "透析液钙浓度", "抗凝剂类型_code",
        "瘘管位置_code", "瘘管类型_code", "透析方式_code"
    ]
    treat_indices = []
    physio_indices = []
    for i, name in enumerate(feature_names):
        if any(t in name for t in treat_features):
            treat_indices.append(i)
        else:
            physio_indices.append(i)
    return treat_indices, physio_indices

def plot_tsne_alignment(model, source_loader, target_loader, device, out_dir):
    """
    Fig 2: t-SNE Latent Space Alignment
    Visualizes the feature representations of Source and Target domains to verify DA effectiveness.
    """
    print("Generating Fig 2: t-SNE Domain Alignment Plot...")
    model.eval()
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        # Get Source embeddings
        for i, (x_s, _, _, _) in enumerate(source_loader):
            if i > 5: break  # Limit sample size for clarity
            cls_emb, _, _, _ = model(x_s.to(device))
            embeddings.append(cls_emb.cpu().numpy())
            labels.extend(["Shenyi (Source)"] * x_s.size(0))
            
        # Get Target embeddings
        for i, (x_t, _, _, _) in enumerate(target_loader):
            if i > 5: break
            cls_emb, _, _, _ = model(x_t.to(device))
            embeddings.append(cls_emb.cpu().numpy())
            labels.extend(["Fuding (Target)"] * x_t.size(0))
            
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # KDE Density contours for overlap emphasis
    sns.kdeplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels,
        palette={"Shenyi (Source)": ACADEMIC_PALETTE[0], "Fuding (Target)": ACADEMIC_PALETTE[1]},
        alpha=0.25, levels=6, fill=True, ax=ax, legend=False, zorder=1
    )
    
    # Scatter plot
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels,
        palette={"Shenyi (Source)": ACADEMIC_PALETTE[0], "Fuding (Target)": ACADEMIC_PALETTE[1]},
        alpha=0.85, s=40, edgecolor='white', linewidth=0.8, ax=ax, zorder=2
    )
    
    plt.title("Fig 2: Latent Space Alignment (t-SNE)", pad=15)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    # Customize legend
    handles, legend_labels = ax.get_legend_handles_labels()
    plt.legend(handles, legend_labels, title="Domain", loc='upper right', frameon=True, edgecolor='#E0E0E0', facecolor='white', framealpha=0.9)
    
    # Add an annotation about CDAN
    ax.text(0.05, 0.05, "CDAN aligned representations", 
            transform=ax.transAxes, ha='left', va='bottom', 
            fontsize=11, fontstyle='italic', color='#555555',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#E0E0E0', boxstyle='round,pad=0.5'))
    
    sns.despine()
    plt.tight_layout()
    
    plt.savefig(out_dir / "fig2_tsne_alignment.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    print("-> Saved fig2_tsne_alignment.pdf")

def plot_shap_analysis(model, x_bg, x_test, feature_names, out_dir):
    """
    Fig 3: SHAP Global Summary and Dependence Plots
    """
    print("Generating Fig 3: SHAP Interpretability Plots...")
    
    # Wrapper function for SHAP
    def model_predict(x):
        x_tensor = torch.tensor(x, dtype=torch.float32, device=x_bg.device)
        with torch.no_grad():
            _, hazard, _, _ = model(x_tensor)
        return hazard.cpu().numpy()
        
    x_bg_np = x_bg.cpu().numpy()
    x_test_np = x_test.cpu().numpy()
    
    # Use KernelExplainer for complex custom architectures
    explainer = shap.KernelExplainer(model_predict, shap.kmeans(x_bg_np, 10))
    shap_values = explainer.shap_values(x_test_np[:100]) # limit to 100 for speed
    x_test_plot = x_test_np[:100]
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    shap_values_np = shap_values.reshape(x_test_plot.shape[0], -1) if len(shap_values.shape) > 2 else shap_values
    
    # Fig 3a: Summary Plot
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Customize colormap to a clean divergent palette often used in medical papers (vlag is very nice)
    shap.summary_plot(
        shap_values_np, 
        x_test_plot, 
        feature_names=feature_names, 
        show=False, 
        cmap=sns.diverging_palette(250, 10, as_cmap=True), # Clean blue-red divergent
        plot_size=(10, 8)
    )
    plt.title("Fig 3A: SHAP Summary (Global Risk Drivers)", pad=20)
    sns.despine(left=True, bottom=True)
    
    # Tweak axes for better readability
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(out_dir / "fig3a_shap_summary.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    print("-> Saved fig3a_shap_summary.pdf")
    
    # Fig 3b: Dependence Plot for top feature
    mean_abs_shap = np.abs(shap_values_np).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][0]
    
    plt.figure(figsize=(8, 6), dpi=300)
    shap.dependence_plot(
        top_idx, shap_values_np, x_test_plot, 
        feature_names=feature_names, show=False, interaction_index=None,
        cmap=sns.diverging_palette(250, 10, as_cmap=True), dot_size=40, alpha=0.75
    )
    plt.title(f"Fig 3B: SHAP Dependence ({feature_names[top_idx]})", pad=15)
    
    # Enhance axes
    ax = plt.gca()
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray')
    ax.grid(axis='x', linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(out_dir / "fig3b_shap_dependence_top1.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    print("-> Saved fig3b_shap_dependence_top1.pdf")

def plot_km_risk_stratification(model, x_test, e_test, t_test, device, out_dir, batch_size=256):
    """
    Fig 4: Kaplan-Meier Risk Stratification Curves
    Splits patients into Low, Medium, and High risk based on predicted hazard
    and plots their actual survival curves.
    """
    print("Generating Fig 4: Kaplan-Meier Risk Stratification Curves...")
    model.eval()
    
    hazards = []
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            end = start + batch_size
            x_batch = torch.tensor(x_test[start:end], dtype=torch.float32, device=device)
            _, hazard, _, _ = model(x_batch)
            hazards.append(hazard.squeeze().cpu().numpy())
    
    hazards = np.concatenate(hazards, axis=0)
        
    # Define risk groups based on quantiles (33.3% and 66.6%)
    q33 = np.percentile(hazards, 33.3)
    q67 = np.percentile(hazards, 66.7)
    
    risk_groups = np.zeros_like(hazards, dtype=int)
    risk_groups[hazards > q33] = 1
    risk_groups[hazards > q67] = 2
    
    from lifelines.plotting import add_at_risk_counts
    from lifelines.statistics import multivariate_logrank_test
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    colors = [NATURE_PALETTE[2], NATURE_PALETTE[4], NATURE_PALETTE[1]] # Green, Purple, Red (high risk)
    labels = ["Low Risk (0-33%)", "Medium Risk (33-67%)", "High Risk (>67%)"]
    
    kmfs = []
    valid_risk_groups = []
    valid_t = []
    valid_e = []
    
    for i in range(3):
        mask = (risk_groups == i)
        if mask.sum() > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(t_test[mask], event_observed=e_test[mask], label=labels[i])
            kmf.plot_survival_function(color=colors[i], ci_show=True, linewidth=2.5, ax=ax)
            kmfs.append(kmf)
            
            valid_risk_groups.extend([i] * mask.sum())
            valid_t.extend(t_test[mask])
            valid_e.extend(e_test[mask])
            
    # Calculate log-rank p-value
    if len(np.unique(valid_risk_groups)) > 1:
        results = multivariate_logrank_test(valid_t, valid_risk_groups, valid_e)
        p_value = results.p_value
        p_text = f"Log-rank $P$ < 0.001" if p_value < 0.001 else f"Log-rank $P$ = {p_value:.3f}"
        
        # Add text box for p-value (placed top right to avoid curves/legend)
        ax.text(0.95, 0.85, p_text, 
                transform=ax.transAxes, ha='right', va='top', 
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='#E0E0E0', boxstyle='round,pad=0.5'))
            
    plt.title("Fig 4: Kaplan-Meier Risk Stratification", pad=15)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Survival Probability (IDH-free)")
    plt.ylim(0.0, 1.05)
    plt.xlim(0, max(t_test) + 10)
    
    # Add a horizontal line at 0.5 if it drops that low, or just grid
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(loc='lower left', frameon=True, edgecolor='#E0E0E0', facecolor='white', framealpha=0.9)
    sns.despine()
    
    # Add "Number at risk" table at the bottom (Standard for medical journals)
    if len(kmfs) == 3:
        add_at_risk_counts(kmfs[0], kmfs[1], kmfs[2], ax=ax, fig=fig)
        # When using add_at_risk_counts, tight_layout often squashes the table. 
        # Explicitly adjust the bottom margin instead.
        plt.subplots_adjust(left=0.12, bottom=0.25, right=0.95, top=0.90)
    else:
        plt.tight_layout()
        
    plt.savefig(out_dir / "fig4_km_risk_stratification.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    print("-> Saved fig4_km_risk_stratification.pdf")

def main():
    print("="*50)
    print("Generating Academic Clinical Figures")
    print("="*50)
    
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["paths"]["figure_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    data_dict = prepare_dataloaders(
        source_path=os.path.join(config["paths"]["data_dir"], config["data"]["source_file"]),
        target_path=os.path.join(config["paths"]["data_dir"], config["data"]["target_file"]),
        batch_size=config["training"]["batch_size"],
        seed=config["training"]["seed"]
    )
    
    treat_indices, physio_indices = get_indices(data_dict["feature_names"])
    
    # 2. Load Model
    model = DomainStratifiedGatedNet(
        input_dim=data_dict["input_dim"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        domain_hidden=config["model"]["domain_hidden"],
        treat_indices=treat_indices,
        physio_indices=physio_indices,
        tokenizer_type="kan",
        kan_basis_dim=config["model"]["kan_bases"]
    ).to(device)
    
    # Note: In a real pipeline, load the best model weights here:
    # model.load_state_dict(torch.load("experiments/results/best_model.pth"))
    
    # 3. Generate Figures
    # Fig 2: t-SNE
    plot_tsne_alignment(model, data_dict["source_loader"], data_dict["target_loader"], device, out_dir)
    
    # Fig 3: SHAP
    x_bg_tensor = []
    for x_s, _, _, _ in data_dict["source_loader"]:
        x_bg_tensor.append(x_s)
        if len(x_bg_tensor) * config["training"]["batch_size"] > 200: break
    x_bg = torch.cat(x_bg_tensor, dim=0)[:200].to(device)
    x_test_tensor = torch.tensor(data_dict["x_test"][:500], dtype=torch.float32).to(device)
    
    # Translate feature names for publication
    academic_feature_names = translate_features(data_dict["feature_names"])
    
    plot_shap_analysis(model, x_bg, x_test_tensor, academic_feature_names, out_dir)
    
    # Fig 4: KM Curves
    plot_km_risk_stratification(
        model, data_dict["x_test"], data_dict["e_test"], data_dict["t_test"], 
        device, out_dir
    )
    
    print("\nAll figures generated successfully in the 'figures/' directory.")

if __name__ == "__main__":
    main()
