import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as boxstyle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.visualization.journal_style import set_journal_style, get_color_palette

def draw_box(ax, xy, width, height, text, facecolor, edgecolor, text_color='black', fontsize=11, fontweight='normal', alpha=0.9):
    box = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.5,
        alpha=alpha
    )
    ax.add_patch(box)
    
    ax.text(
        xy[0] + width/2, xy[1] + height/2,
        text,
        ha='center', va='center',
        fontsize=fontsize,
        fontweight=fontweight,
        color=text_color,
        linespacing=1.4
    )
    return box

def draw_arrow(ax, start, end, color='black', text=None, text_offset=(0, 0), connectionstyle="arc3,rad=0", linestyle='-'):
    arrow = patches.FancyArrowPatch(
        start, end,
        arrowstyle='->,head_width=6,head_length=8',
        color=color,
        linewidth=2,
        linestyle=linestyle,
        connectionstyle=connectionstyle
    )
    ax.add_patch(arrow)
    
    if text:
        ax.text(
            (start[0] + end[0])/2 + text_offset[0],
            (start[1] + end[1])/2 + text_offset[1],
            text,
            ha='center', va='center',
            fontsize=10,
            fontstyle='italic',
            color=color,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5)
        )

def generate_architecture_diagram():
    """生成符合期刊规范的概念框架图，消除强因果暗示，明确为联合分布对齐"""
    set_journal_style("nature")
    colors = get_color_palette(5, "nature")
    
    c_data = '#E3F2FD'
    c_encoder = '#E8F5E9'
    c_mask = '#FFF3E0'
    c_surv = '#FCE4EC'
    c_domain = '#F3E5F5'
    c_edge = '#424242'
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 1. Data Input Module
    draw_box(ax, (0.5, 4.5), 2.5, 1.2, "Source Domain\n(Center A)\n$X_S, Y_S$", c_data, c_edge, fontweight='bold')
    draw_box(ax, (0.5, 2.3), 2.5, 1.2, "Target Domain\n(Center B)\n$X_T$ (Unlabeled)", c_data, c_edge, fontweight='bold')
    ax.text(1.75, 6.2, "Input Data", ha='center', fontsize=12, fontweight='bold')
    
    # 2. Feature Extractor Module
    draw_box(ax, (4.5, 3.4), 2.5, 1.2, "KAN Tokenizer\n&\nFeature Extractor", c_encoder, c_edge, fontweight='bold')
    draw_arrow(ax, (3.0, 5.1), (4.5, 4.0), connectionstyle="arc3,rad=-0.2")
    draw_arrow(ax, (3.0, 2.9), (4.5, 4.0), connectionstyle="arc3,rad=0.2")
    
    draw_box(ax, (4.5, 5.2), 2.5, 1.0, "Gated Sparsity Mask", c_mask, c_edge)
    draw_arrow(ax, (5.75, 4.6), (5.75, 5.2), color=colors[2], linestyle='--')
    ax.text(5.75, 6.5, "Representation Learning", ha='center', fontsize=12, fontweight='bold')
    
    # 3. Domain Adaptation Module (Modified to emphasize joint distribution P(X,Y))
    draw_box(ax, (7.5, 1.5), 3.0, 1.2, "Conditional Domain\nAdversarial Network\n(CDAN)", c_domain, c_edge, fontweight='bold')
    draw_arrow(ax, (7.0, 4.0), (7.5, 2.1), color=colors[3], text="Latent\n$Z$", text_offset=(-0.5, -0.5), connectionstyle="arc3,rad=-0.2")
    
    ax.text(9.0, 0.8, "Joint Distribution Alignment $P(Z, \hat{Y})$", ha='center', fontsize=11, fontweight='bold', color=colors[3])
    
    # 4. Survival Prediction Module
    draw_box(ax, (9.0, 4.5), 3.0, 1.2, "Domain-Stratified\nCox Model", c_surv, c_edge, fontweight='bold')
    draw_box(ax, (9.0, 6.0), 3.0, 0.8, "IPCW\n(Inverse Probability of Censoring)", c_surv, c_edge)
    
    draw_arrow(ax, (7.0, 4.0), (9.0, 5.1))
    draw_arrow(ax, (10.5, 6.0), (10.5, 5.7), color=colors[1], linestyle='--')
    
    # Output
    draw_arrow(ax, (12.0, 5.1), (13.2, 5.1))
    ax.text(13.5, 5.1, "Predicted\nRisk Score", ha='left', va='center', fontsize=12, fontweight='bold', color='#D32F2F')
    
    # Modified Condition Arrow (No "causal" implication)
    draw_arrow(ax, (10.5, 4.5), (9.0, 2.7), color=colors[0], text="Conditioning via\nCross-Covariance", text_offset=(1.0, 0.2), connectionstyle="arc3,rad=-0.2", linestyle=':')
    
    # Grouping boxes
    rect1 = patches.Rectangle((4.0, 3.0), 3.5, 3.5, linewidth=1.5, edgecolor='#9E9E9E', facecolor='none', linestyle='--', alpha=0.5)
    ax.add_patch(rect1)
    
    rect2 = patches.Rectangle((8.5, 4.2), 4.0, 2.8, linewidth=1.5, edgecolor='#9E9E9E', facecolor='none', linestyle='--', alpha=0.5)
    ax.add_patch(rect2)
    
    # Add Figure Legend / Note at bottom
    ax.text(7.0, -0.5, "* Note: CDAN aligns the joint distribution $P(Z, Y)$ statistically, without implying causal DAG interventions.", 
            ha='center', fontsize=10, fontstyle='italic', color='#616161')
            
    plt.tight_layout()
    os.makedirs('figures/Supplementary_Figures', exist_ok=True)
    plt.savefig('figures/Supplementary_Figures/FigS1_Architecture.pdf', dpi=600, bbox_inches=None)
    plt.savefig('figures/Supplementary_Figures/FigS1_Architecture.png', dpi=600, bbox_inches=None)
    plt.close()

if __name__ == "__main__":
    generate_architecture_diagram()
