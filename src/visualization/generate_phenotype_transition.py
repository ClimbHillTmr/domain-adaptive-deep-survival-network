import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.visualization.journal_style import set_journal_style, get_color_palette

def generate_phenotype_transition():
    """生成表型状态转移图，作为定性概念图，移除所有未经统计检验的硬编码概率"""
    set_journal_style("nature")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.4, 7.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 节点定义
    nodes = {
        'Stable': (3, 6, 0.8, "Stable\nHemodynamics\n(Low Historical Rate)", '#A5D6A7'),
        'Vulnerable': (7, 6, 1.0, "Vulnerable State\n(High UF Rate,\nHigh BP Var)", '#FFF59D'),
        'Acute': (7, 2, 1.2, "Acute IDH\n(Critical Deterioration)", '#EF9A9A'),
        'Chronic': (3, 2, 0.9, "Chronic Hypotension\n(Frequent IDH History)", '#CE93D8')
    }
    
    # 绘制节点
    node_patches = {}
    for name, (x, y, r, label, c) in nodes.items():
        circle = patches.Circle((x, y), r, facecolor=c, edgecolor='#424242', linewidth=2, alpha=0.9, zorder=3)
        ax.add_patch(circle)
        node_patches[name] = circle
        ax.text(x, y, label, ha='center', va='center', fontweight='bold', fontsize=11, zorder=4)
        
    # 边定义：移除硬编码概率，改为定性的表述
    edges = [
        ('Stable', 'Vulnerable', 4, "High UFR / Large Weight Gain", 0.2),
        ('Vulnerable', 'Stable', 2, "Adequate UF Profiling", 0.2),
        ('Vulnerable', 'Acute', 6, "Exceeds Plasma Refill Rate", 0.1),
        ('Acute', 'Chronic', 5, "Repeated Episodes\n(Maladaptive Response)", 0.2),
        ('Chronic', 'Vulnerable', 3, "Inter-dialytic Weight Gain", 0.2),
        ('Stable', 'Acute', 1, "Rare Transition", -0.1)
    ]
    
    # 绘制连线
    for start, end, weight, label, rad in edges:
        x1, y1 = nodes[start][0], nodes[start][1]
        x2, y2 = nodes[end][0], nodes[end][1]
        r1, r2 = nodes[start][2], nodes[end][2]
        
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle='->,head_width=8,head_length=12',
            color='#616161',
            linewidth=weight*0.8,
            shrinkA=r1*40, shrinkB=r2*40,
            zorder=2
        )
        ax.add_patch(arrow)
        
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            
            if rad > 0:
                mx += dy * 0.1
                my -= dx * 0.1
            else:
                mx -= dy * 0.1
                my += dx * 0.1
                
            ax.text(
                mx, my, label,
                ha='center', va='center',
                fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5),
                zorder=5
            )
            
    ax.set_title("Hypothesized Longitudinal Phenotype Transitions", fontsize=16, fontweight='bold', pad=20)
    
    # 添加保护性声明
    ax.text(
        0.5,
        -0.06,
        "* Schematic representation of hypothesized transitions. Not derived from a formal multi-state model.",
        transform=ax.transAxes,
        ha='center',
        fontsize=10,
        fontstyle='italic',
        color='#616161'
    )
    
    os.makedirs('figures/Supplementary_Figures', exist_ok=True)
    plt.subplots_adjust(left=0.04, right=0.96, top=0.9, bottom=0.14)
    plt.savefig('figures/Supplementary_Figures/FigS2_Phenotype_Transition.pdf', dpi=600, bbox_inches=None)
    plt.savefig('figures/Supplementary_Figures/FigS2_Phenotype_Transition.png', dpi=600, bbox_inches=None)
    plt.close()

if __name__ == "__main__":
    generate_phenotype_transition()
