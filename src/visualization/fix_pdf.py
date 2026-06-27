import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def test_save():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    circle = patches.Circle((5, 5), 1, facecolor='red')
    ax.add_patch(circle)
    ax.text(5, 5, 'Test', ha='center', va='center')
    
    # 强制不使用 Type 42 字体，测试是否是字体兼容性导致 PDF 损坏
    plt.rcParams['pdf.fonttype'] = 3
    plt.rcParams['ps.fonttype'] = 3
    
    os.makedirs('figures/journal_ready/Supplementary', exist_ok=True)
    plt.savefig('figures/journal_ready/Supplementary/test_pdf.pdf')
    print("Saved test PDF")

if __name__ == "__main__":
    test_save()
