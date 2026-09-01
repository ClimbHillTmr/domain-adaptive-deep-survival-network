import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def set_journal_style(journal_type="nature"):
    """
    设置顶级期刊的可视化标准规范。
    journal_type: 
        - "nature": 适用 Nature Medicine, KI 等，注重简洁、无边框、无衬线字体。
        - "cjasn": 适用 CJASN, BMC Nephrology，可能对颜色有特殊要求。
    """
    # 基础重置
    plt.style.use('default')
    sns.set_theme(style="ticks")
    
    # 字体设置 (Arial/Helvetica 是生物医学期刊的首选)
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['legend.title_fontsize'] = 10
    
    # 线条与坐标轴
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 6
    
    # 去除顶部和右侧的刻度
    mpl.rcParams['xtick.top'] = False
    mpl.rcParams['ytick.right'] = False
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    mpl.rcParams['xtick.major.size'] = 4
    mpl.rcParams['ytick.major.size'] = 4
    mpl.rcParams['xtick.major.width'] = 1.0
    mpl.rcParams['ytick.major.width'] = 1.0
    
    # PDF 输出配置 (使用 Type 42 字体，确保 PDF 中文字可编辑)
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    
    # 保存分辨率
    mpl.rcParams['savefig.dpi'] = 600
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.1

def get_color_palette(n_colors=5, palette_type="nature"):
    """获取符合期刊规范的色板 (色盲友好)"""
    if palette_type == "nature":
        # Nature Publishing Group 推荐的色板 (IBM, colorblind safe)
        colors = ['#648FFF', '#DC267F', '#FFB000', '#FE6100', '#785EF0']
    elif palette_type == "clinical":
        # 适用于临床曲线对比 (蓝、红、绿、紫、橙)
        colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
    else:
        colors = sns.color_palette("colorblind", n_colors).as_hex()
        
    return colors[:n_colors]

def remove_top_right_spines(ax):
    """移除顶部和右侧的边框 (Spines)，这是高级学术图表的标配"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax
