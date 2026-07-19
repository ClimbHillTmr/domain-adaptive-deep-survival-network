# Hugging Face 数据迁移指南

## 问题诊断

### 当前 LFS 状态

| 指标 | 值 |
|------|-----|
| 本地 LFS 对象目录 | `.git/lfs/objects/` |
| 本地 LFS 对象数量 | 50 个 |
| 本地 LFS 占用空间 | ~2.1 GB |
| 当前 .gitattributes | LFS 已禁用 |
| LFS 服务器端点 | GitHub (dadsn, origin) |

### 历史 LFS 追踪模式（已移除）

根据 git 历史分析，曾经追踪的文件类型包括：

- `data/processed/*.csv`
- `data_preprocessing/data/*.csv`
- `archive/runs_old/*.tar.gz`
- `archive/**/model_checkpoints/*.ckpt`
- `scalers/*.pkl`
- `archive/runs/**/*.pt`
- `archive/runs/**/*.pth`
- `figures/**/*.png`
- `figures/**/*.pdf`
- `experiments/results/*.csv`
- `experiments/results/*.pt`

### 当前大文件分布（磁盘上）

```
data/raw/updated_dataset_shenyi.csv     146 MB
data/processed/深医_final_data.csv       234 MB
data/raw/updated_dataset_fuding.csv      51 MB
data/processed/福鼎_final_data.csv       86 MB
───────────────────────────────────────────────
合计：                                   ~517 MB
```

## Hugging Face 迁移方案

### 仓库结构设计

```
LongGoodbye/
├── Shenyi_Fuding_Dataset        (数据集 - 517 MB)
│   ├── raw/
│   │   ├── updated_dataset_shenyi.csv
│   │   └── updated_dataset_fuding.csv
│   └── processed/
│       ├── 深医_final_data.csv
│       └── 福鼎_final_data.csv
│
├── Shenyi_Fuding_Models         (模型权重 - ~1 MB)
│   └── experiments/
│       ├── final_results/
│       │   └── */mlp_models.pt
│       └── binary_results/
│           └── */mlp_models.pt
│
├── Shenyi_Fuding_Predictions    (预测结果 - ~160 MB)
│   └── experiments/
│       ├── final_results/
│       │   └── */test_predictions.csv
│       └── binary_results/
│           └── */test_predictions.csv
│
└── Shenyi_Fuding_Figures        (论文图表 - ~5 MB)
    └── figures/
```

### 迁移步骤

#### 步骤 1：安装依赖

```bash
pip install huggingface_hub pandas torch
```

#### 步骤 2：配置 Hugging Face Token

```bash
export HUGGINGFACE_TOKEN="your_token_here"
# 或写入 ~/.cache/huggingface/token
```

#### 步骤 3：执行迁移（使用现有脚本）

```bash
# 查看待上传文件（dry run）
python scripts/upload_to_huggingface.py --dry-run --owner LongGoodbye

# 执行上传
python scripts/upload_to_huggingface.py --owner LongGoodbye --bulk

# 使用镜像加速（国内）
python scripts/upload_to_huggingface.py --owner LongGoodbye --bulk --mirror
```

#### 步骤 4：验证上传

```bash
python scripts/validate_huggingface_upload.py
```

### 数据加载方式

#### Python API

```python
from src.data.huggingface_loader import HFDataLoader, load_hf_data

# 创建加载器
loader = HFDataLoader(owner="LongGoodbye", use_mirror=True)

# 列出可用项目
print(loader.list_projects())

# 加载 CSV
df = loader.load_csv("Shenyi_Fuding_Dataset", "processed/深医_final_data.csv")

# 加载模型
model = loader.load_model("Shenyi_Fuding_Models", "experiments/final_results/main_mechanism_aware/mlp_models.pt")

# 便捷函数
data = load_hf_data("Shenyi_Fuding_Dataset", "raw/updated_dataset_shenyi.csv")
```

#### 命令行下载

```bash
# 使用 huggingface-cli
huggingface-cli download LongGoodbye/Shenyi_Fuding_Dataset --local-dir data/

# 使用 git
git clone https://huggingface.co/datasets/LongGoodbye/Shenyi_Fuding_Dataset
```

## 清理 GitHub LFS 配额

### 方案 A：删除本地 LFS 对象（推荐）

**注意：此操作仅清理本地缓存，不会影响远程 LFS 服务器配额。**

```bash
# 删除本地 LFS 对象
git lfs prune

# 验证清理结果
du -sh .git/lfs/objects/
```

### 方案 B：完整清理远程 LFS（需要重写历史）

**警告：此操作会重写 git 历史，需要强制推送，会影响所有协作者。**

```bash
# 安装 git-filter-repo
pip install git-filter-repo

# 创建备份
git clone --mirror https://github.com/ClimbHillTmr/domain-adaptive-deep-survival-network.git backup.git

# 过滤掉所有 LFS 大文件（大于 100MB）
git filter-repo --force --blob-size-limit 100M

# 推送重写后的历史
git push --force origin --all
git push --force origin --tags

# 在 GitHub 仓库设置中手动删除旧的 LFS 对象
# Settings > Large files > Remove
```

### 方案 C：迁移到新仓库

1. 创建新的 GitHub 仓库（不带 LFS）
2. 仅推送代码和小文件
3. 大文件通过 Hugging Face 分发

## 推荐配置

### 更新 .gitignore

确保大文件目录已在 .gitignore 中：

```gitignore
# 数据文件（存储在 Hugging Face）
data/raw/
data/processed/
data_preprocessing/data/

# 模型权重（存储在 Hugging Face）
experiments/**/mlp_models.pt
experiments/**/logistic_models.pkl

# 预测结果（存储在 Hugging Face）
experiments/**/test_predictions.csv
```

### 更新 .gitattributes

```gitattributes
# LFS disabled - large files stored on Hugging Face
# Dataset: https://huggingface.co/datasets/LongGoodbye/Shenyi_Fuding_Dataset
# Models: https://huggingface.co/LongGoodbye/Shenyi_Fuding_Models
# Predictions: https://huggingface.co/datasets/LongGoodbye/Shenyi_Fuding_Predictions
# Figures: https://huggingface.co/datasets/LongGoodbye/Shenyi_Fuding_Figures
```

### 更新 README.md

在 README 中添加数据获取说明：

```markdown
## 数据获取

本项目的大型数据文件存储在 Hugging Face：

- **数据集**: [LongGoodbye/Shenyi_Fuding_Dataset](https://huggingface.co/datasets/LongGoodbye/Shenyi_Fuding_Dataset)
- **模型权重**: [LongGoodbye/Shenyi_Fuding_Models](https://huggingface.co/LongGoodbye/Shenyi_Fuding_Models)
- **预测结果**: [LongGoodbye/Shenyi_Fuding_Predictions](https://huggingface.co/datasets/LongGoodbye/Shenyi_Fuding_Predictions)
- **论文图表**: [LongGoodbye/Shenyi_Fuding_Figures](https://huggingface.co/datasets/LongGoodbye/Shenyi_Fuding_Figures)

### 自动下载

```python
from src.data.huggingface_loader import HFDataLoader

loader = HFDataLoader(owner="LongGoodbye", use_mirror=True)
df = loader.load_csv("Shenyi_Fuding_Dataset", "processed/深医_final_data.csv")
```
```

## 迁移检查清单

- [ ] 安装 huggingface_hub 依赖
- [ ] 配置 Hugging Face Token
- [ ] 执行数据上传脚本
- [ ] 验证上传完整性（MD5 校验）
- [ ] 更新 .gitignore 排除大文件
- [ ] 更新 .gitattributes 禁用 LFS
- [ ] 更新 README.md 添加数据获取说明
- [ ] 运行本地测试确保数据加载正常
- [ ] 清理本地 LFS 对象（git lfs prune）
- [ ] （可选）重写 git 历史清理远程 LFS

## 参考

- Hugging Face Hub 文档: https://huggingface.co/docs/huggingface_hub/
- GitHub LFS 清理指南: https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-git-large-file-storage
