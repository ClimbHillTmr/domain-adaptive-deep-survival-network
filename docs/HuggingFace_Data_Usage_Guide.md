# Hugging Face 数据存储与调用指南

## 概述

本项目将数据文件按项目分类存储到 Hugging Face，支持跨项目的数据调用和下载需求。所有数据文件保持独立的项目结构，并保留完整的元数据信息。

## 项目分类结构

| 项目名称 | Hugging Face 仓库 ID | 类型 | 内容说明 |
|---------|---------------------|------|---------|
| `data-shenyi-fuding` | `hbd-survival/data-shenyi-fuding` | dataset | 深医和福鼎的原始数据和处理后数据 |
| `archive-backup` | `hbd-survival/archive-backup` | dataset | 项目备份文件（配置、脚本、历史数据） |
| `model-checkpoints` | `hbd-survival/model-checkpoints` | model | 训练好的模型权重和实验结果 |
| `figures-publication` | `hbd-survival/figures-publication` | dataset | 论文发表用的图表文件 |
| `docs-reports` | `hbd-survival/docs-reports` | dataset | 文档和报告文件 |

### 数据项目目录结构

```
data-shenyi-fuding/
└── processed/              # 处理后数据
    ├── 深医_final_data.csv    # 深医最终分析数据集 (216,604 sessions)
    └── 福鼎_final_data.csv    # 福鼎最终分析数据集 (75,224 sessions)
```

> **Note**: Raw data and intermediate processing files have been moved to archive for storage optimization.
> To re-run the data pipeline, restore raw data from `archive/data_backup_20260627/raw/`.

## 上传数据

### 准备工作

1. 获取 Hugging Face API Token（需要写入权限）
2. 安装依赖：`pip install huggingface_hub`
3. 推荐配置 Git LFS 跟踪大文件（已在 `.gitattributes` 中配置）

### 使用上传脚本

```bash
# 查看帮助
python scripts/upload_to_huggingface.py --help

# 模拟上传（查看将要上传的文件）
python scripts/upload_to_huggingface.py --dry-run

# 执行上传（Token 从环境变量或 ~/.cache/huggingface/token 读取）
python scripts/upload_to_huggingface.py

# 使用代理上传（如果网络受限）
python scripts/upload_to_huggingface.py --proxy http://proxy:port

# 使用国内镜像上传（推荐中国用户）
python scripts/upload_to_huggingface.py --mirror

# 使用批量上传模式（更快，适合大量文件）
python scripts/upload_to_huggingface.py --mirror --bulk

# 显式指定 Token
python scripts/upload_to_huggingface.py --token YOUR_TOKEN --mirror
```

### 上传参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--token` | Hugging Face API Token（可选，优先从环境变量读取） | 无 |
| `--owner` | Hugging Face 组织/用户名 | `hbd-survival` |
| `--dry-run` | 模拟上传，不实际执行 | False |
| `--proxy` | HTTP/HTTPS 代理地址 | 无 |
| `--mirror` | 使用 hf-mirror.com 国内镜像 | False |
| `--bulk` | 使用批量上传模式（`upload_folder`） | False |
| `--retry` | 失败重试次数 | 3 |

### Token 安全

Token 可以通过以下方式提供（优先级从高到低）：
1. 命令行参数 `--token`
2. 环境变量 `HUGGINGFACE_TOKEN`
3. 文件 `~/.cache/huggingface/token`

**不推荐**在命令行直接传递 Token，因为会在进程列表中暴露。

### 上传报告

上传完成后会生成 `huggingface_upload_report.json`，包含：
- 每个文件的上传状态
- 文件元数据（大小、MD5、时间戳）
- 上传统计摘要

## 下载与调用数据

### 方式一：使用 HFDataLoader 类

```python
from src.data import HFDataLoader

# 基础用法
loader = HFDataLoader(
    owner="hbd-survival",
    cache_dir="~/.cache/hbd-survival",
)

# 使用国内镜像
loader = HFDataLoader(
    owner="hbd-survival",
    use_mirror=True,
)

# 使用代理
loader = HFDataLoader(
    owner="hbd-survival",
    proxy="http://proxy:port",
)
```

#### 列出所有项目

```python
projects = loader.list_projects()
print(projects)
# 输出: ['data-shenyi-fuding', 'archive-backup', 'model-checkpoints', 'figures-publication', 'docs-reports']
```

#### 列出项目中的文件

```python
files = loader.list_files("data-shenyi-fuding")
print(files)
```

#### 下载单个文件

```python
# 下载到缓存并返回路径
file_path = loader.download_file(
    project_name="data-shenyi-fuding",
    file_path="processed/深医_final_data.csv"
)

# 下载到指定目录
file_path = loader.download_file(
    project_name="data-shenyi-fuding",
    file_path="processed/深医_final_data.csv",
    local_dir="./data"
)

# 下载并返回元数据
metadata = loader.download_file(
    project_name="data-shenyi-fuding",
    file_path="processed/深医_final_data.csv",
    return_metadata=True
)
print(metadata)
```

#### 下载整个项目

```python
downloaded_files = loader.download_project(
    project_name="data-shenyi-fuding",
    local_dir="./downloaded_data"
)
```

#### 直接加载数据

```python
# 加载 CSV 文件
df = loader.load_csv(
    project_name="data-shenyi-fuding",
    file_path="processed/深医_final_data.csv",
    encoding="utf-8"
)

# 加载 JSON 文件
results = loader.load_json(
    project_name="model-checkpoints",
    file_path="v5_validation/v5_validation_results.json"
)

# 加载模型权重
model = loader.load_model(
    project_name="model-checkpoints",
    file_path="v5_validation/base/dadsn_final.pt",
    device="cpu"
)
```

### 方式二：使用便捷函数 load_hf_data

```python
from src.data import load_hf_data

# 自动识别文件类型并加载
df = load_hf_data(
    project_name="data-shenyi-fuding",
    file_path="processed/深医_final_data.csv"
)

# 显式指定文件类型
results = load_hf_data(
    project_name="model-checkpoints",
    file_path="v5_validation/v5_validation_results.json",
    file_type="json"
)
```

### 方式三：跨项目调用

```python
from src.data import HFDataLoader

loader = HFDataLoader(use_mirror=True)

# 从数据项目加载训练数据
train_data = loader.load_csv(
    project_name="data-shenyi-fuding",
    file_path="processed/深医_final_data.csv"
)

# 从模型项目加载预训练模型
model = loader.load_model(
    project_name="model-checkpoints",
    file_path="v5_optimized/base_transformer/dadsn_final.pt"
)

# 从图表项目获取可视化参考
fig_files = loader.list_files("figures-publication")
print(fig_files)
```

## 数据完整性验证

### 验证上传结果

```bash
# 使用服务器端元数据验证（快速）
python scripts/validate_huggingface_upload.py --mirror

# 下载文件进行 MD5 校验（更精确）
python scripts/validate_huggingface_upload.py --mirror --no-server-metadata
```

验证报告 `huggingface_validation_report.json` 包含：
- 每个文件的校验结果
- 文件大小比对（服务器端元数据模式）
- MD5 校验（下载比对模式）
- 缺失/多余文件检测

### 使用 Loader 验证

```python
from src.data import HFDataLoader

loader = HFDataLoader(use_mirror=True)

# 获取项目信息
info = loader.get_project_info("data-shenyi-fuding")
print(info)

# 验证文件完整性
validation_results = loader.validate_integrity("data-shenyi-fuding")

# 检查特定文件的 MD5
expected_md5s = {
    "processed/深医_final_data.csv": "abc123..."
}
validation_results = loader.validate_integrity(
    "data-shenyi-fuding",
    expected_md5s=expected_md5s
)
```

## 在其他项目中使用

### 安装依赖

```bash
pip install huggingface_hub pandas torch
```

### 复制 loader 文件

将 `src/data/huggingface_loader.py` 复制到其他项目中：

```python
from huggingface_loader import HFDataLoader, load_hf_data

loader = HFDataLoader(use_mirror=True)

# 使用方式相同
df = loader.load_csv(
    project_name="data-shenyi-fuding",
    file_path="processed/深医_final_data.csv"
)
```

### 通过 Hugging Face Hub 直接访问

```python
from huggingface_hub import hf_hub_download
import os

# 设置国内镜像（中国用户）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 直接下载文件
file_path = hf_hub_download(
    repo_id="hbd-survival/data-shenyi-fuding",
    filename="processed/深医_final_data.csv",
    repo_type="dataset"
)

# 加载数据
import pandas as pd
df = pd.read_csv(file_path)
```

## 缓存机制

### 默认缓存目录

```
~/.cache/hbd-survival/
```

### 缓存策略

- 文件首次下载后会缓存在本地
- 再次请求相同文件时优先使用缓存
- 可通过 `force_download=True` 强制重新下载

### 自定义缓存目录

```python
loader = HFDataLoader(
    cache_dir="/path/to/custom/cache"
)
```

## 文件类型支持

| 文件类型 | 扩展名 | 加载方法 |
|---------|--------|---------|
| CSV | `.csv` | `load_csv()` |
| JSON | `.json` | `load_json()` |
| PyTorch 模型 | `.pt`, `.pth` | `load_model()` |
| Pickle | `.pkl` | `load_pickle()` |
| 其他 | 任意 | `download_file()` 返回路径 |

## Git LFS 配置

已在 `.gitattributes` 中配置大文件跟踪：

```
data/raw/*.csv filter=lfs diff=lfs merge=lfs -text
data/processed/*.csv filter=lfs diff=lfs merge=lfs -text
archive/runs/**/*.pt filter=lfs diff=lfs merge=lfs -text
archive/runs/**/*.pth filter=lfs diff=lfs merge=lfs -text
figures/**/*.png filter=lfs diff=lfs merge=lfs -text
figures/**/*.pdf filter=lfs diff=lfs merge=lfs -text
```

如需初始化 LFS：

```bash
git lfs install
git lfs track "data/**/*.csv"
git add .gitattributes
```

## 注意事项

1. **网络访问**：中国用户推荐使用 `--mirror` 参数
2. **Token 权限**：上传需要写入权限的 Token，下载私有仓库也需要 Token
3. **文件大小**：大文件下载可能需要较长时间，请确保网络稳定
4. **缓存管理**：定期清理缓存目录以释放磁盘空间
5. **数据更新**：数据更新后需要重新上传，建议使用版本号或时间戳区分
6. **批量上传**：大量文件推荐使用 `--bulk` 参数提高效率

## 故障排除

### 网络连接错误

```bash
# 使用国内镜像
python scripts/upload_to_huggingface.py --mirror

# 使用代理
python scripts/upload_to_huggingface.py --proxy http://proxy:port

# 检查网络连接
curl -I https://hf-mirror.com
```

### Token 权限不足

```bash
# 登录验证
python -c "from huggingface_hub import login; login(token='YOUR_TOKEN')"
```

### 文件下载失败

```python
# 强制重新下载
loader.download_file(
    project_name="data-shenyi-fuding",
    file_path="processed/深医_final_data.csv",
    force_download=True
)
```

### 完整性验证失败

检查验证报告中的详细信息，重新上传失败的文件。

## 项目文件清单

| 文件路径 | 说明 |
|---------|------|
| `scripts/upload_to_huggingface.py` | 上传脚本 |
| `scripts/validate_huggingface_upload.py` | 验证脚本 |
| `src/data/huggingface_loader.py` | 数据加载器 |
| `src/data/__init__.py` | 模块导出 |
| `src/__init__.py` | 包导出 |
| `.gitattributes` | LFS 配置 |
| `docs/HuggingFace_Data_Usage_Guide.md` | 使用文档 |