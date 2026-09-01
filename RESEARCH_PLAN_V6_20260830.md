# 重构研究方案 v6.0 — Label-Efficient Target-Center Updating（重建底盘）

**版本**: v6.0 (2026-08-30)
**状态**: 重建锚定草案 — 数据/代码/训练全部以本文为准
**前身**: v4 / v5 / amendment（判定为方法论废料，不继承）
**决策人结论**: 抛弃修修补补，白纸重建。

---

## 0. 为什么必须重建（v6 的成立理由）

v4–v5 的结论层已被 Socratic 审查判定为不可信，具体证据链：

| # | 缺陷 | 证据 | 处置 |
|---|------|------|------|
| R1 | **基线代际混杂** | `analyze_amendment_v2.py` 的 Δ 用 8-01 旧损失 run（epoch_log 5列）作为 vs 新损失 λ runs（17列）的对照，Δ 无法归因 | v6 全部模型统一用**同一代际**的 patient-balanced NLL 重训，基线/处理在同代际 |
| R2 | **主族错位** | v4 把 selective_* 机制注册为主族（Holm-16 全败），真正 label-efficiency 主线被降级 | v6 主族只注册<b =10%> 聚焦对比，Holm-2 |
| R3 | **零功效检验** | CORAL λ=0.01 梯度低于有效带 34×/15×，"CORAL 无效"建立在 Type II error 上 | v6 对齐层降为 supplementary，仅同代际重训后报告描述性结果 |
| R4 | **Table5 语义倒置** | falsification_supported 读作证伪，实际是支持；三态标签混乱 | v6 废弃该表，主结论不依赖选择性验证 |
| R5 | **代码版本分裂** | delivery_package 旧于根目录 | v6 全新契约，单源真值 |
| R6 | **损失函数双实现** | main_time_models(pos_weight) vs production(patient-balanced) 冲突 | v6 白纸重写为单一 patient-balanced NLL |

**v6 核心决议（已拍板）**：
1. 主线：**标签效率 + 域适应**（延续题目）。
2. 主假设：**只聚焦 b=10%**，supervised_update vs target_only，Holm-2。
3. 损失：**统一 patient-balanced NLL**，全部模型同代际。
4. 对齐层（CORAL/MMD/DANN）：**supplementary**，不进主族。
5. 目标域 history：**隐藏**（v5 现状，部署冷启动视角）。
6. 重建方式：**新 v6 目录平行**，v4/v5 存档，原地不删。

---

## 1. 研究定位（一页纸）

### 1.1 题目（保留，但解释权收敛）

> **Label-Efficient Target-Center Updating for Dynamic Prediction of Intradialytic Hemodynamic Instability: A Two-Center Discrete-Time Survival Study**

### 1.2 核心主张（v6 只承诺这一条）

> 在目标中心标签极度稀缺（预算 10%，约 30-35 例患者）时，**源中心预训练 + 目标中心极少量监督更新**（supervised update）的离散时间生存模型，其 4 时点 IPCW-AUC 显著优于**仅用目标数据训练的 target-only 模型**。

**单条主假设**（Holm-2，α=0.05）：

```
H10: 预算 b=10% 时，IDH 的 supervised_update − target_only ΔIPCW-AUC(240min 均值) > 0
H20: 预算 b=10% 时，IH  的 supervised_update − target_only ΔIPCW-AUC(240min 均值) > 0
```

- 方向：单侧（supervised_update 应更优），但报告双侧 CI。
- 主终点：**4 时点 IPCW-AUC 的算术均值**（与 v4 定义一致）。
- 若 Holm 校正后 H10 通过但 H20 不通过：如实报告，不取消 H10。

### 1.3 明确不再主张的内容（斩断 v4/v5 遗留）

- ❌ ~~"CORAL/MMD/DANN 提供/不提供增量价值"~~ → 一律不承诺，仅作 supplementary 描述报告（同代际重训，附 bootstrap CI，不做显著性断言）。
- ❌ ~~"选择性临床特征分解假说被证实/证伪"~~ → 从主叙事中删除。
- ❌ ~~"全预算矩阵 Holm-16 主族"~~ → 主族只 2 条，其余预算（25/50/100）作为**支持性证据（descriptive + CI）**，不参与主族校正。
- ❌ 任何建立在跨代际损失 / 非患者级 bootstrap / 未校正弱检验上的显著性措辞。

### 1.4 支持性证据（descriptive，不校正）

用于支撑主结论的稳健性，均报告 point estimate + patient-cluster bootstrap 95% CI：
- 其他预算 b ∈ {25%, 50%, 100%} 的同对比。
- source_only（零样本迁移）vs target_only。
- supervised_update vs source_only（监督更新相对零样本迁移的增量）。
- 校正（calibration ECE，Platt 后）作为次要指标。

---

## 2. 数据设计（v6_data，复用 v5 数据层）

### 2.1 数据源（不可再生，哈希锁死）

| 项 | 值 |
|---|---|
| source raw | `data/raw/updated_dataset_shenyi.csv`（152MB, sha256 锁） |
| target raw | `data/raw/updated_dataset_fuding.csv`（52MB, sha256 锁） |
| 构建函数 | `build_v4_restart_survival_cohort`（data_process.py，逻辑已验证） |
| 数据资产层 | 复用 `v5_data_stage.py` / `v5_budget_subsets.py`（患者拆分/隐藏目标历史/19分区/嵌套预算/SHA256冻结） |

> **关键**：数据研制阶段复用 v5 的**代码**，但输出到独立根 `data/v6_20260830/`，且**更新协议版本为 contract_v6**，禁止调用旧 contract 校验。这样拿到的是全新冻结资产，不用动 v5 原始文件。

### 2.2 特征（19 项锁定，不变）

Physiology & History (18) + Treatment Context (1: 透前体重-干体重)。log1p: history_prior_session_count。与 v5 完全一致——这些特征定义本身是健康资产。

### 2.3 拆分与预算

- 患者级分层拆分：source 85/15，target update/calibration/test 70/10/20。
- 预算：b ∈ {0.10, 0.25, 0.50, 1.00}，**结果盲 (outcome-blind)**，患者级嵌套，分母 = update+calibration 患者并集。
- **目标域 history 特征隐藏**（v5 现状）：target_cohort 的 history_*/历史平均 置 NaN。仅预算内 target_update/calibration 患者的结局历史可用于训练（若有）。

### 2.4 预处理

仅在 `source_train` 拟合（中位数插补 + mean/population-sd 标准化），冻结为 `preprocessing.json`，应用到 target 全部子集。禁止在 source_test 或 target 上重拟合。

### 2.5 v6 数据研制步骤（scripts/v6/run_v6_stage_data.py）

调用 `build_scientific_assets`（参数化 data_root + protocol tag），输出到 `data/v6_20260830/`：
`source_cohort` / `target_cohort` / `split_manifest` / `preprocessing` / `primary_partition` / `random_partitions` / `budget_subsets` + 全部 SHA256 锁。

---

## 3. 模型与损失（v6 白纸重写）

### 3.1 主线模型族（v6_focal 家族，只做这些）

| family | 说明 |
|---|---|
| `target_only` | 仅用 b 预算选中的 target_update 训练（基线，最弱对照） |
| `source_only` | 仅用 source_train 预训练，target 上零样本（零样本迁移基线） |
| `supervised_update` | source 预训练 + b 预算 target 监督更新（**主处理**） |

> v6 主线**不带任何对齐机制**（CORAL/MMD/DANN 全放 supplementary）。这样主结论干净：只有「源预训练 + 目标微调」这一件事。

### 3.2 单一损失：Patient-Balanced Discrete Survival NLL（唯一实现）

所有终点、所有 family 统一用**同一个**损失函数，杜绝代际混杂：

```
对 session i、interval k，观测 mask o_ik ∈ {0,1}（该区间在随访窗口内且未提前事件），
hazard logit z_ik，hazard 概率 p_ik = σ(z_ik)。
单样本 NLL：L_i = -Σ_k o_ik * [ y_ik·log p_ik + (1-y_ik)·log(1-p_ik) ]
患者均衡权重：w_i = 1 / (该患者 session 数)   ← 每个患者对总损失贡献相等
总损失：L = -Σ_i w_i * L_i / Σ_i w_i
```

- **无 pos_weight，无 class_ratio**（R6 根除）。
- 网络输出 4 个 hazard logit（对应 4 区间），累积风险 = 1−Π(1−p_ik)。
- 训练分两阶段（supervised_update）：
  - stage1 `source_pretrain`: lr 1e-3, max 40 ep, patience 5，仅 source_train。
  - stage2 `target_update`: lr 1e-4, max 20 ep, patience 5，仅预算目标 update。
  - target_only 只跑 update 段；source_only 只跑 pretrain 段。

### 3.3 架构（同一骨干，三 family 共享权重结构）

- 骨干：MLP 19 → 64 → 32 → 4（hazard logits），dropout 0.2。
- 共享参数骨架，不设 dual-branch（主线无对齐）；dual 仅 supplementary 里有。
- batch_size 256，Adam，weight_decay 1e-4，grad clip 1.0。

### 3.4 评估（复用 `src/evaluate/discrete_survival_metrics.py`，已验证健康）

- IPCW-AUC / IPCW-Brier @ 4 时点 + 均值，Reverse-KM 截尾权重，min_g 1e-8。
- patient-cluster bootstrap（`patient_cluster_bootstrap_indices`）生成 1000 复制。
- Platt 校正：在独立 target_calibration 上拟合并共享 hazard 尺度。

---

## 4. 统计推断框架（v6 干净版）

### 4.1 主推断（患者级 bootstrap + Holm-2）

1. 在 target 患者级构造 1000 次 patient-cluster bootstrap。
2. 对 2 条主假设（H10 IDH / H20 IH, b=10%）计算 ΔAUC 的点估计、bootstrap 双侧 95% CI、以及 bootstrap 单侧 p 值（+1 伪计数，最小 p=2/1001≈0.002）。
3. **Holm-Bonferroni 校正族 = 仅这 2 条**。调整后 p 在 α=0.05 判显著。
4. 报告：Δ、CI、raw_p、holm_adj_p、判定（通过/不通过）。

### 4.2 支持性证据（descriptive，不校正）

b ∈ {25,50,100} 与 source_only 等对比：只报 Δ + bootstrap 95% CI，明确标注「descriptive，未做多重比较校正」。

### 4.3 效应量与样本量声明

- 主对比核心矛盾在 CI 而非 p；即使 p 达 bootstrap 下限，也结合 ΔAUC 与 CI 解读。
- 报告约 30 例 (b=10%) 患者下的 CI 宽度，说明功效边界。

### 4.4 IPCW G(t) 估计性诊断

每个 bootstrap 复制若 G(t) < min_g 则标记不可估；汇总 `frac_bootstrap_not_estimable`。若 240min 点估计 G < 0.05 需在 Limitation 披露（保留 v4 已实现的诊断，纳入 v6 主输出）。

---

## 5. Supplementary（不在主叙事结论里）

### 5.1 域对齐层重训（supplementary，同代际）

- 在 `supervised_update` 骨架上加 CORAL / MMD / DANN，**全部用 §3.2 同一损失**，统一 runner。
- CORAL λ 不做重网格；用预注册 λ 集合跑同代际；只报 5-seed 均值 + bootstrap CI，**不做显著性断言**。
- 目的：回答「在干净同代际损失下，对齐是否在描述性层面改变 Δ」——作为讨论素材，不作结论。

### 5.2 随机分区（可选，作稳健性）

19 随机分区仅作描述性稳健性，不放结论主表。

---

## 6. 目录与代码结构（v6 平行，单源真值）

```
conf/v6_20260830.yaml                  # 唯一契约，github 式单源
data/v6_20260830/                      # 冻结数据资产 + SHA256 锁
experiments/v6_20260830/               # 训练 run + 结果 + evidence
scripts/v6/
  run_v6_stage_data.py                 # 数据研制（复用 v5_data_stage）
  run_v6_train.py                      # 白纸训练 runner（主线三 family）
  run_v6_supp_align.py                 # supplementary 对齐层同代际训练
  run_v6_infer.py                      # patient-cluster bootstrap 推断 + Holm-2
  build_v6_tables.py                   # 主表/支持性表
  build_v6_figures.py                  # 出版级图
src/  （新模块，v6 专用，不改旧 src 语义）
```

**关键原则**：
- 所有输出带协议版本 `v6_20260830`，refuse_overwrite 默认，SHA256 全程记录。
- 不 import 任何 v4/v5 runner 的 λ_grid / amendment 逻辑（R1/R3 根除）。
- 单一损失函数源码文件，被所有 family 引用（R6 根除）。

---

## 7. 执行顺序（验收门禁）

| 阶段 | 交付 | 门禁 |
|---|---|---|
| P0 方案 | 本文档 | ✓ 已锚定拍板 |
| P1 数据 | `data/v6_20260830/` 冻结 | sha256 清单 + 患者级无泄漏校验 |
| P2 代码 | 训练 + 推断脚本 | 单损失单一实现、bootstrap 绿测 |
| P3 主线训练 | 三 family × 5 seed × 2 endpoint × 4 budget | 无 NaN loss、跨 seed 稳定 |
| P4 推断 | ΔAUC + Holm-2 判定表 | CI 生成、p 范围合法 |
| P5 文档 | 修订论文（基于 v6 干净结果） | 事实单源可追溯 |

> P1 后**任何对数据的改动必须先改此方案再重跑 P1**，绝不允许先改数据后补方案（v4/v5 的坑）。

---

## 8. 风险与边界

| 风险 | 说明 | 缓解 |
|---|---|---|
| b=10% 功效不足 | 约 30 例患者下 ΔAUC CI 可能跨 0 | 如实报告；主结论依赖效应量+CI，不硬撑 p |
| target-only 崩或不崩 | v4 曾见 target-only deep 崩到 0.546 | 无妨：无论 target-only 好坏，主假设都可检验 |
| 目标历史隐藏导致 target-only 过弱 | 冷启动视角本就如此 | 这是设定而非缺陷，写入 Methods |
| 复现性 | 需 RNG 种子全链路固定 | 沿用 seed 派生链，bootstrap indices 固化 |

---

*End of RESEARCH_PLAN_V6_20260830.md — v6 为唯一重建权威。*