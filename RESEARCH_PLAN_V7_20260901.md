# 重构研究方案 v7.0 — 跨域判决机制：什么打坏、什么救回（诊断 + 干预）

**版本**: v7.0 (2026-09-01)
**状态**: 方法论重建锚定草案 — 数据/代码/训练全部以本文为准
**前身**: v6（废掉稻草人主假设）
**决策人拍板**：
1. 核心问题从"谁虐谁"的 superiority **改为机制**：什么影响跨域判决、什么能救它。
2. 主终点：**判别 + 校准 双终点**。
3. 范围：**聚焦版**（诊断=漂移三分，不做亚组；干预=标签 + 对齐）。
4. 预注册主族：**Holm-4，含 alignment（CORAL 为主）**。

---

## 0. 为什么从 v6 升到 v7（v7 的成立理由）

v6 的主假设被 Socratic 审查判定为**稻草人**，必须重构：

| # | v6 缺陷 | 证据 | v7 处置 |
|---|---------|------|---------|
| R1 | **主对照被构造性混染** | H: `supervised_update − target_only`，前者=源预训练+目标微调，后者=30例 target-only。实际在测"211k 源数据有没有用"，**从未分离出"标签更新"这个机制**。无论 H 过不过，都测不到题目里的核心变量 | 主对比改为 `supervised_update vs source_only`（隔离标签增量）与 `supervised_update+CORAL vs supervised_update`（隔离对齐增量） |
| R2 | **退化基线双向判死** | b=10% 时 target-only 深度 MLP 崩到 ~0.546，logistic 达 0.819。"target-only 崩或不崩都可检验"实为失算——对照是坏模型时什么都检验不了 | target_only 仅作诊断基线/学习曲线，**不负主假设**；其低预算崩溃必须先修（logistic 或强正则），否则禁止入诊断主线 |
| R3 | **效应小且无功效** | 标签增量 `supervised_update − source_only ≈ +0.009~0.015`，30 例 CI 大概率跨零。拿 +1% 冲 headline 是追噪声 | 主族明确承诺"效应量+CI+机制方向"的诚实解读；不硬撑 p |
| R4 | **对齐层被降级投闲置散** | CORAL/MMD/DANN 被降为 supplementary，与"什么帮助跨域"这个真实问题脱节 | alignment 回归**主族（CORAL 为主），Holm-4 之一**；MMD/DANN 作 descriptive 敏感性 |

**v7 核心决议（已拍板）**：
1. 主线：**漂移诊断（what affects）+ 干预杠杆（what helps）** 双半结构。
2. 主终点：**判别（meanAUC）+ 校准（ECE@240）**，均在 held-out target_test。
3. 预注册主族：**Holm-4，b=10%，聚焦 IDH**；IH 与 b∈{25,50,100} 全部 descriptive。
4. 数据/模型/损失：**复用 v6 冻结资产与 patient-balanced NLL**，白纸不重造。
5. 允许产出**谦逊阴性/机制结论**；不承诺必中正向 headline。

---

## 1. 研究定位（一页纸）

### 1.1 题目（保留，解释权收敛到机制）

> **Label-Efficient Target-Center Updating for Dynamic Prediction of Intradialytic Hemodynamic Instability: A Two-Center Discrete-Time Survival Study**

### 1.2 核心主张（v7 只承诺这一条）

> 在双中心离散时间生存迁移中，**哪类漂移（协变量 / 标签 / 概念）主导跨中心判别力与校准的空缺，以及目标标签与表征对齐这两个杠杆、分别在哪个终点（判别/校准）上能把它补回来。**

### 1.3 主终点（双，预指定）

| 维度 | 指标 | 定义 | 更好方向 |
|---|---|---|---|
| 判别 Discrimination | `meanAUC` | 4 时点 IPCW-AUC 的算术均值（60/120/180/240min） | 高 |
| 校准 Calibration | `ECE@240` | 在独立 `target_calibration` 上拟合 Platt 共享 hazard 尺度后，于 `target_test` 计算的期望校准误差（风险，t=240min） | 低 |

- 两者均报告 point estimate + `patient-cluster bootstrap ×1000` 的 95% CI。
- IPCW 截尾权重 = Reverse Kaplan-Meier；G(t)<min_g 标记不可估（保留 v6 诊断）。

### 1.4 不再主张的内容（斩断 v6/v5 遗留）

- ❌ ~~"supervised_update 显著优于 b=10% target_only"~~ → 稻草人，废。
- ❌ ~~"CORAL/MMD/DANN 无效"~~ → v6 的否定建立在 Type II / 跨代际上，彻底丢弃；v7 在主族中同代际重估，允许「真无效」或「真有效」两种诚实结果。
- ❌ ~~"选择性临床特征分解假说"~~ → 删除。
- ❌ ~~"全预算矩阵 Holm-16 主族"~~ → 主族 4 条，其余预算 descriptive。

---

## 2. 数据设计（复用 v6 冻结资产，冻结）

### 2.1 数据源与资产

- **直接复用 `data/v6_20260830/` 冻结资产**（sha256 锁、患者级无泄漏、嵌套预算均已验证）。
- 若 v7 需要修数据，须先改本文档再重跑 P1（同 v6 门禁），禁止先改数据后补方案。

### 2.2 特征 / 拆分 / 预算 / 预处理

| 项 | 值 |
|---|---|
| 特征 | 19 项锁定（Physiology & History 18 + Treatment Context 1） |
| 拆分 | 患者级：source 85/15；target 70/10/20（update/calibration/test） |
| 预算 | b ∈ {0.10, 0.25, 0.50, 1.00}，结果盲、患者级嵌套 |
| 目标域 history | 隐藏（部署冷启动视角），仅预算内 target_update/calibration 结局历史可用 |
| 预处理 | 仅 `source_train` 拟合，冻结 `preprocessing.json`，禁止 source_test/target 重拟合 |

### 2.3 强制前置：修 target_only 低预算崩溃

半1 诊断需要**行为良好**的 target_only 学习曲线。若 b=10% target_only(MLP) 仍崩（meanAUC < 0.6 且明显劣于 logistic）：
- 将 target_only 骨干替换为 **logistic / 强 L2 或 early-stop 的线性离散生存**，或
- 对低预算采用强正则化，保证学习曲线单调合理。
- **门禁**：target_only 学习曲线若不单调，禁止进入半1/半2 主族结论。

---

## 3. 半1 DIAGNOSE —「什么影响跨域」（预指定、descriptive、不显著性）

目标：量化"跨域空缺在哪、由哪类漂移主导"。全部为预指定估计量，**不做显著性断言**，输出机制归属，供半2 解释。

### 3.1 迁移空缺量化
- `source_only` 在 **source_test vs target_test** 上的 ΔmeanAUC、ΔECE。
- 建立双终点的空缺基线（判别空缺、校准空缺分开报）。

### 3.2 漂移三分分解
| 漂移类型 | 估计量 | 用途 |
|---|---|---|
| **协变量漂移** | 标准化特征上 源 vs 目标 的 **C2ST（分类器判别 AUC）** 与/或 **锚点 MMD** | 特征分布差异有多大 |
| **标签漂移** | 目标/源结局**事件率比**（IDH≈24% 差在其他中心）；以标签漂移 IPW 重加权校准，度量**校准空缺被闭合多少** | 量级失衡占校准缺空缺多大 |
| **概念/校准漂移** | 标签漂移重加权后的**残留校准空缺** | 风险函数本身是否跨域改变 |

### 3.3 机制归属输出
- 结论：**哪类漂移主导判别空缺、哪类主导校准空缺**。写成 Methods，不进主假设。

> 审讯式纪律：禁止事后挑特征/挑时点解释；分解顺序与估计量在 P0 冻结，与 v7 版本号一并锁死。

---

## 4. 半2 INTERVENE —「什么救它回来」（预注册主族，Holm-4）

### 4.1 主线模型族（复用 v6 patient-balanced NLL 单一实现）

| family | 说明 | 归属 |
|---|---|---|
| `source_only` | 仅 source_train 预训练，target 零样本 | **免费杠杆**（诊断基线） |
| `supervised_update` | source 预训练 + b 预算 target 监督更新 | **标签杠杆** |
| `supervised_update+CORAL` | 上述 + 表征对齐 CORAL | **对齐杠杆**（主族） |
| `supervised_update+{MMD,DANN}` | 对齐敏感性 | descriptive |
| `target_only` | 仅 b 预算 target 训练 | 诊断学习曲线（§2.3 修复后） |

### 4.2 预注册主族（Holm-4，α=0.05，均 b=10%、聚焦 IDH、held-out target_test）

| # | 对比（同代际、同损失） | 终点 | 注册方向 | 机制解读 |
|---|------------------------|------|---------|---------|
| H1 | `supervised_update − source_only` | ΔmeanAUC | 单向>0 | 标签→判别增量 |
| H2 | `supervised_update − source_only` | ΔECE | 注册改善(Δ<0)，报双侧 CI | 标签→校准增量 |
| H3 | `supervised_update+CORAL − supervised_update` | ΔmeanAUC | 单向>0 | 对齐→判别增量 |
| H4 | `supervised_update+CORAL − supervised_update` | ΔECE | 注册改善(Δ<0)，报双侧 CI | 对齐→校准增量 |

- **Holm-Bonferroni 族 = 仅这 4 条**；raw p 用 +1 伪计数（下限 2/1001≈0.002）。
- 结果判定：Δ、双侧 95% CI、raw_p、holm_adj_p、判定。

### 4.3 支持性证据（descriptive，不校正）

- IH：对 H1–H4 做对称镜像描述。
- b ∈ {25, 50, 100}：`supervised_update vs source_only`、`+CORAL` 的 Δ + bootstrap CI。
- MMD/DANN 对齐敏感性。
- 学习曲线：target_only 与 supervised_update 的 meanAUC/ECE vs b。

### 4.4 机制闭环

半2 结果须回到半1：若半1 诊断**概念/校准漂移**主导、半2 显示**对齐主要闭合校准缺口** → 自洽机制故事；若诊断**协变量漂移**主导、半2 仅标签微调改善排序不动量级 → 同样自洽。**机制一致性是审稿主判据**。

---

## 5. 统计推断框架

1. target 患者级 1000 次 patient-cluster bootstrap（固定 INF_SEED，跨 run 共享索引，保证配对）。
2. 每条主假设算 Δ、双侧 95% percentile CI、bootstrap 单向 p（+1 伪计数）。
3. **Holm-4 校正**；调整 p < α=0.05 判显著。
4. **主结论依赖效应量 + CI + 机制方向**，不硬撑 p；即使 Holm 后全不显著，也按预注册如实报告"谦逊阴性/机制"。
5. IPCW G(t) 估计性诊断（frac_bootstrap_not_estimable，G(240)<0.05 在 Limitation 披露）。

---

## 6. 目录与代码（复用 v6 底座，新增半1 诊断脚本）

```
data/v6_20260830/            # 冻结资产（复用）
scripts/v6/
  run_v6_stage_data.py       # P1 数据（未变则复用）
  run_v6_train.py            # 训练 runner（扩展 +CORAL/MMD/DANN family）
  run_v6_infer.py            # bootstrap + Holm-4
  run_v6_diagnose.py         # 【新增】半1 漂移分解（C2ST/MMD/标签漂移IPW/概念残留）
  run_v6_build_tables.py     # 主表 + 半1 机制表 + 支持性表
  run_v6_build_figures.py    # 出版级图
src/v6_acc/                  # 单一损失、模型、加载（复用 + 对齐族扩展）
```

**关键原则**：不 import v4/v5/v6-false-grid 逻辑；单一损失单文件；所有输出带 `v7_20260901` 版本 tag，refuse_overwrite 默认。

---

## 7. 执行顺序（验收门禁）

| 阶段 | 交付 | 门禁 |
|---|---|---|
| P0 方案 | 本文档 | ✓ 本次锚定 |
| P1 数据 | `data/v6_20260830/` 复用确认 | sha256 清单仍有效 |
| P2 代码 | 诊断 + 训练 + 推断 + 扩展对齐族 | 单一损失、bootstrap 绿测 |
| P3 模型 | 修正 target_only 崩溃；训练三杠杆×2终点×4预算×5seed | 学习曲线单调、无 NaN loss |
| P4 半1 诊断 | 漂移分解机制表 | 估计量可复现 |
| P4 半2 推断 | Δ + Holm-4 判定表 | CI/p 范围合法 |
| P5 文档 | 修订论文（机制叙事） | 事实单源可追溯 |

> P1 后任何数据改动必须先改本文档再重跑 P1（与 v6 同纪律）。

---

## 8. 风险与边界（丑话在前）

| 风险 | 说明 | 处置 |
|---|---|---|
| **四条主族全空且 CI 巨宽** | 标签判别增量仅 +0.01；CORAL 曾在 λ 网格表现差。b=10%、30 例主食功可能不足 | 主结论依赖效应量+CI+机制方向；诚实阴性/机制论文是**可接受交付物** |
| **对齐可能真无效** | H3/H4 若空，是预注册的合法发现 | 如实报告；配合半1 诊断解释为何无效 |
| target_only 崩溃 | 半1 诊断基线缺陷 | §2.3 门禁强制修复 |
| 缺乏独立验证中心 | 只有两中心 | Limitation 如实披露 |
| 复现性 | 全链路 RNG 种子 | 沿用 seed 派生链，bootstrap indices 固化 |

---

*End of RESEARCH_PLAN_V7_20260901.md — v7 为唯一重建权威。*