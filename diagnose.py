"""
快速诊断脚本 — 逐步检查训练管线的各个环节。
运行: python diagnose.py
"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

CHECKS = []


def check(name):
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn
    return decorator


@check("1. 基本导入 (torch, sklearn, huggingface_hub)")
def _():
    import torch
    print(f"   torch {torch.__version__}, CUDA={torch.cuda.is_available()}")
    import sklearn
    print(f"   sklearn {sklearn.__version__}")
    import huggingface_hub
    print(f"   huggingface_hub {huggingface_hub.__version__}")


@check("2. src 包导入")
def _():
    from src.reproducibility import seed_everything
    from src.data.dataset import prepare_dataloaders, build_feature_tables
    from src.models.cdan_gsn import DomainStratifiedGatedNet
    from src.train.trainer import run_source_pretrain, run_domain_stratified_da
    from src.evaluate.metrics import evaluate_survival_metrics
    print("   所有核心模块导入成功")


@check("3. feature_allowlist.csv 存在且可读")
def _():
    import pandas as pd
    path = "experiments/audit/feature_allowlist.csv"
    assert os.path.exists(path), f"文件不存在: {path}"
    df = pd.read_csv(path)
    yes_count = (df["allowed"].str.lower() == "yes").sum()
    print(f"   {path}: {yes_count} 个允许特征")


@check("4. 数据加载 + 分类编码 + allowlist 过滤")
def _():
    from src.data.dataset import build_feature_tables
    x_s, x_t, e_s, t_s, e_t, t_t, features = build_feature_tables(
        "data/processed/深医_final_data.csv",
        "data/processed/福鼎_final_data.csv",
    )
    print(f"   深医: {x_s.shape}, 福鼎: {x_t.shape}")
    print(f"   最终特征数: {len(features)}")
    print(f"   特征: {features[:5]}...")


@check("5. prepare_dataloaders (patient-level split)")
def _():
    from src.data.dataset import prepare_dataloaders
    data = prepare_dataloaders(
        source_path="data/processed/深医_final_data.csv",
        target_path="data/processed/福鼎_final_data.csv",
        batch_size=256, seed=42,
        target_adapt_ratio=0.2, target_val_ratio=0.2,
        patient_col="患者id", split_strategy="patient",
    )
    print(f"   input_dim={data['input_dim']}, test sessions={len(data['idx_test'])}")
    print(f"   split_strategy={data['split_strategy']}")


@check("6. 模型初始化")
def _():
    import torch
    from src.data.dataset import prepare_dataloaders
    from src.models.cdan_gsn import DomainStratifiedGatedNet

    data = prepare_dataloaders(
        source_path="data/processed/深医_final_data.csv",
        target_path="data/processed/福鼎_final_data.csv",
        batch_size=256, seed=42,
        target_adapt_ratio=0.2, target_val_ratio=0.2,
        patient_col="患者id", split_strategy="patient",
    )
    model = DomainStratifiedGatedNet(
        input_dim=data["input_dim"], d_model=64, nhead=4,
        num_layers=2, dropout=0.15, domain_hidden=64,
        treat_indices=[0, 1], physio_indices=list(range(2, data["input_dim"])),
        tokenizer_type="kan", kan_basis_dim=8,
    )
    x = torch.randn(4, data["input_dim"])
    out = model(x, grl_coeff=None)
    print(f"   模型前向传播成功, 输出元组长度={len(out)}")


@check("7. 单 epoch source pretrain (快速验证)")
def _():
    import torch
    from src.data.dataset import prepare_dataloaders
    from src.models.cdan_gsn import DomainStratifiedGatedNet
    from src.train.trainer import run_source_pretrain

    data = prepare_dataloaders(
        source_path="data/processed/深医_final_data.csv",
        target_path="data/processed/福鼎_final_data.csv",
        batch_size=256, seed=42,
        target_adapt_ratio=0.2, target_val_ratio=0.2,
        patient_col="患者id", split_strategy="patient",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DomainStratifiedGatedNet(
        input_dim=data["input_dim"], d_model=64, nhead=4,
        num_layers=2, dropout=0.15, domain_hidden=64,
        treat_indices=[0, 1], physio_indices=list(range(2, data["input_dim"])),
        tokenizer_type="kan", kan_basis_dim=8,
    ).to(device)
    best_val, best_ep, state = run_source_pretrain(
        model, data["source_loader"],
        data["x_val"], data["e_val"], data["t_val"],
        lr=0.001, device=device, max_epochs=1,
    )
    print(f"   1 epoch pretrain 完成, val C-index={best_val:.4f}")


def main():
    print("=" * 50)
    print("DA-DSN 训练管线诊断")
    print("=" * 50)
    for name, fn in CHECKS:
        print(f"\n[{name}]")
        try:
            fn()
            print(f"   ✓ 通过")
        except Exception as e:
            print(f"   ✗ 失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n诊断停止于: {name}")
            print("请修复上述错误后重新运行 python diagnose.py")
            sys.exit(1)

    print("\n" + "=" * 50)
    print("所有检查通过！可以安全运行: python run_all.py --train")
    print("=" * 50)


if __name__ == "__main__":
    main()
