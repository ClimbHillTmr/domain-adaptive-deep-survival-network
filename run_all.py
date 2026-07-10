import os
import sys
import subprocess

# Ensure src is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.reproducibility import seed_everything, record_environment


def run_step(description, cmd, critical=True):
    """Run a subprocess and check for errors."""
    print(f"\n{'='*50}")
    print(f"  {description}")
    print(f"{'='*50}")
    result = subprocess.run(
        cmd, shell=True,
        env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.abspath(__file__))},
    )
    if result.returncode != 0:
        msg = f"FAILED (exit code {result.returncode}): {description}"
        print(f"\n❌ {msg}")
        if critical:
            print("Pipeline halted — fix the error above before continuing.")
            sys.exit(result.returncode)
        else:
            print("⚠️  Non-critical step failed, continuing...")
    else:
        print(f"✓ {description}")
    return result.returncode


def main():
    if "--train" not in sys.argv:
        print("Refusing to run training by default. Use: python run_all.py --train")
        return

    print("==================================================")
    print("Starting DA-DSN Full Pipeline Execution")
    print("==================================================")

    # 0. 固定环境与随机种子
    seed_everything(42)
    record_environment()

    # 1. 架构图生成 (Fig 1) — non-critical
    run_step("[1/7] Generating Architecture Diagram (Fig 1)",
             "python src/visualization/generate_architecture.py", critical=False)

    # 2. 生成基线特征表 (Table 1) — non-critical
    run_step("[2/7] Generating Baseline Characteristics Table (Table 1)",
             "python src/generate_table1.py", critical=False)

    # 3. 训练与评估主模型 — CRITICAL
    run_step("[3/7] Training and Evaluating Main CDAN-GSN Model",
             "python src/main.py", critical=True)

    # 4. 导出真实预测与校准/DCA 指标 — CRITICAL
    run_step("[4/7] Exporting Real Test Predictions & Calibration Metrics",
             "python src/evaluate/export_real_predictions.py", critical=True)

    # 5. 执行消融实验 (Table S2) — non-critical (can rerun separately)
    run_step("[5/7] Running Ablation Studies",
             "python src/train/run_ablations.py", critical=False)

    # 6. 生成临床与可解释性图表
    run_step("[6/7] Generating Academic Clinical Figures",
             "python src/visualization/generate_figures.py", critical=False)

    # 7. 打包学术交付物
    run_step("[7/7] Packaging Academic Deliverables",
             "zip -q -r submission_materials.zip figures/ tables/ conf/ src/ experiments/results/",
             critical=False)

    print("\n==================================================")
    print("Pipeline Execution Completed")
    print("==================================================")
    print("Key outputs to verify:")
    print("  experiments/results/evaluation_results.json")
    print("  experiments/results/real_test_predictions.csv")
    print("  experiments/results/calibration_dca_metrics.json")
    print("  experiments/results/cdan_gsn_final.pt")


if __name__ == "__main__":
    main()
