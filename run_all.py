import os
import sys

# Ensure src is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.reproducibility import seed_everything, record_environment

def main():
    if "--train" not in sys.argv:
        print("Refusing to run training by default. Use: python run_all.py --train")
        return

    print("==================================================")
    print("🚀 Starting DA-DSN Full Pipeline Execution")
    print("==================================================")
    
    # 0. 固定环境与随机种子
    seed_everything(42)
    record_environment()
    
    # 1. 架构图生成 (Fig 1)
    print("\n[1/7] Generating Architecture Diagram (Fig 1)...")
    os.system("PYTHONPATH=. python src/visualization/generate_architecture.py")
    
    # 2. 生成基线特征表 (Table 1)
    print("\n[2/7] Generating Baseline Characteristics Table (Table 1)...")
    os.system("PYTHONPATH=. python src/generate_table1.py")
    
    # 3. 训练与评估主模型
    print("\n[3/7] Training and Evaluating Main CDAN-GSN Model...")
    os.system("PYTHONPATH=. python src/main.py")
    
    # 4. 执行消融实验 (Table S2)
    print("\n[4/7] Running Ablation Studies...")
    os.system("PYTHONPATH=. python src/train/run_ablations.py")
    
    # 5. 生成临床与可解释性图表 (Fig 2, Fig 3, Fig 4)
    print("\n[5/7] Generating Academic Clinical Figures (t-SNE, SHAP, KM Curves)...")
    os.system("PYTHONPATH=. python src/visualization/generate_figures.py")
    
    # 6. 打包学术交付物
    print("\n[6/7] Packaging Academic Deliverables...")
    os.system("zip -q -r submission_materials.zip figures/ tables/ conf/ src/ experiments/results/")
    print("-> Generated submission_materials.zip")
    
    print("\n[7/7] Pipeline Execution Completed Successfully!")

if __name__ == "__main__":
    main()
