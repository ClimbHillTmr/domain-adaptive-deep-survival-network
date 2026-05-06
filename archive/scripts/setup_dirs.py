import os
from pathlib import Path

def setup_project_structure(base_dir: str = "."):
    """
    建立标准化的项目目录结构
    """
    directories = [
        "data",
        "src/data",
        "src/features",
        "src/models",
        "src/train",
        "src/evaluate",
        "src/visualization",
        "experiments/logs",
        "experiments/results",
        "experiments/ablation",
        "figures",
        "tables",
        "notebooks",
        "tests",
        "docs",
        "conf/ablation"
    ]
    
    base_path = Path(base_dir)
    
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        # Create __init__.py in src subdirectories to make them modules
        if dir_path.startswith("src/"):
            init_file = full_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()

if __name__ == "__main__":
    setup_project_structure()
    print("Standard project structure created successfully.")
