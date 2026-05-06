import os
import random
import numpy as np
import torch
import subprocess
from pathlib import Path
import datetime

def seed_everything(seed: int = 42) -> None:
    """
    固定所有随机种子，确保实验的绝对可复现性。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 强制确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def record_environment(log_dir: str = "experiments/logs") -> None:
    """
    记录当前运行的 Python 环境依赖，保存到 logs 目录下。
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    env_file = Path(log_dir) / f"pip_freeze_{timestamp}.txt"
    try:
        with open(env_file, "w") as f:
            subprocess.run(["pip", "freeze"], stdout=f, check=True)
        print(f"环境依赖已记录至: {env_file}")
    except Exception as e:
        print(f"环境记录失败: {e}")
