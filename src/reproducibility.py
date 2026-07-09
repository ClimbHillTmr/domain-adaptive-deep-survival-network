import datetime
import hashlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

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


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_fingerprint(config: Dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(config).encode("utf-8")).hexdigest()


def prepare_locked_run_context(
    config: Dict[str, Any],
    tracked_inputs: Dict[str, str],
    run_root: str = "experiments/runs",
    run_prefix: str = "locked_run",
) -> Dict[str, Any]:
    """
    Create a deterministic run directory with a run_id derived from the config and
    tracked input hashes. The timestamp keeps runs human-sortable, while the hash
    suffix prevents collisions across materially different runs.
    """
    normalized_inputs = {}
    for key, value in tracked_inputs.items():
        path = Path(value)
        normalized_inputs[key] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.exists() else None,
        }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    signature_payload = {
        "config_sha256": config_fingerprint(config),
        "inputs": normalized_inputs,
    }
    signature = hashlib.sha256(_stable_json(signature_payload).encode("utf-8")).hexdigest()[:12]
    run_id = f"{run_prefix}_{timestamp}_{signature}"

    run_dir = ROOT / run_root / run_id
    logs_dir = run_dir / "logs"
    snapshot_dir = run_dir / "snapshots"
    artifact_dir = run_dir / "artifacts"
    for path in (run_dir, logs_dir, snapshot_dir, artifact_dir):
        path.mkdir(parents=True, exist_ok=True)

    context = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "logs_dir": str(logs_dir),
        "snapshot_dir": str(snapshot_dir),
        "artifact_dir": str(artifact_dir),
        "created_at": timestamp,
        "config_sha256": signature_payload["config_sha256"],
        "tracked_inputs": normalized_inputs,
    }
    write_json(run_dir / "run_context.json", context)
    return context


def snapshot_files(file_paths: Iterable[str], destination_dir: str) -> None:
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for file_path in file_paths:
        source = Path(file_path)
        if not source.exists():
            continue
        shutil.copy2(source, destination / source.name)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def record_environment(log_dir: str = "experiments/logs", filename: str = None) -> Path:
    """
    记录当前运行的 Python 环境依赖，保存到 logs 目录下。
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    env_file = log_path / (filename or f"pip_freeze_{timestamp}.txt")
    try:
        with open(env_file, "w") as f:
            subprocess.run(["pip", "freeze"], stdout=f, check=True)
        print(f"环境依赖已记录至: {env_file}")
    except Exception as e:
        print(f"环境记录失败: {e}")
    return env_file
