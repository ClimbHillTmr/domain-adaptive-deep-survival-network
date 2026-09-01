import os
import json
import hashlib
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

try:
    from huggingface_hub import configure_http_backend
except ImportError:
    configure_http_backend = None


def configure_network(proxy: Optional[str] = None, use_mirror: bool = False):
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    if configure_http_backend is not None:
        try:
            import httpx
            transport = httpx.HTTPTransport(proxy=proxy) if proxy else None
            configure_http_backend(backend="httpx", transport=transport)
        except Exception:
            pass


class HFDataLoader:
    def __init__(
        self,
        owner: str = "hbd-survival",
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        proxy: Optional[str] = None,
        use_mirror: bool = False,
    ):
        self.owner = owner
        self.token = token or os.environ.get("HUGGINGFACE_TOKEN")
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "hbd-survival")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        configure_network(proxy=proxy, use_mirror=use_mirror)
        
        self.project_repos = {
            "data-shenyi-fuding": "dataset",
            "archive-backup": "dataset",
            "model-checkpoints": "model",
            "figures-publication": "dataset",
            "docs-reports": "dataset",
        }

    def _calculate_md5(self, file_path: str) -> str:
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _get_cache_path(self, repo_name: str, file_path: str) -> str:
        cache_key = hashlib.md5(f"{repo_name}/{file_path}".encode()).hexdigest()
        return os.path.join(self.cache_dir, repo_name, cache_key[:2], cache_key[2:], os.path.basename(file_path))

    def list_projects(self) -> List[str]:
        return list(self.project_repos.keys())

    def list_files(self, project_name: str) -> List[str]:
        if project_name not in self.project_repos:
            raise ValueError(f"Unknown project: {project_name}. Available projects: {self.list_projects()}")
        
        repo_id = f"{self.owner}/{project_name}"
        repo_type = self.project_repos[project_name]
        
        try:
            files = list_repo_files(repo_id=repo_id, repo_type=repo_type, token=self.token)
            return sorted(files)
        except Exception as e:
            raise ValueError(f"Failed to list files in {repo_id}: {str(e)}")

    def download_file(
        self,
        project_name: str,
        file_path: str,
        local_dir: Optional[str] = None,
        force_download: bool = False,
        return_metadata: bool = False,
    ) -> Union[str, Dict]:
        if project_name not in self.project_repos:
            raise ValueError(f"Unknown project: {project_name}. Available projects: {self.list_projects()}")
        
        repo_id = f"{self.owner}/{project_name}"
        repo_type = self.project_repos[project_name]
        
        cache_path = self._get_cache_path(project_name, file_path)
        
        if not force_download and os.path.exists(cache_path):
            target_path = cache_path
        else:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            target_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type=repo_type,
                cache_dir=os.path.dirname(cache_path),
                token=self.token,
                local_files_only=False,
            )
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            if target_path != cache_path:
                import shutil
                shutil.copy2(target_path, cache_path)
        
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, os.path.basename(file_path))
            import shutil
            shutil.copy2(cache_path, local_path)
            target_path = local_path
        
        if return_metadata:
            stat = os.stat(target_path)
            return {
                "file_path": target_path,
                "file_name": os.path.basename(target_path),
                "file_size": stat.st_size,
                "md5_hash": self._calculate_md5(target_path),
                "downloaded_at": datetime.now().isoformat(),
                "project_name": project_name,
                "original_path": file_path,
            }
        
        return target_path

    def download_project(
        self,
        project_name: str,
        local_dir: str,
        force_download: bool = False,
    ) -> List[Dict]:
        if project_name not in self.project_repos:
            raise ValueError(f"Unknown project: {project_name}. Available projects: {self.list_projects()}")
        
        os.makedirs(local_dir, exist_ok=True)
        
        downloaded_files = []
        files = self.list_files(project_name)
        
        for file_path in files:
            result = self.download_file(
                project_name=project_name,
                file_path=file_path,
                local_dir=os.path.join(local_dir, os.path.dirname(file_path)),
                force_download=force_download,
                return_metadata=True,
            )
            downloaded_files.append(result)
        
        return downloaded_files

    def load_csv(
        self,
        project_name: str,
        file_path: str,
        **kwargs,
    ) -> pd.DataFrame:
        local_path = self.download_file(project_name, file_path)
        return pd.read_csv(local_path, **kwargs)

    def load_json(
        self,
        project_name: str,
        file_path: str,
    ) -> Dict:
        local_path = self.download_file(project_name, file_path)
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_model(
        self,
        project_name: str,
        file_path: str,
        device: str = "cpu",
    ) -> torch.nn.Module:
        local_path = self.download_file(project_name, file_path)
        return torch.load(local_path, map_location=device, weights_only=True)

    def load_pickle(
        self,
        project_name: str,
        file_path: str,
    ) -> Any:
        import pickle
        local_path = self.download_file(project_name, file_path)
        with open(local_path, "rb") as f:
            return pickle.load(f)

    def get_project_info(self, project_name: str) -> Dict:
        if project_name not in self.project_repos:
            raise ValueError(f"Unknown project: {project_name}. Available projects: {self.list_projects()}")
        
        repo_id = f"{self.owner}/{project_name}"
        repo_type = self.project_repos[project_name]
        
        files = self.list_files(project_name)
        
        total_size = 0
        for file_path in files:
            try:
                local_path = self.download_file(project_name, file_path, return_metadata=False)
                total_size += os.path.getsize(local_path)
            except Exception:
                pass
        
        return {
            "project_name": project_name,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "file_count": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "files": files,
        }

    def validate_integrity(
        self,
        project_name: str,
        expected_md5s: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict]:
        results = {}
        files = self.list_files(project_name)
        
        for file_path in files:
            metadata = self.download_file(project_name, file_path, return_metadata=True)
            actual_md5 = metadata["md5_hash"]
            
            result = {
                "file_path": file_path,
                "actual_md5": actual_md5,
                "file_size": metadata["file_size"],
                "valid": True,
            }
            
            if expected_md5s and file_path in expected_md5s:
                expected_md5 = expected_md5s[file_path]
                result["expected_md5"] = expected_md5
                result["valid"] = actual_md5 == expected_md5
            
            results[file_path] = result
        
        return results


def load_hf_data(
    project_name: str,
    file_path: str,
    file_type: Optional[str] = None,
    **kwargs,
) -> Any:
    loader = HFDataLoader()
    
    if file_type is None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            file_type = "csv"
        elif ext == ".json":
            file_type = "json"
        elif ext in (".pt", ".pth"):
            file_type = "model"
        elif ext == ".pkl":
            file_type = "pickle"
        else:
            raise ValueError(f"Unknown file type for {file_path}. Specify file_type parameter.")
    
    if file_type == "csv":
        return loader.load_csv(project_name, file_path, **kwargs)
    elif file_type == "json":
        return loader.load_json(project_name, file_path)
    elif file_type == "model":
        return loader.load_model(project_name, file_path, **kwargs)
    elif file_type == "pickle":
        return loader.load_pickle(project_name, file_path)
    else:
        raise ValueError(f"Unsupported file_type: {file_type}")