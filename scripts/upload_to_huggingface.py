import os
import json
import hashlib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError

try:
    from huggingface_hub import configure_http_backend
except ImportError:
    configure_http_backend = None


def configure_network(proxy: Optional[str] = None, use_mirror: bool = False):
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        print(f"Using proxy: {proxy}")
    
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using HF mirror: https://hf-mirror.com")


def get_token(token_arg: Optional[str]) -> str:
    if token_arg:
        return token_arg
    if "HUGGINGFACE_TOKEN" in os.environ:
        return os.environ["HUGGINGFACE_TOKEN"]
    token_file = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            return f.read().strip()
    raise ValueError("No token provided. Use --token, HUGGINGFACE_TOKEN env var, or ~/.cache/huggingface/token")


class DataUploader:
    def __init__(self, token: str, owner: str = "hbd-survival"):
        self.api = HfApi(token=token)
        self.owner = owner
        self.upload_metadata = {}

    def _calculate_md5(self, file_path: str) -> str:
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _get_file_metadata(self, file_path: str) -> Dict:
        stat = os.stat(file_path)
        return {
            "file_name": os.path.basename(file_path),
            "file_size": stat.st_size,
            "md5_hash": self._calculate_md5(file_path),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "absolute_path": os.path.abspath(file_path),
        }

    def _create_repo_if_not_exists(self, repo_name: str, repo_type: str = "dataset") -> str:
        repo_id = f"{self.owner}/{repo_name}"
        try:
            self.api.repo_info(repo_id=repo_id, repo_type=repo_type)
            print(f"Repository {repo_id} already exists")
        except RepositoryNotFoundError:
            print(f"Creating repository {repo_id}...")
            create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=self.api.token)
        except Exception as e:
            print(f"Error checking repo: {e}. Assuming repo needs creation...")
            try:
                create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=self.api.token)
            except Exception as create_e:
                print(f"Failed to create repo: {create_e}")
                raise
        return repo_id

    def upload_file_to_repo(self, local_path: str, repo_id: str, repo_type: str, hf_path: str) -> Dict:
        metadata = self._get_file_metadata(local_path)
        
        try:
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=hf_path,
                repo_id=repo_id,
                repo_type=repo_type,
                token=self.api.token,
            )
            print(f"✓ Uploaded: {hf_path} ({metadata['file_size']/1024/1024:.2f} MB)")
            metadata["upload_success"] = True
            metadata["uploaded_at"] = datetime.now().isoformat()
            metadata["repo_id"] = repo_id
            metadata["path_in_repo"] = hf_path
        except Exception as e:
            print(f"✗ Failed to upload {hf_path}: {str(e)}")
            metadata["upload_success"] = False
            metadata["error_message"] = str(e)
        
        return metadata

    def upload_directory(self, local_dir: str, repo_id: str, repo_type: str, base_path: str = "") -> List[Dict]:
        results = []
        all_files = []
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_dir)
                hf_path = os.path.join(base_path, rel_path) if base_path else rel_path
                all_files.append((local_path, hf_path))
        
        print(f"  Found {len(all_files)} files in {local_dir}")
        
        for local_path, hf_path in all_files:
            result = self.upload_file_to_repo(local_path, repo_id, repo_type, hf_path)
            results.append(result)
        
        return results

    def upload_directory_bulk(self, local_dir: str, repo_id: str, repo_type: str) -> Dict:
        try:
            result = upload_folder(
                folder_path=local_dir,
                repo_id=repo_id,
                repo_type=repo_type,
                token=self.api.token,
            )
            print(f"✓ Bulk upload completed for {local_dir}")
            
            results = []
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, local_dir)
                    metadata = self._get_file_metadata(local_path)
                    metadata["upload_success"] = True
                    metadata["uploaded_at"] = datetime.now().isoformat()
                    metadata["repo_id"] = repo_id
                    metadata["path_in_repo"] = rel_path
                    results.append(metadata)
            
            return {"success": True, "results": results, "api_result": str(result)}
        except Exception as e:
            print(f"✗ Bulk upload failed for {local_dir}: {str(e)}")
            return {"success": False, "error": str(e), "results": []}

    def save_upload_report(self, output_path: str):
        report = {
            "generated_at": datetime.now().isoformat(),
            "owner": self.owner,
            "upload_metadata": self.upload_metadata,
            "summary": self.get_summary(),
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\nUpload report saved to: {output_path}")

    def get_summary(self) -> Dict:
        summary = {"total_files": 0, "success_count": 0, "failed_count": 0, "total_size_bytes": 0}
        
        for project, files in self.upload_metadata.items():
            summary["total_files"] += len(files)
            summary["total_size_bytes"] += sum(f.get("file_size", 0) for f in files)
            summary["success_count"] += sum(1 for f in files if f.get("upload_success"))
            summary["failed_count"] += sum(1 for f in files if not f.get("upload_success"))
        
        summary["total_size_mb"] = summary["total_size_bytes"] / (1024 * 1024)
        return summary


def main():
    parser = argparse.ArgumentParser(description="Upload data files to Hugging Face")
    parser.add_argument("--token", help="Hugging Face API token (or use HUGGINGFACE_TOKEN env)")
    parser.add_argument("--owner", default="LongGoodbye", help="Hugging Face organization/user name")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be uploaded")
    parser.add_argument("--proxy", help="HTTP proxy for network access")
    parser.add_argument("--mirror", action="store_true", help="Use hf-mirror.com for China access")
    parser.add_argument("--retry", type=int, default=3, help="Number of retries for failed uploads")
    parser.add_argument("--bulk", action="store_true", help="Use bulk upload (faster for large directories)")
    args = parser.parse_args()

    configure_network(proxy=args.proxy, use_mirror=args.mirror)
    token = get_token(args.token)
    uploader = DataUploader(token=token, owner=args.owner)
    
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")

    projects = {
        "data-shenyi-fuding": {
            "repo_type": "dataset",
            "paths": [
                {"local": project_root / "data" / "raw", "base_path": "raw"},
                {"local": project_root / "data" / "processed", "base_path": "processed"},
            ],
        },
        "archive-backup": {
            "repo_type": "dataset",
            "paths": [
                {"local": project_root / "archive" / "backup", "base_path": ""},
            ],
        },
        "model-checkpoints": {
            "repo_type": "model",
            "paths": [
                {"local": project_root / "archive" / "runs", "base_path": ""},
            ],
        },
        "figures-publication": {
            "repo_type": "dataset",
            "paths": [
                {"local": project_root / "figures", "base_path": ""},
            ],
        },
        "docs-reports": {
            "repo_type": "dataset",
            "paths": [
                {"local": project_root / "docs", "base_path": ""},
            ],
        },
    }

    if args.dry_run:
        print("\n=== DRY RUN - Would upload the following files ===")
        total_size = 0
        file_count = 0
        for project_name, config in projects.items():
            print(f"\nProject: {project_name}")
            for path_cfg in config["paths"]:
                local_dir = path_cfg["local"]
                base_path = path_cfg["base_path"]
                if not local_dir.exists():
                    print(f"  ⚠ Directory not found: {local_dir}")
                    continue
                for root, dirs, files in os.walk(local_dir):
                    for file in files:
                        local_path = os.path.join(root, file)
                        rel_path = os.path.relpath(local_path, local_dir)
                        hf_path = os.path.join(base_path, rel_path) if base_path else rel_path
                        file_size = os.path.getsize(local_path)
                        total_size += file_size
                        file_count += 1
                        print(f"  {hf_path} ({file_size/1024/1024:.2f} MB)")
        
        print(f"\nTotal: {file_count} files, {total_size/1024/1024:.2f} MB")
        return

    print("\n=== Starting upload ===")
    
    for project_name, config in projects.items():
        print(f"\n--- Processing project: {project_name} ---")
        repo_id = uploader._create_repo_if_not_exists(project_name, config["repo_type"])
        
        uploader.upload_metadata[project_name] = []
        
        for path_cfg in config["paths"]:
            local_dir = path_cfg["local"]
            base_path = path_cfg["base_path"]
            
            if not local_dir.exists():
                print(f"  ⚠ Directory not found: {local_dir}")
                continue
            
            if args.bulk:
                result = uploader.upload_directory_bulk(
                    local_dir=str(local_dir),
                    repo_id=repo_id,
                    repo_type=config["repo_type"],
                )
                if result.get("success"):
                    uploader.upload_metadata[project_name].extend(result["results"])
            else:
                results = uploader.upload_directory(
                    local_dir=str(local_dir),
                    repo_id=repo_id,
                    repo_type=config["repo_type"],
                    base_path=base_path,
                )
                uploader.upload_metadata[project_name].extend(results)

    summary = uploader.get_summary()
    print("\n=== Upload Summary ===")
    print(f"Total files: {summary['total_files']}")
    print(f"Success: {summary['success_count']}")
    print(f"Failed: {summary['failed_count']}")
    print(f"Total size: {summary['total_size_mb']:.2f} MB")

    report_path = project_root / "huggingface_upload_report.json"
    uploader.save_upload_report(str(report_path))


if __name__ == "__main__":
    main()