import os
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from huggingface_hub import HfApi, list_repo_files, hf_hub_download


def configure_network(proxy: Optional[str] = None, use_mirror: bool = False):
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


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


def calculate_md5(file_path: str) -> str:
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def validate_upload(token: str, owner: str = "hbd-survival", use_server_metadata: bool = True):
    api = HfApi(token=token)
    
    projects = {
        "data-shenyi-fuding": "dataset",
        "archive-backup": "dataset",
        "model-checkpoints": "model",
        "figures-publication": "dataset",
        "docs-reports": "dataset",
    }
    
    project_root = Path(__file__).parent.parent
    
    validation_report = {
        "generated_at": datetime.now().isoformat(),
        "owner": owner,
        "use_server_metadata": use_server_metadata,
        "projects": {},
        "summary": {"total_files": 0, "valid_files": 0, "invalid_files": 0, "missing_files": 0},
    }
    
    for project_name, repo_type in projects.items():
        repo_id = f"{owner}/{project_name}"
        print(f"\n=== Validating project: {project_name} ===")
        
        try:
            hf_files = list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)
        except Exception as e:
            print(f"  ⚠ Failed to list files in {repo_id}: {str(e)}")
            validation_report["projects"][project_name] = {
                "status": "error",
                "error": str(e),
                "files": [],
            }
            continue
        
        local_files = {}
        local_dirs = {
            "data-shenyi-fuding": [
                (project_root / "data" / "raw", "raw"),
                (project_root / "data" / "processed", "processed"),
            ],
            "archive-backup": [
                (project_root / "archive" / "backup", ""),
            ],
            "model-checkpoints": [
                (project_root / "archive" / "runs", ""),
            ],
            "figures-publication": [
                (project_root / "figures", ""),
            ],
            "docs-reports": [
                (project_root / "docs", ""),
            ],
        }
        
        for local_dir, base_path in local_dirs.get(project_name, []):
            if not local_dir.exists():
                continue
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, local_dir)
                    hf_path = f"{base_path}/{rel_path}" if base_path else rel_path
                    
                    local_files[hf_path] = {
                        "path": local_path,
                        "md5": calculate_md5(local_path),
                        "size": os.path.getsize(local_path),
                    }
        
        project_report = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "local_file_count": len(local_files),
            "hf_file_count": len(hf_files),
            "files": [],
        }
        
        for hf_path in hf_files:
            if hf_path in local_files:
                local_info = local_files[hf_path]
                
                if use_server_metadata:
                    try:
                        repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
                        hf_size = None
                        if hasattr(repo_info, 'siblings'):
                            for sibling in repo_info.siblings:
                                if sibling.rfilename == hf_path:
                                    hf_size = sibling.size
                                    break
                        
                        if hf_size is not None:
                            is_valid = hf_size == local_info["size"]
                            project_report["files"].append({
                                "path": hf_path,
                                "status": "valid" if is_valid else "invalid",
                                "local_size": local_info["size"],
                                "hf_size": hf_size,
                                "validation_method": "server_metadata",
                            })
                        else:
                            raise ValueError("Could not get server metadata")
                    except Exception as e:
                        print(f"  ⚠ Could not get server metadata for {hf_path}, falling back to download...")
                        use_server_metadata = False
                
                if not use_server_metadata:
                    hf_download_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=hf_path,
                        repo_type=repo_type,
                        token=token,
                        local_files_only=False,
                    )
                    hf_md5 = calculate_md5(hf_download_path)
                    hf_size = os.path.getsize(hf_download_path)
                    
                    is_valid = (hf_md5 == local_info["md5"]) and (hf_size == local_info["size"])
                    
                    project_report["files"].append({
                        "path": hf_path,
                        "status": "valid" if is_valid else "invalid",
                        "local_md5": local_info["md5"],
                        "hf_md5": hf_md5,
                        "local_size": local_info["size"],
                        "hf_size": hf_size,
                        "validation_method": "download_and_compare",
                    })
                
                validation_report["summary"]["total_files"] += 1
                if project_report["files"][-1]["status"] == "valid":
                    validation_report["summary"]["valid_files"] += 1
                    print(f"  ✓ {hf_path}")
                else:
                    validation_report["summary"]["invalid_files"] += 1
                    print(f"  ✗ {hf_path} (SIZE mismatch)")
            else:
                project_report["files"].append({
                    "path": hf_path,
                    "status": "extra",
                    "message": "File exists on HF but not locally",
                })
                print(f"  ⚠ {hf_path} (extra file on HF)")
        
        for local_path in local_files:
            if local_path not in hf_files:
                project_report["files"].append({
                    "path": local_path,
                    "status": "missing",
                    "message": "File exists locally but not on HF",
                })
                validation_report["summary"]["missing_files"] += 1
                print(f"  ✗ {local_path} (missing on HF)")
        
        project_report["valid_count"] = sum(1 for f in project_report["files"] if f["status"] == "valid")
        project_report["invalid_count"] = sum(1 for f in project_report["files"] if f["status"] == "invalid")
        project_report["missing_count"] = sum(1 for f in project_report["files"] if f["status"] == "missing")
        project_report["extra_count"] = sum(1 for f in project_report["files"] if f["status"] == "extra")
        
        project_report["status"] = "success" if project_report["invalid_count"] == 0 and project_report["missing_count"] == 0 else "partial"
        validation_report["projects"][project_name] = project_report
    
    print("\n=== Validation Summary ===")
    print(f"Total files validated: {validation_report['summary']['total_files']}")
    print(f"Valid: {validation_report['summary']['valid_files']}")
    print(f"Invalid: {validation_report['summary']['invalid_files']}")
    print(f"Missing: {validation_report['summary']['missing_files']}")
    
    report_path = project_root / "huggingface_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, ensure_ascii=False, indent=2)
    
    print(f"\nValidation report saved to: {report_path}")
    
    return validation_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Hugging Face upload integrity")
    parser.add_argument("--token", help="Hugging Face API token (or use HUGGINGFACE_TOKEN env)")
    parser.add_argument("--owner", default="hbd-survival", help="Hugging Face organization/user name")
    parser.add_argument("--proxy", help="HTTP proxy for network access")
    parser.add_argument("--mirror", action="store_true", help="Use hf-mirror.com for China access")
    parser.add_argument("--no-server-metadata", action="store_true", help="Always download files for validation instead of using server metadata")
    args = parser.parse_args()
    
    configure_network(proxy=args.proxy, use_mirror=args.mirror)
    token = get_token(args.token)
    
    validate_upload(token, args.owner, use_server_metadata=not args.no_server_metadata)