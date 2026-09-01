"""不可覆盖清单、并发安全 JSONL 哈希链与训练授权核查。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

GENESIS_HASH = "0" * 64


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _record_hash(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_hash"}
    return canonical_sha256(payload)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """创建清单；目标已存在时拒绝覆盖。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if "manifest_sha256" in manifest:
        raise ValueError("清单不得自行提供 manifest_sha256")
    payload = dict(manifest)
    payload["manifest_sha256"] = canonical_sha256(payload)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return target


def verify_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        manifest = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"清单不可读: {target}") from error
    if not isinstance(manifest, dict):
        raise ValueError("清单必须是 JSON 对象")
    stored_hash = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if stored_hash != canonical_sha256(payload):
        raise ValueError("清单自哈希不匹配")
    return manifest


def verify_jsonl_chain(path: str | Path) -> list[dict[str, Any]]:
    """读取并验证完整 JSONL 链，任何断链或篡改均报错。"""
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    previous_hash = GENESIS_HASH
    with target.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"JSONL 哈希链第 {line_number} 行为空")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL 哈希链第 {line_number} 行不是有效 JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL 哈希链第 {line_number} 行必须是对象")
            if record.get("previous_hash") != previous_hash:
                raise ValueError(f"JSONL 哈希链第 {line_number} 行断链")
            expected_hash = _record_hash(record)
            if record.get("record_hash") != expected_hash:
                raise ValueError(f"JSONL 哈希链第 {line_number} 行哈希不匹配")
            records.append(record)
            previous_hash = expected_hash
    return records


def append_jsonl(path: str | Path, event: Mapping[str, Any]) -> dict[str, Any]:
    """验证已有链后追加一个带前序哈希和自身哈希的事件。"""
    if "previous_hash" in event or "record_hash" in event:
        raise ValueError("事件不得自行提供 previous_hash 或 record_hash")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(target):
        records = verify_jsonl_chain(target)
        previous_hash = records[-1]["record_hash"] if records else GENESIS_HASH
        record = {**event, "previous_hash": previous_hash}
        record["record_hash"] = _record_hash(record)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return record


def require_explicit_authorization(
    authorization: Mapping[str, Any] | str | Path,
    manifest: Mapping[str, Any] | str | Path,
    run_plan: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """要求显式 true 授权，并绑定不可变 manifest 与 run-plan。"""
    if isinstance(authorization, (str, Path)):
        path = Path(authorization)
        if not path.is_file():
            raise PermissionError(f"授权记录不存在: {path}")
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise PermissionError(f"授权记录不可读: {path}") from error
    else:
        loaded = dict(authorization)
    if not isinstance(loaded, dict) or loaded.get("training_authorized") is not True:
        raise PermissionError("训练未获显式授权：training_authorized 必须为布尔值 true")
    binding = loaded.get("authorization")
    if not isinstance(binding, dict):
        raise PermissionError("授权缺少 manifest/run-plan 绑定")
    if isinstance(manifest, (str, Path)):
        try:
            manifest_hash = verify_manifest(manifest)["manifest_sha256"]
        except ValueError as error:
            raise PermissionError("绑定的 manifest 不可验证") from error
    else:
        manifest_payload = dict(manifest)
        stored_hash = manifest_payload.pop("manifest_sha256", None)
        manifest_hash = canonical_sha256(manifest_payload)
        if stored_hash is not None and stored_hash != manifest_hash:
            raise PermissionError("绑定的 manifest 自哈希不匹配")
    if binding.get("manifest_sha256") != manifest_hash:
        raise PermissionError("授权与 manifest 绑定不匹配")
    plan_hash = sha256_file(run_plan) if isinstance(run_plan, (str, Path)) else canonical_sha256(run_plan)
    if binding.get("run_plan_sha256") != plan_hash:
        raise PermissionError("授权与 run-plan 绑定不匹配")
    return loaded
