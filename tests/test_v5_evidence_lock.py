import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.evidence_lock import (
    GENESIS_HASH,
    append_jsonl,
    require_explicit_authorization,
    verify_jsonl_chain,
    verify_manifest,
    write_manifest,
)


def test_manifest_cannot_be_overwritten(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(path, {"protocol_version": "v5", "training_authorized": False})

    with pytest.raises(FileExistsError):
        write_manifest(path, {"protocol_version": "changed"})

    assert verify_manifest(path)["protocol_version"] == "v5"


def test_jsonl_records_form_a_verified_hash_chain(tmp_path):
    path = tmp_path / "evidence.jsonl"
    first = append_jsonl(path, {"event": "prepared"})
    second = append_jsonl(path, {"event": "audited", "passed": True})

    records = verify_jsonl_chain(path)
    assert first["previous_hash"] == GENESIS_HASH
    assert second["previous_hash"] == first["record_hash"]
    assert records == [first, second]


def test_jsonl_chain_rejects_tampering_before_append(tmp_path):
    path = tmp_path / "evidence.jsonl"
    append_jsonl(path, {"event": "prepared"})
    record = json.loads(path.read_text(encoding="utf-8"))
    record["event"] = "tampered"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="哈希不匹配"):
        verify_jsonl_chain(path)
    with pytest.raises(ValueError, match="哈希不匹配"):
        append_jsonl(path, {"event": "must_not_append"})



def test_manifest_self_hash_detects_tampering(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(path, {"protocol_version": "v5", "training_authorized": False})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["manifest_sha256"]) == 64
    payload["protocol_version"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="自哈希不匹配"):
        verify_manifest(path)


def test_concurrent_jsonl_appends_remain_complete_chain(tmp_path):
    path = tmp_path / "evidence.jsonl"
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda index: append_jsonl(path, {"event": "gate", "index": index}), range(40)))
    verified = verify_jsonl_chain(path)
    assert len(verified) == 40
    assert {record["record_hash"] for record in verified} == {record["record_hash"] for record in records}


def test_authorization_requires_literal_boolean_true_and_bindings(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    run_plan_path = tmp_path / "run_plan.json"
    write_manifest(manifest_path, {"protocol_version": "v5", "training_authorized": False})
    run_plan_path.write_text(json.dumps({"runs": []}), encoding="utf-8")
    manifest_hash = verify_manifest(manifest_path)["manifest_sha256"]
    from src.evidence_lock import sha256_file
    binding = {"manifest_sha256": manifest_hash, "run_plan_sha256": sha256_file(run_plan_path)}
    for value in (False, None, 1, "true"):
        authorization = {"training_authorized": value, "authorization": binding}
        with pytest.raises(PermissionError, match="显式授权"):
            require_explicit_authorization(authorization, manifest_path, run_plan_path)
    authorization = {"training_authorized": True, "scope": "v5", "authorization": binding}
    assert require_explicit_authorization(authorization, manifest_path, run_plan_path)["scope"] == "v5"
    authorization["authorization"]["run_plan_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="run-plan 绑定不匹配"):
        require_explicit_authorization(authorization, manifest_path, run_plan_path)


def test_authorization_rejects_missing_record(tmp_path):
    manifest = {"training_authorized": False}
    run_plan = {"runs": []}
    with pytest.raises(PermissionError, match="不存在"):
        require_explicit_authorization(tmp_path / "missing.json", manifest, run_plan)
