from scripts.v3_scientific_audit import audit


def test_v3_evidence_tree_is_structurally_complete_but_training_is_blocked():
    report = audit()

    assert report["structure_passed"] is True
    assert report["data_preflight_ready"] is True
    assert report["training_ready"] is False
    assert report["status"] == "blocked"
    assert report["counts"]["evidence_nodes"] >= 15
    assert report["counts"]["random_control_families"] == 4
    assert report["counts"]["v3_registered_runs"] == 0
    assert "phase_status_training_authorized_false" in report["blockers"]
    assert "v3_server_run_grid_and_training_launcher_not_generated_or_reviewed" not in report["execution_gates"]


def test_v3_experiments_are_indexed_by_evidence_nodes():
    report = audit()

    assert not any("experiment_tree_matrix_mismatch" in error for error in report["errors"])
    assert not any("tree_experiment_missing_from_matrix" in error for error in report["errors"])


def test_v3_does_not_require_adaptive_alignment_before_fixed_hypothesis_test():
    report = audit()

    assert not any("adaptive_alignment" in blocker for blocker in report["blockers"])


def test_scope_and_downstream_limits_do_not_masquerade_as_training_gates():
    report = audit()

    assert "measurement_process_tier_not_observed_in_current_common_feature_table" in report["scope_limitations"]
    assert not any("measurement_process_tier" in blocker for blocker in report["blockers"])
    assert "new_temporal_or_external_confirmatory_test_cohort_not_identified" in report["downstream_gates"]
    assert "clinical_threshold_and_alert_workflow_not_prespecified" in report["downstream_gates"]


def test_claude_contract_locks_the_canonical_data_entrypoints_and_protocol():
    report = audit()

    assert not any("research_contract_missing" in error for error in report["errors"])
    assert not any("data_protocol_version_mismatch" in error for error in report["errors"])
