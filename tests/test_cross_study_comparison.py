import copy
import hashlib
import json
from pathlib import Path

import pytest

from compare_research_studies import validate_compatibility, verify_checksums


def compatible_specs():
    return {
        "seeds": [7, 11],
        "environment": {"name": "uav", "n_agents": 4},
        "evaluation": {"episodes": 5, "risk_tiers": {"easy": {}}},
        "generalization": {"suite": "suite.json"},
        "training": {"episodes": 50},
    }


def test_pair_compatibility_ignores_algorithm_training_details():
    treatment = compatible_specs()
    baseline = copy.deepcopy(treatment)
    baseline["training"] = {"episodes": 0, "controller": "rule"}
    report = validate_compatibility(treatment, baseline)
    assert report["status"] == "compatible"
    assert report["seed_count"] == 2


def test_pair_compatibility_rejects_evaluation_mismatch():
    treatment = compatible_specs()
    baseline = copy.deepcopy(treatment)
    baseline["evaluation"]["episodes"] = 100
    with pytest.raises(ValueError, match="evaluation"):
        validate_compatibility(treatment, baseline)


def test_checksum_audit_detects_source_mutation(tmp_path: Path):
    artifact = tmp_path / "result.json"
    artifact.write_text(json.dumps({"value": 1}), encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "checksums.sha256").write_text(
        f"{digest}  result.json\n", encoding="utf-8"
    )
    assert verify_checksums(tmp_path)["files_checked"] == 1
    artifact.write_text(json.dumps({"value": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_checksums(tmp_path)
