import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from final_ood_registration import (  # noqa: E402
    load_final_ood_registration,
    validate_final_ood_registration,
)


REGISTRATION = ROOT / "configs" / "data" / "citynav_final_ood_registration.v2.json"
GATE = ROOT / "configs" / "research" / "preference_relevance_gate.aerialvln_dev.v1.json"


def test_citynav_registration_validates_against_frozen_gate():
    result = validate_final_ood_registration(
        load_final_ood_registration(REGISTRATION), gate_path=GATE
    )
    assert result["dataset_id"] == "citynav-iccv2025"
    assert result["gate_verified"] is True
    assert result["pass_if_far_at_most"] == 0.05


def test_registration_rejects_post_evaluation_retuning():
    payload = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(payload)
    mutated["blinding_and_change_control"][
        "gate_or_threshold_changes_after_evaluation_allowed"
    ] = True
    with pytest.raises(ValueError, match="change-control"):
        validate_final_ood_registration(mutated)


def test_superseded_broad_glob_registration_is_not_runnable():
    old = load_final_ood_registration(
        ROOT / "configs" / "data" / "citynav_final_ood_registration.v1.json"
    )
    with pytest.raises(ValueError, match="frozen before text access"):
        validate_final_ood_registration(old)


def test_registration_rejects_gate_hash_mismatch(tmp_path: Path):
    changed_gate = tmp_path / "gate.json"
    changed_gate.write_bytes(GATE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_final_ood_registration(
            load_final_ood_registration(REGISTRATION), gate_path=changed_gate
        )
