from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plan_experiment_runtime import build_runtime_plan, workload  # noqa: E402


def spec(study_id, episodes, eval_episodes, seeds, variants, algorithms=None):
    algorithms = algorithms or ["imappo"] * len(variants)
    return {
        "study_id": study_id,
        "seeds": seeds,
        "training": {"episodes": episodes, "steps": 100},
        "evaluation": {"episodes": eval_episodes, "risk_tiers": {"hard": {}}},
        "variants": [
            {"key": key, "algorithm": algorithm}
            for key, algorithm in zip(variants, algorithms)
        ],
    }


def test_workload_counts_training_and_all_tiers():
    value = spec("x", 10, 3, [7], ["a"])
    value["evaluation"]["risk_tiers"]["easy"] = {}
    assert workload(value) == {"training": 1000, "evaluation": 600, "total": 1600}


def test_plan_preserves_protocol_and_emits_resumable_chunks():
    smoke = spec("smoke", 10, 3, [7], ["a", "b"])
    paper = spec("paper", 100, 10, [7, 11], ["a", "b"])
    plan = build_runtime_plan(
        paper, smoke, {"a": 10.0, "b": 12.0},
        config_path="paper.json", max_chunk_hours=0.1,
    )
    assert plan["run_count"] == 4
    assert plan["safety_contract"]["protocol_reduction_allowed"] is False
    assert "--resume" not in plan["chunks"][0]["command"]
    assert all("--resume" in item["command"] for item in plan["chunks"][1:])


def test_mixed_algorithm_slow_variant_is_not_misclassified_as_cold_start():
    smoke = spec("smoke", 10, 3, [7], ["imappo", "happo"], ["imappo", "happo"])
    paper = spec("paper", 100, 10, [7], ["imappo", "happo"], ["imappo", "happo"])
    plan = build_runtime_plan(
        paper, smoke, {"imappo": 5.0, "happo": 30.0}, config_path="paper.json"
    )
    assert plan["fixed_cold_start_seconds"] == 0.0
    assert plan["per_variant"]["happo"]["recurring_smoke_seconds"] == 30.0


def test_plan_rejects_variant_mismatch():
    smoke = spec("smoke", 10, 3, [7], ["a"])
    paper = spec("paper", 100, 10, [7], ["b"])
    with pytest.raises(ValueError, match="variants must match"):
        build_runtime_plan(paper, smoke, {"b": 1.0}, config_path="paper.json")
