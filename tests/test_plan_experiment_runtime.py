from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plan_experiment_runtime import (  # noqa: E402
    build_runtime_plan,
    measured_variant_times,
    workload,
)


def spec(study_id, episodes, eval_episodes, seeds, variants, algorithms=None):
    algorithms = algorithms or ["imappo"] * len(variants)
    return {
        "study_id": study_id,
        "seeds": seeds,
        "environment": {"name": "test-uav"},
        "training": {
            "episodes": episodes,
            "steps": 100,
            "eval_interval": episodes,
        },
        "evaluation": {"episodes": eval_episodes, "risk_tiers": {"hard": {}}},
        "variants": [
            {"key": key, "algorithm": algorithm}
            for key, algorithm in zip(variants, algorithms)
        ],
    }


def test_workload_counts_training_and_all_tiers():
    value = spec("x", 10, 3, [7], ["a"])
    value["evaluation"]["risk_tiers"]["easy"] = {}
    assert workload(value) == {
        "training": 1000,
        "periodic_evaluation": 300,
        "collision_probe": 300,
        "final_evaluation": 600,
        "evaluation": 1200,
        "total": 2200,
    }


def test_workload_distinguishes_monitoring_probe_and_matd3():
    value = spec("x", 100, 20, [7], ["mappo", "matd3"], ["mappo", "matd3"])
    value["training"]["eval_interval"] = 25
    value["training"]["monitor_eval_episodes"] = 5
    on_policy = workload(value, value["variants"][0])
    off_policy = workload(value, value["variants"][1])
    assert on_policy["periodic_evaluation"] == 2000
    assert on_policy["collision_probe"] == 2000
    assert off_policy["periodic_evaluation"] == 0
    assert off_policy["collision_probe"] == 0


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


def test_process_cpu_calibration_is_not_labelled_as_gpu_time():
    reference = spec("calibration", 100, 20, [7], ["a"])
    paper = spec("paper", 2000, 100, list(range(10)), ["a"])
    plan = build_runtime_plan(
        paper,
        reference,
        {"a": 120.0},
        config_path="paper.json",
        reference_label="calibration",
        timing_field="process_cpu_time_seconds",
    )
    assert plan["schema_version"] == 2
    assert plan["timing_field"] == "process_cpu_time_seconds"
    assert "not GPU device time" in plan["timing_semantics"]
    assert "estimated_gpu_occupancy_hours_all_seeds_low" not in plan["per_variant"]["a"]


def test_measured_times_require_the_registered_timing_field(tmp_path):
    result = tmp_path / "a" / "seed_7" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        '{"variant":{"key":"a"},"resource_audit":{"wall_time_seconds":3.0}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="process_cpu_time_seconds"):
        measured_variant_times(
            tmp_path, {"a"}, timing_field="process_cpu_time_seconds"
        )
