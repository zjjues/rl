"""Estimate paper-study runtime and emit safe resumable run chunks.

The estimate extrapolates a registered smoke or calibration timing field by
environment-step workload. Wall and process-CPU time are kept semantically
distinct; neither is relabelled as measured GPU compute time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Dict, Mapping


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def workload(
    spec: Mapping[str, object], variant: Mapping[str, object] | None = None
) -> Dict[str, int]:
    """Count executed environment steps, including in-training evaluations."""
    steps = int(spec["training"]["steps"])
    training_episodes = int(spec["training"]["episodes"])
    algorithm = str((variant or {}).get("algorithm", "imappo"))
    training = (
        0 if algorithm == "rule_planner" else training_episodes * steps
    )
    periodic_evaluation = 0
    collision_probe = 0
    if algorithm not in {"rule_planner", "matd3"}:
        interval = max(int(spec["training"].get("eval_interval", training_episodes)), 1)
        events = math.ceil(training_episodes / interval)
        monitor_episodes = int(
            spec["training"].get(
                "monitor_eval_episodes", spec["evaluation"]["episodes"]
            )
        )
        periodic_evaluation = events * monitor_episodes * steps
        if not str(spec["environment"]["name"]).startswith("vmas:"):
            collision_probe = periodic_evaluation
    final_evaluation = (
        int(spec["evaluation"]["episodes"])
        * len(spec["evaluation"]["risk_tiers"])
        * steps
    )
    evaluation = periodic_evaluation + collision_probe + final_evaluation
    return {
        "training": training,
        "periodic_evaluation": periodic_evaluation,
        "collision_probe": collision_probe,
        "final_evaluation": final_evaluation,
        "evaluation": evaluation,
        "total": training + evaluation,
    }


def measured_variant_times(
    result_root: Path,
    expected_variants: set[str],
    timing_field: str = "wall_time_seconds",
) -> Dict[str, float]:
    measured: Dict[str, float] = {}
    for path in result_root.rglob("result.json"):
        result = read_json(path)
        variant = result.get("variant")
        key = str(variant.get("key")) if isinstance(variant, Mapping) else str(variant)
        if key not in expected_variants:
            continue
        resource = result.get("resource_audit", {})
        if timing_field not in resource:
            raise ValueError(
                f"result {path} is missing resource_audit.{timing_field}"
            )
        seconds = float(resource[timing_field])
        if seconds <= 0 or key in measured:
            raise ValueError(f"invalid or duplicate smoke timing for {key!r}")
        measured[key] = seconds
    missing = sorted(expected_variants - set(measured))
    if missing:
        raise ValueError(f"smoke results missing variant timings: {missing}")
    return measured


def build_runtime_plan(
    paper: Mapping[str, object],
    smoke: Mapping[str, object],
    measured_seconds: Mapping[str, float],
    *,
    config_path: str,
    max_chunk_hours: float = 12.0,
    reference_label: str = "smoke",
    timing_field: str = "wall_time_seconds",
) -> Dict[str, object]:
    paper_variants = [str(item["key"]) for item in paper["variants"]]
    smoke_variants = [str(item["key"]) for item in smoke["variants"]]
    if paper_variants != smoke_variants:
        raise ValueError("paper and smoke variants must match in order")
    if set(measured_seconds) != set(paper_variants):
        raise ValueError("measured timings must exactly match registered variants")
    if max_chunk_hours <= 0:
        raise ValueError("max_chunk_hours must be positive")
    smoke_variant_defs = {
        str(item["key"]): item for item in smoke["variants"]
    }
    paper_variant_defs = {
        str(item["key"]): item for item in paper["variants"]
    }
    reference_workloads = {
        key: workload(smoke, smoke_variant_defs[key]) for key in paper_variants
    }
    paper_workloads = {
        key: workload(paper, paper_variant_defs[key]) for key in paper_variants
    }
    scale_components = (
        "training",
        "periodic_evaluation",
        "collision_probe",
        "final_evaluation",
    )
    scales = {}
    for key in paper_variants:
        reference_work = reference_workloads[key]
        target_work = paper_workloads[key]
        component_scales = {}
        for component in scale_components:
            reference_value = reference_work[component]
            target_value = target_work[component]
            if reference_value == 0:
                if target_value != 0:
                    raise ValueError(
                        f"reference workload has no {component} for {key!r}"
                    )
                continue
            component_scales[component] = target_value / reference_value
        scales[key] = {
            "blended": target_work["total"] / reference_work["total"],
            "conservative": max(component_scales.values()),
            "components": component_scales,
        }

    timings = {key: float(measured_seconds[key]) for key in paper_variants}
    fixed_cold_start = 0.0
    algorithms = {str(item.get("algorithm", "")) for item in smoke["variants"]}
    timing_median = median(timings.values())
    slowest = max(timings, key=timings.get)
    # Only homogeneous-algorithm studies can safely treat one extreme first-run
    # timing as shared model/cache cold start. Mixed studies (e.g. HAPPO) retain it.
    if len(algorithms) == 1 and timings[slowest] > 2.5 * timing_median:
        fixed_cold_start = timings[slowest] - timing_median
        timings[slowest] = timing_median

    seed_count = len(paper["seeds"])
    per_variant = {}
    total_low = fixed_cold_start
    total_high = fixed_cold_start
    for key in paper_variants:
        low_per_run = timings[key] * scales[key]["blended"]
        high_per_run = timings[key] * scales[key]["conservative"]
        per_variant[key] = {
            "reference_seconds": float(measured_seconds[key]),
            "recurring_reference_seconds": timings[key],
            "reference_workload_units": reference_workloads[key],
            "paper_workload_units": paper_workloads[key],
            "workload_scales": scales[key],
            "estimated_seconds_per_seed_low": low_per_run,
            "estimated_seconds_per_seed_high": high_per_run,
            "estimated_active_hours_all_seeds_low": low_per_run * seed_count / 3600,
            "estimated_active_hours_all_seeds_high": high_per_run * seed_count / 3600,
        }
        if timing_field == "wall_time_seconds":
            per_variant[key]["smoke_wall_seconds"] = float(measured_seconds[key])
            per_variant[key]["recurring_smoke_seconds"] = timings[key]
            per_variant[key]["estimated_gpu_occupancy_hours_all_seeds_low"] = (
                low_per_run * seed_count / 3600
            )
            per_variant[key]["estimated_gpu_occupancy_hours_all_seeds_high"] = (
                high_per_run * seed_count / 3600
            )
        total_low += low_per_run * seed_count
        total_high += high_per_run * seed_count

    chunks = []
    seeds = [int(seed) for seed in paper["seeds"]]
    first = True
    for key in paper_variants:
        upper_hours_per_seed = per_variant[key]["estimated_seconds_per_seed_high"] / 3600
        seeds_per_chunk = max(1, int(max_chunk_hours // max(upper_hours_per_seed, 1e-12)))
        seeds_per_chunk = min(seeds_per_chunk, len(seeds))
        for start in range(0, len(seeds), seeds_per_chunk):
            selected = seeds[start:start + seeds_per_chunk]
            command = [
                "python", "run_research_study.py", "--config", config_path,
                "--only-variants", key,
                "--only-seeds", ",".join(str(seed) for seed in selected),
            ]
            if not first:
                command.append("--resume")
            chunks.append({
                "index": len(chunks) + 1,
                "variant": key,
                "seeds": selected,
                "estimated_hours_high": upper_hours_per_seed * len(selected),
                "command": command,
            })
            first = False

    return {
        "schema_version": 2,
        "study_id": str(paper["study_id"]),
        "method": (
            "linear environment-step extrapolation from measured "
            f"{reference_label} {timing_field}"
        ),
        "reference_label": str(reference_label),
        "timing_field": str(timing_field),
        "timing_semantics": (
            "process CPU time excludes host suspension but is not GPU device time"
            if timing_field == "process_cpu_time_seconds"
            else "wall time estimates exclusive accelerator occupancy and may include host suspension"
        ),
        "uncertainty": (
            "moderate-to-high; one calibration seed does not capture seed/runtime variance"
            if reference_label == "calibration"
            else "high; run a 100-training-episode calibration before final reservation"
        ),
        "reference_workload_units_by_variant": reference_workloads,
        "paper_workload_units_by_variant": paper_workloads,
        "workload_scales_by_variant": scales,
        "fixed_cold_start_seconds": fixed_cold_start,
        "seed_count": seed_count,
        "run_count": seed_count * len(paper_variants),
        "estimated_active_hours_low": total_low / 3600,
        "estimated_active_hours_high": total_high / 3600,
        "per_variant": per_variant,
        "max_chunk_hours": max_chunk_hours,
        "chunks": chunks,
        "safety_contract": {
            "protocol_reduction_allowed": False,
            "resume_required_after_first_chunk": True,
            "final_summary_written_only_when_all_registered_pairs_exist": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-config", type=Path, required=True)
    parser.add_argument("--smoke-config", type=Path, required=True)
    parser.add_argument("--smoke-results", type=Path, required=True)
    parser.add_argument("--max-chunk-hours", type=float, default=12.0)
    parser.add_argument(
        "--timing-field",
        choices=("wall_time_seconds", "process_cpu_time_seconds"),
        default="wall_time_seconds",
    )
    parser.add_argument(
        "--reference-label", choices=("smoke", "calibration"), default="smoke"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paper = read_json(args.paper_config)
    smoke = read_json(args.smoke_config)
    variants = {str(item["key"]) for item in paper["variants"]}
    timings = measured_variant_times(
        args.smoke_results, variants, timing_field=args.timing_field
    )
    plan = build_runtime_plan(
        paper,
        smoke,
        timings,
        config_path=str(args.paper_config),
        max_chunk_hours=args.max_chunk_hours,
        reference_label=args.reference_label,
        timing_field=args.timing_field,
    )
    rendered = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
