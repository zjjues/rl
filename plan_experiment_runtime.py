"""Estimate paper-study runtime and emit safe resumable run chunks.

The estimate extrapolates measured smoke wall time by environment-step workload.
It is a planning bound, not a performance claim; a 100-episode calibration run is
still required before reserving the final GPU window.
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


def workload(spec: Mapping[str, object]) -> Dict[str, int]:
    steps = int(spec["training"]["steps"])
    training = int(spec["training"]["episodes"]) * steps
    evaluation = (
        int(spec["evaluation"]["episodes"])
        * len(spec["evaluation"]["risk_tiers"])
        * steps
    )
    return {"training": training, "evaluation": evaluation, "total": training + evaluation}


def measured_variant_times(
    result_root: Path, expected_variants: set[str]
) -> Dict[str, float]:
    measured: Dict[str, float] = {}
    for path in result_root.rglob("result.json"):
        result = read_json(path)
        variant = result.get("variant")
        key = str(variant.get("key")) if isinstance(variant, Mapping) else str(variant)
        if key not in expected_variants:
            continue
        seconds = float(result["resource_audit"]["wall_time_seconds"])
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
) -> Dict[str, object]:
    paper_variants = [str(item["key"]) for item in paper["variants"]]
    smoke_variants = [str(item["key"]) for item in smoke["variants"]]
    if paper_variants != smoke_variants:
        raise ValueError("paper and smoke variants must match in order")
    if set(measured_seconds) != set(paper_variants):
        raise ValueError("measured timings must exactly match registered variants")
    if max_chunk_hours <= 0:
        raise ValueError("max_chunk_hours must be positive")
    smoke_work = workload(smoke)
    paper_work = workload(paper)
    blended_scale = paper_work["total"] / smoke_work["total"]
    training_scale = paper_work["training"] / smoke_work["training"]
    evaluation_scale = paper_work["evaluation"] / smoke_work["evaluation"]
    upper_scale = max(training_scale, evaluation_scale)

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
        low_per_run = timings[key] * blended_scale
        high_per_run = timings[key] * upper_scale
        per_variant[key] = {
            "smoke_wall_seconds": float(measured_seconds[key]),
            "recurring_smoke_seconds": timings[key],
            "estimated_seconds_per_seed_low": low_per_run,
            "estimated_seconds_per_seed_high": high_per_run,
            "estimated_gpu_hours_all_seeds_low": low_per_run * seed_count / 3600,
            "estimated_gpu_hours_all_seeds_high": high_per_run * seed_count / 3600,
        }
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
        "schema_version": 1,
        "study_id": str(paper["study_id"]),
        "method": "linear environment-step extrapolation from measured smoke wall time",
        "uncertainty": "high; run a 100-training-episode calibration before final reservation",
        "smoke_workload_units": smoke_work,
        "paper_workload_units": paper_work,
        "blended_scale": blended_scale,
        "conservative_scale": upper_scale,
        "fixed_cold_start_seconds": fixed_cold_start,
        "seed_count": seed_count,
        "run_count": seed_count * len(paper_variants),
        "estimated_gpu_hours_low": total_low / 3600,
        "estimated_gpu_hours_high": total_high / 3600,
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paper = read_json(args.paper_config)
    smoke = read_json(args.smoke_config)
    variants = {str(item["key"]) for item in paper["variants"]}
    timings = measured_variant_times(args.smoke_results, variants)
    plan = build_runtime_plan(
        paper,
        smoke,
        timings,
        config_path=str(args.paper_config),
        max_chunk_hours=args.max_chunk_hours,
    )
    rendered = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
