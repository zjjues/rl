"""Audit a completed study for intent-to-behavior controllability."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compare_research_studies import verify_checksums
from intent_generalization import intent_behavior_controllability, load_generalization_suite
from research_statistics import paired_difference_summary, summarize_sample


METRICS = (
    "safety_tradeoff_spearman",
    "collision_preference_spearman",
    "task_preference_spearman",
    "collision_rate_range",
    "task_completion_range",
)
ALIGNMENT_METRICS = METRICS[:3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--treatment-variant", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--level", choices=("smoke", "pilot", "paper"), default="pilot")
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(root).as_posix()}"
            )
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_diagnostics(
    diagnostics: List[Dict[str, object]], bootstrap_seed: int
) -> Dict[str, object]:
    summary = {}
    for tier, scopes in diagnostics[0].items():
        tier_summary = {}
        for scope, first_metrics in scopes.items():
            scope_summary = {"n_queries": int(first_metrics["n_queries"])}
            for metric in METRICS:
                values = [item[tier][scope][metric] for item in diagnostics]
                if all(value is not None for value in values):
                    scope_summary[metric] = summarize_sample(values, seed=bootstrap_seed)
                    bootstrap_seed += 2
            tier_summary[scope] = scope_summary
        summary[tier] = tier_summary
    return summary


def compare_diagnostics(
    treatment: List[Dict[str, object]],
    baseline: List[Dict[str, object]],
    bootstrap_seed: int,
) -> Dict[str, object]:
    comparison = {}
    for tier, scopes in treatment[0].items():
        tier_comparison = {}
        for scope in scopes:
            scope_comparison = {}
            for metric in ALIGNMENT_METRICS:
                treatment_values = [item[tier][scope][metric] for item in treatment]
                baseline_values = [item[tier][scope][metric] for item in baseline]
                if all(value is not None for value in treatment_values + baseline_values):
                    scope_comparison[metric] = paired_difference_summary(
                        treatment_values,
                        baseline_values,
                        lower_is_better=False,
                        seed=bootstrap_seed,
                    )
                    bootstrap_seed += 1
            tier_comparison[scope] = scope_comparison
        comparison[tier] = tier_comparison
    return comparison


def result_card(payload: Dict[str, object]) -> str:
    lines = [
        f"# Intent Controllability Card: {payload['study_id']}",
        "",
        "- Evidence status: exploratory post-hoc diagnostic; pre-register before paper use",
        f"- Source checksum audit: `{payload['source_audit']['status']}`",
        f"- Treatment: `{payload['treatment_variant']}`",
        "- Statistical unit: one trained seed; queries are correlated within-seed probes",
        "",
        "## All-query safety-tradeoff alignment",
        "",
        "| Variant | Tier | Mean Spearman | 95% bootstrap CI |",
        "|---|---|---:|---:|",
    ]
    for variant, tiers in payload["variant_summaries"].items():
        for tier, scopes in tiers.items():
            metric = scopes["all"].get("safety_tradeoff_spearman")
            if metric:
                ci = metric["mean_ci"]
                lines.append(
                    f"| {variant} | {tier} | {metric['mean']:.3f} | "
                    f"[{ci['low']:.3f}, {ci['high']:.3f}] |"
                )
    lines.extend(
        [
            "",
            "A positive correlation means intents that request more safety relative to task speed "
            "move behavior toward fewer collisions and/or lower completion. Correlation alone does "
            "not establish superiority; absolute safety and completion remain co-primary outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    source = args.source_study.resolve() if args.source_study.is_absolute() else (ROOT / args.source_study).resolve()
    audit = verify_checksums(source)
    spec = read_json(source / "config.json")
    suite = load_generalization_suite(ROOT / str(spec["generalization"]["suite"]))
    seeds = [int(seed) for seed in spec["seeds"]]
    variants = [str(item["key"]) for item in spec["variants"]]
    if args.treatment_variant not in variants:
        raise ValueError("treatment variant is not present in source study")

    diagnostics_by_variant = {}
    summaries = {}
    for variant in variants:
        diagnostics = []
        for seed in seeds:
            result = read_json(source / variant / f"seed_{seed}" / "result.json")
            diagnostics.append(
                intent_behavior_controllability(
                    suite["queries"], result["intent_generalization"]["behavior"]
                )
            )
        diagnostics_by_variant[variant] = diagnostics
        summaries[variant] = summarize_diagnostics(diagnostics, args.bootstrap_seed)

    comparisons = {}
    treatment = diagnostics_by_variant[args.treatment_variant]
    for variant, baseline in diagnostics_by_variant.items():
        if variant != args.treatment_variant:
            comparisons[variant] = compare_diagnostics(
                treatment, baseline, args.bootstrap_seed + 1000
            )

    payload = {
        "schema_version": 1,
        "study_id": args.study_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "analysis_status": "exploratory_post_hoc",
        "source_study": str(source),
        "source_audit": audit,
        "treatment_variant": args.treatment_variant,
        "seeds": seeds,
        "metric_definition": {
            "safety_preference": "mean(collision,safety)-mean(task,time)",
            "observed_safety_tradeoff": "z(-collision_rate)-z(task_completion)",
            "correlation": "tie-aware Spearman within seed; bootstrap across seeds",
        },
        "variant_summaries": summaries,
        "paired_comparisons": comparisons,
    }
    output = ROOT / "experiments" / args.level / args.study_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite analysis: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "analysis.json", payload)
    (output / "RESULT_CARD.md").write_text(result_card(payload), encoding="utf-8")
    write_checksums(output)
    print(json.dumps({"output_dir": str(output), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
