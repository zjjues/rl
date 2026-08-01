"""Create an audited paired comparison between variants from separate studies."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment-study", type=Path, required=True)
    parser.add_argument("--treatment-variant", required=True)
    parser.add_argument("--baseline-study", type=Path, required=True)
    parser.add_argument("--baseline-variant", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--level", choices=("smoke", "pilot", "paper"), default="pilot")
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_checksums(study_dir: Path) -> Dict[str, object]:
    checksum_path = study_dir / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"missing checksum manifest: {checksum_path}")
    checked = 0
    failures = []
    for raw_line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, relative = raw_line.split("  ", 1)
        artifact = study_dir / Path(relative)
        observed = hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.is_file() else None
        checked += 1
        if observed != digest:
            failures.append({"path": relative, "expected": digest, "observed": observed})
    if failures:
        raise ValueError(f"source study checksum verification failed: {failures}")
    return {"status": "verified", "files_checked": checked}


def validate_compatibility(
    treatment: Dict[str, object], baseline: Dict[str, object]
) -> Dict[str, object]:
    """Validate fields that make deterministic paired evaluation meaningful."""
    exact_fields = ("seeds", "environment", "evaluation", "generalization")
    mismatches = []
    for field in exact_fields:
        if treatment.get(field) != baseline.get(field):
            mismatches.append(field)
    if mismatches:
        raise ValueError(f"studies are not pair-compatible; mismatched fields: {mismatches}")
    return {
        "status": "compatible",
        "matched_fields": list(exact_fields),
        "seed_count": len(treatment["seeds"]),
        "training_note": (
            "Training configurations may differ by algorithm; pairing is defined by identical "
            "evaluation scenarios, query suite, seeds, and deterministic reset schedule."
        ),
    }


def load_variant_results(
    study_dir: Path, variant: str, seeds: List[int]
) -> List[Dict[str, object]]:
    results = []
    for seed in seeds:
        path = study_dir / variant / f"seed_{seed}" / "result.json"
        result = read_json(path)
        if int(result["seed"]) != int(seed):
            raise ValueError(f"seed mismatch in {path}")
        if str(result["variant"]["key"]) != variant:
            raise ValueError(f"variant mismatch in {path}")
        results.append(result)
    return results


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def result_card(
    study_id: str,
    treatment_name: str,
    baseline_name: str,
    comparison: Dict[str, object],
    seed_count: int,
) -> str:
    lines = [
        f"# Cross-study Result Card: {study_id}",
        "",
        f"- Contrast: `{treatment_name}` minus `{baseline_name}`",
        f"- Paired seeds: `{seed_count}`",
        "- Source checksums: verified before comparison",
        "- Pairing basis: identical seeds, environment, risk tiers, evaluation episodes, and query suite",
        "",
        "## Main-risk-tier paired differences",
        "",
        "| Tier | Metric | Mean difference | 95% bootstrap CI | Treatment win rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for tier, metrics in comparison["risk_tiers"].items():
        for metric, summary in metrics.items():
            ci = summary["difference_ci"]
            lines.append(
                f"| {tier} | {metric} | {summary['mean_difference']:.4f} | "
                f"[{ci['low']:.4f}, {ci['high']:.4f}] | {summary['win_rate']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Differences are treatment minus baseline; lower is better only for collision metrics.",
            "- Five-seed feasibility intervals are diagnostic, not paper-grade confirmatory evidence.",
            "- A confidence interval containing zero does not support a stable directional claim.",
            "- Generalization comparisons average queries within split and seed before bootstrapping.",
            "- This artifact does not alter either checksummed source study.",
            "",
        ]
    )
    return "\n".join(lines)


def resolve_study(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    treatment_dir = resolve_study(args.treatment_study)
    baseline_dir = resolve_study(args.baseline_study)
    treatment_spec = read_json(treatment_dir / "config.json")
    baseline_spec = read_json(baseline_dir / "config.json")
    compatibility = validate_compatibility(treatment_spec, baseline_spec)
    source_audit = {
        "treatment": verify_checksums(treatment_dir),
        "baseline": verify_checksums(baseline_dir),
    }
    seeds = [int(seed) for seed in treatment_spec["seeds"]]
    treatment_results = load_variant_results(treatment_dir, args.treatment_variant, seeds)
    baseline_results = load_variant_results(baseline_dir, args.baseline_variant, seeds)

    from run_research_study import summarize_comparisons

    comparisons = summarize_comparisons(
        {
            args.treatment_variant: treatment_results,
            args.baseline_variant: baseline_results,
        },
        treatment_key=args.treatment_variant,
        bootstrap_seed=args.bootstrap_seed,
    )
    comparison = comparisons[args.baseline_variant]
    output_dir = ROOT / "experiments" / args.level / args.study_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing comparison: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "study_id": args.study_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contrast": {
            "direction": "treatment_minus_baseline",
            "treatment_study": str(treatment_dir),
            "treatment_variant": args.treatment_variant,
            "baseline_study": str(baseline_dir),
            "baseline_variant": args.baseline_variant,
        },
        "compatibility": compatibility,
        "source_audit": source_audit,
        "comparison": comparison,
    }
    write_json(output_dir / "comparison.json", payload)
    (output_dir / "RESULT_CARD.md").write_text(
        result_card(
            args.study_id,
            args.treatment_variant,
            args.baseline_variant,
            comparison,
            len(seeds),
        ),
        encoding="utf-8",
    )
    write_checksums(output_dir)
    print(json.dumps({"output_dir": str(output_dir), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
