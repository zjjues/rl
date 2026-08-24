"""Confirmatory seed-level statistics for the frozen semantic generalization study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from generalization_protocol import (
    CONFIRMATORY_BASELINES,
    DESCRIPTIVE_ORACLES,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_split_value(
    result: Mapping[str, object], split: str, tier: str, metric: str
) -> float:
    behavior = result["intent_generalization"]["behavior"]
    values = [
        float(item["risk_tiers"][tier][metric])
        for item in behavior.values()
        if str(item["split"]) == split
    ]
    if not values:
        raise ValueError(f"result has no behavior queries for split {split!r}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"result has non-finite {metric!r} values for split {split!r}")
    return float(np.mean(values))


def compute_generalization_statistics(
    results_by_variant: Mapping[str, Mapping[int, Mapping[str, object]]],
    spec: Mapping[str, object],
) -> Dict[str, object]:
    """Aggregate queries within seed, then test paired seeds without pseudoreplication."""

    from research_statistics import (
        holm_adjust,
        paired_difference_summary,
        summarize_sample,
    )

    contract = spec["reporting"]["generalization_contract"]
    treatment_key = str(contract["treatment"])
    baselines = tuple(map(str, contract["confirmatory_baselines"]))
    if baselines != CONFIRMATORY_BASELINES:
        raise ValueError("confirmatory baselines differ from the frozen protocol")
    splits = tuple(map(str, contract["primary_splits"]))
    metrics = tuple(map(str, contract["primary_metrics"]))
    tier = str(contract["primary_tier"])
    seeds = tuple(map(int, spec["seeds"]))
    expected_variants = {
        treatment_key, *baselines, *DESCRIPTIVE_ORACLES, "no_intent"
    }
    if set(results_by_variant) != expected_variants:
        raise ValueError("result variants differ from the frozen generalization protocol")
    for variant, by_seed in results_by_variant.items():
        if set(map(int, by_seed)) != set(seeds):
            raise ValueError(f"variant {variant!r} does not contain the exact registered seeds")

    hypotheses: Dict[str, Dict[str, object]] = {}
    p_values: Dict[str, float] = {}
    bootstrap_seed = int(spec["bootstrap_seed"])
    for split in splits:
        for baseline in baselines:
            for metric in metrics:
                treatment_values = [
                    _seed_split_value(results_by_variant[treatment_key][seed], split, tier, metric)
                    for seed in seeds
                ]
                baseline_values = [
                    _seed_split_value(results_by_variant[baseline][seed], split, tier, metric)
                    for seed in seeds
                ]
                key = f"{split}:{treatment_key}_vs_{baseline}:{metric}"
                paired = paired_difference_summary(
                    treatment_values,
                    baseline_values,
                    lower_is_better=False,
                    seed=bootstrap_seed,
                )
                bootstrap_seed += 2
                hypotheses[key] = {
                    "split": split,
                    "tier": tier,
                    "metric": metric,
                    "treatment": treatment_key,
                    "baseline": baseline,
                    "n_paired_seeds": len(seeds),
                    "within_seed_query_aggregation": "arithmetic_mean",
                    "treatment_by_seed": treatment_values,
                    "baseline_by_seed": baseline_values,
                    "paired": paired,
                }
                p_values[key] = float(paired["randomization_test"]["p_value"])
    if len(hypotheses) != int(contract["family_size"]):
        raise ValueError("computed hypothesis family size differs from the frozen contract")
    holm = holm_adjust(p_values, alpha=0.05)
    for key, hypothesis in hypotheses.items():
        hypothesis["holm_adjusted_p_value"] = holm["adjusted_p_values"][key]
        hypothesis["holm_reject_0_05"] = holm["reject"][key]

    descriptive_oracles: Dict[str, object] = {}
    for oracle in DESCRIPTIVE_ORACLES:
        split_summary = {}
        for split in splits:
            split_summary[split] = {}
            for metric in metrics:
                values = [
                    _seed_split_value(results_by_variant[oracle][seed], split, tier, metric)
                    for seed in seeds
                ]
                split_summary[split][metric] = summarize_sample(
                    values, seed=bootstrap_seed
                )
                bootstrap_seed += 2
        descriptive_oracles[oracle] = split_summary
    return {
        "schema_version": 1,
        "status": "valid",
        "study_id": str(spec["study_id"]),
        "confirmatory_unit": "seed",
        "query_pseudoreplication": False,
        "seed_count": len(seeds),
        "family_size": len(hypotheses),
        "multiple_testing": holm,
        "hypotheses": hypotheses,
        "descriptive_oracles": descriptive_oracles,
        "claim_boundary": (
            "Random-dense and one-hot variants receive canonical-label identity and are "
            "descriptive oracle anchors, not evidence of text understanding. Fixed-query "
            "representation diagnostics are descriptive; confirmatory uncertainty is over seeds."
        ),
    }


def load_complete_generalization_results(
    study_dir: str | Path, spec: Mapping[str, object]
) -> Dict[str, Dict[int, Dict[str, object]]]:
    study_dir = Path(study_dir).resolve()
    loaded: Dict[str, Dict[int, Dict[str, object]]] = {}
    for variant in spec["variants"]:
        key = str(variant["key"])
        loaded[key] = {}
        for seed in map(int, spec["seeds"]):
            path = study_dir / key / f"seed_{seed}" / "result.json"
            if not path.is_file():
                raise ValueError(f"missing registered result: {path}")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"result is not a JSON object: {path}")
            loaded[key][seed] = value
    return loaded


def build_generalization_statistics_report(
    repository_root: str | Path,
    study_dir: str | Path,
    config_path: str | Path,
) -> Dict[str, object]:
    root = Path(repository_root).resolve()
    resolved_study = (root / study_dir).resolve()
    resolved_config = (root / config_path).resolve()
    spec = json.loads(resolved_config.read_text(encoding="utf-8"))
    from research_artifact import validate_study_artifact

    artifact = validate_study_artifact(resolved_study, spec, verify_checksums=True)
    if artifact["status"] != "valid":
        raise ValueError(
            f"generalization statistics require a valid complete artifact: "
            f"{len(artifact.get('errors', []))} errors"
        )
    report = compute_generalization_statistics(
        load_complete_generalization_results(resolved_study, spec), spec
    )
    report.update({
        "study_dir": resolved_study.relative_to(root).as_posix(),
        "config_path": resolved_config.relative_to(root).as_posix(),
        "config_sha256": _sha256(resolved_config),
        "artifact_status": artifact["status"],
        "artifact_checksum_count": artifact["checksum_count"],
    })
    return report

