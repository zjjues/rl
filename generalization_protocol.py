"""Frozen protocol checks for the UAV semantic generalization paper study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping


PREFERENCE_OBJECTIVES = ("distance", "energy", "safety", "task", "time", "threat")
EXPECTED_VARIANTS = {
    "objective_grounded_semantic": ("objective_grounded_semantic", "nli_prototype_gated"),
    "pretrained_semantic": ("pretrained_semantic", "none"),
    "legacy_hash": ("legacy_hash", "none"),
    "random_dense_oracle": ("random_dense", "none"),
    "identity_oracle": ("onehot", "none"),
    "no_intent": ("none", "none"),
}
CONFIRMATORY_BASELINES = (
    "pretrained_semantic",
    "legacy_hash",
    "no_intent",
)
DESCRIPTIVE_ORACLES = ("random_dense_oracle", "identity_oracle")
INVARIANT_VARIANT_FIELDS = {
    "algorithm": "imappo",
    "critic_mode": "attention",
    "policy_mode": "direct",
    "rule_prior_context": "neutral",
    "residual_action_scale": 0.25,
    "safety_filter_mode": "pairwise_cbf",
    "cbf_base_min_distance": 1.0,
    "cbf_iterations": 4,
    "use_action_mask": False,
    "disable_intent_reward": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_generalization_paper_protocol(
    spec: Mapping[str, object], suite: Mapping[str, object]
) -> Dict[str, object]:
    """Reject leakage, oracle misuse, weak statistics, and incomplete paper scope."""

    if spec.get("level") != "paper":
        raise ValueError("semantic generalization protocol must have level='paper'")
    seeds = [int(seed) for seed in spec.get("seeds", [])]
    if len(seeds) < 10 or len(seeds) != len(set(seeds)):
        raise ValueError("semantic generalization paper protocol needs at least 10 unique seeds")
    evaluation = spec.get("evaluation", {})
    if int(evaluation.get("episodes", 0)) < 100:
        raise ValueError("semantic generalization paper protocol needs at least 100 episodes")
    if "hard" not in evaluation.get("risk_tiers", {}):
        raise ValueError("semantic generalization paper protocol requires a hard risk tier")
    intent = spec.get("intent", {})
    if int(intent.get("dim", 0)) != 64:
        raise ValueError("all representation controls must use the frozen 64-D policy input")
    for key in ("encoder_revision", "nli_model_revision"):
        if not str(intent.get(key, "")).strip():
            raise ValueError(f"semantic generalization protocol must pin intent.{key}")

    if tuple(suite.get("preference_objectives", ())) != PREFERENCE_OBJECTIVES:
        raise ValueError("suite must expose exactly the six negotiable preference objectives")
    if suite.get("safety_constraints") != ["collision"]:
        raise ValueError("collision must remain the sole non-relaxable safety constraint")
    queries = list(suite.get("queries", []))
    split_counts = {
        split: sum(str(query.get("split")) == split for query in queries)
        for split in ("seen", "paraphrase", "unseen", "counterfactual")
    }
    if split_counts != {"seen": 2, "paraphrase": 4, "unseen": 6, "counterfactual": 18}:
        raise ValueError(f"unexpected frozen suite split counts: {split_counts}")
    expected_behavior_keys = [
        str(query["key"])
        for query in queries
        if str(query["split"]) in {"seen", "paraphrase", "unseen"}
    ]
    observed_behavior_keys = list(map(
        str, spec.get("generalization", {}).get("behavior_query_keys", [])
    ))
    if observed_behavior_keys != expected_behavior_keys:
        raise ValueError(
            "behavior queries must exactly match the frozen seen/paraphrase/unseen order; "
            "counterfactual queries are descriptive-only in this protocol"
        )

    variants = {
        str(variant.get("key")): dict(variant) for variant in spec.get("variants", [])
    }
    if set(variants) != set(EXPECTED_VARIANTS):
        raise ValueError(
            f"generalization variants must be exactly {sorted(EXPECTED_VARIANTS)}"
        )
    for key, (intent_source, decoder) in EXPECTED_VARIANTS.items():
        variant = variants[key]
        if variant.get("intent_source") != intent_source:
            raise ValueError(f"variant {key!r} has the wrong intent_source")
        if variant.get("intent_profile_decoder") != decoder:
            raise ValueError(f"variant {key!r} has the wrong intent_profile_decoder")
        if "intent_dim" in variant:
            raise ValueError(f"variant {key!r} must not override the shared 64-D input")
        for field, expected in INVARIANT_VARIANT_FIELDS.items():
            if variant.get(field) != expected:
                raise ValueError(
                    f"variant {key!r} field {field!r} expected {expected!r}, "
                    f"observed {variant.get(field)!r}"
                )

    reporting = spec.get("reporting", {})
    contract = reporting.get("generalization_contract", {})
    expected_contract = {
        "version": 1,
        "confirmatory_unit": "seed",
        "within_seed_query_aggregation": "arithmetic_mean",
        "primary_tier": "hard",
        "primary_splits": ["paraphrase", "unseen"],
        "treatment": "objective_grounded_semantic",
        "confirmatory_baselines": list(CONFIRMATORY_BASELINES),
        "primary_metrics": ["task_completion", "episode_return"],
        "family_size": 12,
        "paired_test": "exact_two_sided_sign_flip",
        "multiple_testing": "holm",
        "oracle_anchors_descriptive_only": list(DESCRIPTIVE_ORACLES),
        "representation_diagnostics": "descriptive_fixed_query_set",
        "counterfactual_behavior_status": "not_run_in_this_protocol",
    }
    if contract != expected_contract:
        raise ValueError("reporting.generalization_contract differs from the frozen contract")
    if list(reporting.get("primary_metrics", [])) != [
        "task_completion", "episode_return"
    ]:
        raise ValueError("paper primary metrics must be task_completion and episode_return")
    required_resources = {
        "energy_remaining",
        "speed",
        "distance_to_target",
        "min_neighbor_distance",
        "threat_zone_violation",
        "distance_to_threat",
    }
    if not required_resources.issubset(set(reporting.get("artifact_metrics", []))):
        raise ValueError("artifact metrics omit registered semantic-behavior diagnostics")
    return {
        "status": "valid",
        "study_id": str(spec.get("study_id", "")),
        "seed_count": len(seeds),
        "variant_count": len(variants),
        "expected_result_count": len(seeds) * len(variants),
        "suite_id": str(suite.get("suite_id", "")),
        "suite_split_counts": split_counts,
        "behavior_query_count": len(observed_behavior_keys),
        "representation_query_count": len(queries),
        "confirmatory_hypothesis_count": 12,
        "confirmatory_unit": "seed",
        "descriptive_oracle_anchors": list(DESCRIPTIVE_ORACLES),
        "suite_resolved_sha256": _canonical_sha256(suite),
    }


def validate_calibration_compatibility(
    paper: Mapping[str, object], calibration: Mapping[str, object]
) -> None:
    """Ensure calibration exercises the exact representation and query contract."""

    for key in ("environment", "intent", "variants", "generalization", "reporting"):
        if calibration.get(key) != paper.get(key):
            raise ValueError(f"calibration differs from paper protocol field {key!r}")
    if calibration.get("level") != "pilot" or len(calibration.get("seeds", [])) != 1:
        raise ValueError("generalization calibration must be a one-seed pilot")


def build_generalization_protocol_audit(
    repository_root: str | Path,
    paper_config: str | Path,
    calibration_config: str | Path,
) -> Dict[str, object]:
    root = Path(repository_root).resolve()
    paper_path = (root / paper_config).resolve()
    calibration_path = (root / calibration_config).resolve()
    paper = json.loads(paper_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    from intent_generalization import load_generalization_suite

    suite_path = (root / str(paper["generalization"]["suite"])).resolve()
    suite = load_generalization_suite(suite_path)
    report = validate_generalization_paper_protocol(paper, suite)
    validate_calibration_compatibility(paper, calibration)
    report.update({
        "paper_config": paper_path.relative_to(root).as_posix(),
        "paper_config_sha256": _sha256(paper_path),
        "calibration_config": calibration_path.relative_to(root).as_posix(),
        "calibration_config_sha256": _sha256(calibration_path),
        "suite_entrypoint": suite_path.relative_to(root).as_posix(),
        "suite_entrypoint_sha256": _sha256(suite_path),
        "calibration_compatible": True,
    })
    return report

