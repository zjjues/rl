"""Integrity and provenance audits for checksummed research studies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping

import numpy as np


PROTOCOL_KEYS = (
    "schema_version",
    "study_id",
    "level",
    "seeds",
    "bootstrap_seed",
    "environment",
    "training",
    "intent",
    "evaluation",
    "generalization",
    "ablation_contract",
)
PRIMARY_METRICS = ("collision_rate", "task_completion", "episode_return")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, errors: list[str]) -> Dict[str, object] | None:
    if not path.is_file():
        errors.append(f"missing required file: {path.name}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {path.name}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path.name} must contain a JSON object")
        return None
    return value


def _variants(spec: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    return {
        str(item["key"]): dict(item)
        for item in spec.get("variants", [])
        if isinstance(item, Mapping) and "key" in item
    }


def _verify_checksums(study_dir: Path, errors: list[str]) -> int:
    checksum_path = study_dir / "checksums.sha256"
    if not checksum_path.is_file():
        errors.append("missing required file: checksums.sha256")
        return 0
    declared: Dict[str, str] = {}
    for number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            errors.append(f"invalid checksum line {number}")
            continue
        relative = parts[1].strip().replace("\\", "/")
        declared[relative] = parts[0].lower()
    actual = {
        path.relative_to(study_dir).as_posix(): path
        for path in study_dir.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    missing = sorted(set(actual) - set(declared))
    stale = sorted(set(declared) - set(actual))
    if missing:
        errors.append(f"files absent from checksum manifest: {missing}")
    if stale:
        errors.append(f"checksum entries without files: {stale}")
    for relative in sorted(set(actual) & set(declared)):
        observed = _sha256(actual[relative])
        if observed != declared[relative]:
            errors.append(f"checksum mismatch: {relative}")
    return len(declared)


def validate_study_artifact(
    study_dir: str | Path,
    expected_spec: Mapping[str, object] | None = None,
    *,
    verify_checksums: bool = True,
) -> Dict[str, object]:
    """Validate study structure, cached identities, summaries, and provenance."""

    study_dir = Path(study_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    config = _load_json(study_dir / "config.json", errors)
    manifest = _load_json(study_dir / "manifest.json", errors)
    summary = _load_json(study_dir / "summary.json", errors)
    checksum_count = _verify_checksums(study_dir, errors) if verify_checksums else 0
    if config is None:
        return {
            "status": "invalid",
            "study_dir": str(study_dir),
            "errors": errors,
            "warnings": warnings,
        }

    reference = dict(expected_spec) if expected_spec is not None else config
    try:
        from research_protocol import validate_variant_protocol

        protocol_audit = validate_variant_protocol(reference)
    except ValueError as exc:
        protocol_audit = None
        errors.append(f"invalid variant protocol: {exc}")
    contract_audit = None
    if "ablation_contract" in reference:
        try:
            from research_ablation import validate_ablation_contract

            contract_audit = validate_ablation_contract(reference)
        except ValueError as exc:
            errors.append(f"invalid ablation contract: {exc}")
    for key in PROTOCOL_KEYS:
        if config.get(key) != reference.get(key):
            errors.append(f"artifact config differs from expected protocol field {key!r}")
    expected_variants = _variants(reference)
    artifact_variants = _variants(config)
    if artifact_variants != expected_variants:
        errors.append("artifact variant definitions differ from expected config")
    if reference.get("treatment_key") is not None and (
        config.get("treatment_key") != reference.get("treatment_key")
    ):
        errors.append("artifact treatment_key differs from expected config")

    seeds = [int(seed) for seed in reference.get("seeds", [])]
    artifact_level = str(reference.get("level", ""))
    tiers = list(reference.get("evaluation", {}).get("risk_tiers", {}))
    result_values: Dict[str, Dict[str, Dict[str, list[float]]]] = {}
    observed_paths = set()
    for variant_key, variant in expected_variants.items():
        result_values[variant_key] = {
            tier: {metric: [] for metric in PRIMARY_METRICS} for tier in tiers
        }
        for seed in seeds:
            relative = Path(variant_key) / f"seed_{seed}" / "result.json"
            observed_paths.add(relative.as_posix())
            result = _load_json(study_dir / relative, errors)
            if result is None:
                continue
            if int(result.get("seed", -1)) != seed:
                errors.append(f"result seed mismatch: {relative.as_posix()}")
            result_variant = result.get("variant")
            if not isinstance(result_variant, Mapping) or dict(result_variant) != variant:
                errors.append(f"result variant mismatch: {relative.as_posix()}")
            if artifact_level == "paper":
                resource = result.get("resource_audit")
                if not isinstance(resource, Mapping):
                    errors.append(f"paper result lacks resource_audit: {relative.as_posix()}")
                else:
                    required_resource = {
                        "wall_time_seconds",
                        "device",
                        "cuda_peak_allocated_mb",
                        "model_parameters",
                        "frozen_text_model_cache",
                    }
                    missing_resource = sorted(required_resource - set(resource))
                    if missing_resource:
                        errors.append(
                            f"paper resource_audit missing {missing_resource}: "
                            f"{relative.as_posix()}"
                        )
                    elif (
                        not np.isfinite(float(resource["wall_time_seconds"]))
                        or float(resource["wall_time_seconds"]) <= 0.0
                        or not np.isfinite(float(resource["cuda_peak_allocated_mb"]))
                    ):
                        errors.append(
                            f"paper resource_audit is non-finite: {relative.as_posix()}"
                        )
            tier_metrics = result.get("tier_metrics")
            if not isinstance(tier_metrics, Mapping):
                errors.append(f"missing tier_metrics: {relative.as_posix()}")
                continue
            for tier in tiers:
                values = tier_metrics.get(tier)
                if not isinstance(values, Mapping):
                    errors.append(f"missing tier {tier!r}: {relative.as_posix()}")
                    continue
                for metric in PRIMARY_METRICS:
                    key = f"{tier}_{metric}"
                    if key not in values or not np.isfinite(float(values[key])):
                        errors.append(f"missing/non-finite {key}: {relative.as_posix()}")
                    else:
                        result_values[variant_key][tier][metric].append(float(values[key]))
    extra_results = sorted(
        path.relative_to(study_dir).as_posix()
        for path in study_dir.glob("*/seed_*/result.json")
        if path.relative_to(study_dir).as_posix() not in observed_paths
    )
    if extra_results:
        errors.append(f"unexpected per-seed results: {extra_results}")

    summary_variants = summary.get("variants", {}) if summary else {}
    if set(summary_variants) != set(expected_variants):
        errors.append("summary variant keys do not match expected variants")
    for variant_key in set(summary_variants) & set(expected_variants):
        risk_tiers = summary_variants[variant_key].get("risk_tiers", {})
        for tier in tiers:
            for metric in PRIMARY_METRICS:
                try:
                    raw = risk_tiers[tier][metric]["raw"]
                except (KeyError, TypeError):
                    errors.append(
                        f"summary missing {variant_key}/{tier}/{metric}/raw"
                    )
                    continue
                observed = result_values[variant_key][tier][metric]
                if len(raw) != len(seeds) or not np.allclose(raw, observed):
                    errors.append(
                        f"summary raw values differ for {variant_key}/{tier}/{metric}"
                    )
    if reference.get("treatment_key") and len(expected_variants) > 1:
        multiplicity = summary.get("primary_multiplicity") if summary else None
        if not isinstance(multiplicity, Mapping):
            errors.append("summary lacks primary_multiplicity audit")
        elif int(multiplicity.get("family_size", -1)) <= 0:
            errors.append("primary_multiplicity family is empty")
    if contract_audit is not None and summary is not None:
        if summary.get("ablation_contract_audit") != contract_audit:
            errors.append("summary ablation contract audit differs from config")

    if manifest is not None:
        if manifest.get("config") != config:
            errors.append("manifest embedded config differs from artifact config")
        if manifest.get("status") != "complete":
            errors.append("manifest status is not complete")
        records = manifest.get("run_history", [manifest])
        if not isinstance(records, list) or not records:
            errors.append("manifest run_history is empty")
        else:
            covered = set()
            for record in records:
                record_config = record.get("config", {}) if isinstance(record, Mapping) else {}
                covered.update(_variants(record_config))
                command = record.get("command", []) if isinstance(record, Mapping) else []
                if "--config" in command:
                    config_index = command.index("--config") + 1
                    if config_index < len(command) and not Path(command[config_index]).is_file():
                        warnings.append(
                            f"recorded command references absent config: {command[config_index]}"
                        )
                if record.get("git_status_short"):
                    warnings.append("study invocation used a dirty Git worktree")
            if covered != set(expected_variants):
                errors.append("manifest run history does not cover all expected variants")

    level = artifact_level
    eval_episodes = int(reference.get("evaluation", {}).get("episodes", 0))
    if level == "paper":
        if len(seeds) < 10 or eval_episodes < 100:
            errors.append("paper artifact violates minimum seed/evaluation protocol")
        if any("dirty Git worktree" in warning for warning in warnings):
            errors.append("paper artifact was produced from a dirty Git worktree")
    return {
        "status": "valid" if not errors else "invalid",
        "study_id": reference.get("study_id"),
        "level": level,
        "study_dir": str(study_dir),
        "variant_count": len(expected_variants),
        "seed_count": len(seeds),
        "expected_result_count": len(expected_variants) * len(seeds),
        "checksum_entry_count": checksum_count,
        "variant_protocol_audit": protocol_audit,
        "errors": errors,
        "warnings": sorted(set(warnings)),
    }
