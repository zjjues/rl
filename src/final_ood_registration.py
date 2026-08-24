"""Machine-checkable preregistration for one-shot external OOD evaluation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Mapping

from external_language_corpus import sha256_file
from preference_relevance_gate import PreferenceRelevanceGate


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EXPECTED_PRIMARY_METRIC = "false_accept_rate_at_frozen_threshold"


def load_final_ood_registration(path: str | Path) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("final OOD registration must be a JSON object")
    return payload


def validate_final_ood_registration(
    registration: Mapping[str, object],
    *,
    gate_path: str | Path | None = None,
) -> Dict[str, object]:
    """Reject registrations that do not freeze source, gate, and decision rule."""
    required = {
        "schema_version", "registration_id", "registered_at_utc", "status",
        "source", "evaluation_population", "frozen_gate", "metrics",
        "decision_rule", "blinding_and_change_control", "claim_boundary",
    }
    missing = sorted(required - set(registration))
    if missing:
        raise ValueError(f"final OOD registration missing fields: {missing}")
    if int(registration["schema_version"]) != 1:
        raise ValueError("unsupported final OOD registration schema")
    if registration["status"] != "frozen_before_text_access":
        raise ValueError("final OOD registration must be frozen before text access")

    source = registration["source"]
    if not isinstance(source, Mapping):
        raise ValueError("registration source must be an object")
    for key in (
        "dataset_id", "source_url", "source_revision", "download_url",
        "license_name", "license_url", "language_provenance",
    ):
        if not str(source.get(key, "")).strip():
            raise ValueError(f"registration source field {key!r} is empty")
    revision = str(source["source_revision"]).lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("source_revision must be a full Git commit hash")

    population = registration["evaluation_population"]
    if not isinstance(population, Mapping):
        raise ValueError("evaluation_population must be an object")
    if population.get("sampling") != "all_released_records":
        raise ValueError("final OOD evaluation must use all released records")
    registered_paths = population.get("registered_paths")
    registered_glob = str(population.get("registered_path_glob", "")).strip()
    if registered_paths is not None:
        if (
            not isinstance(registered_paths, list)
            or not registered_paths
            or any(not str(value).strip() for value in registered_paths)
            or len(set(map(str, registered_paths))) != len(registered_paths)
        ):
            raise ValueError("evaluation population registered_paths are invalid")
    elif not registered_glob:
        raise ValueError("evaluation population requires registered paths or a glob")

    frozen_gate = registration["frozen_gate"]
    if not isinstance(frozen_gate, Mapping):
        raise ValueError("frozen_gate must be an object")
    gate_hash = str(frozen_gate.get("sha256", "")).lower()
    if not SHA256_PATTERN.fullmatch(gate_hash):
        raise ValueError("frozen gate SHA-256 is invalid")
    threshold = float(frozen_gate.get("threshold", float("nan")))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("frozen gate threshold must be finite and in [0, 1]")

    metrics = registration["metrics"]
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be an object")
    if metrics.get("primary") != EXPECTED_PRIMARY_METRIC:
        raise ValueError("unexpected final OOD primary metric")
    if metrics.get("confidence_interval") != "wilson_95_percent":
        raise ValueError("final OOD FAR requires a registered Wilson 95% interval")

    decision = registration["decision_rule"]
    if not isinstance(decision, Mapping):
        raise ValueError("decision_rule must be an object")
    pass_at = float(decision.get("pass_if_far_at_most", float("nan")))
    caution_at = float(decision.get("caution_if_far_at_most", float("nan")))
    if not (0.0 <= pass_at <= caution_at < 1.0):
        raise ValueError("invalid pass/caution FAR boundaries")

    control = registration["blinding_and_change_control"]
    if not isinstance(control, Mapping):
        raise ValueError("blinding_and_change_control must be an object")
    required_controls = {
        "text_inspected_before_registration": False,
        "gate_or_threshold_changes_after_evaluation_allowed": False,
        "evaluation_is_one_shot": True,
    }
    for key, expected in required_controls.items():
        if control.get(key) is not expected:
            raise ValueError(f"invalid change-control value for {key}")

    gate_verified = False
    if gate_path is not None:
        actual_hash = sha256_file(gate_path)
        if actual_hash != gate_hash:
            raise ValueError(
                f"frozen gate hash mismatch: expected {gate_hash}, got {actual_hash}"
            )
        gate = PreferenceRelevanceGate.load(gate_path)
        if float(gate.threshold) != threshold:
            raise ValueError("frozen gate threshold does not match gate artifact")
        if gate.encoder_model != str(frozen_gate.get("encoder_model", "")):
            raise ValueError("frozen gate encoder model mismatch")
        if gate.encoder_revision != str(frozen_gate.get("encoder_revision", "")):
            raise ValueError("frozen gate encoder revision mismatch")
        gate_verified = True

    return {
        "registration_id": str(registration["registration_id"]),
        "dataset_id": str(source["dataset_id"]),
        "source_revision": revision,
        "gate_verified": gate_verified,
        "primary_metric": EXPECTED_PRIMARY_METRIC,
        "pass_if_far_at_most": pass_at,
        "caution_if_far_at_most": caution_at,
    }
