"""Machine-auditable gates for an honest paper-submission readiness claim."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence


SUPPORTED_GATE_KINDS = {
    "study_artifact",
    "formal_preference_dataset",
    "json_contract",
}
REQUIRED_GATE_FIELDS = {
    "study_artifact": {"config", "study_dir"},
    "formal_preference_dataset": {"manifest"},
    "json_contract": {"path"},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        candidate = relative.resolve()
    else:
        candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"evidence path escapes repository root: {value}") from exc
    return candidate


def _load_object(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _field(value: Mapping[str, object], dotted: str) -> object:
    current: object = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    return current


def _json_contract(root: Path, gate: Mapping[str, object]) -> Dict[str, object]:
    path = _resolve(root, gate["path"])
    if not path.is_file():
        return {
            "met": False,
            "reasons": [
                f"missing evidence file: {path.relative_to(root).as_posix()}"
            ],
        }
    try:
        payload = _load_object(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"met": False, "reasons": [f"invalid evidence JSON: {exc}"]}
    reasons: list[str] = []
    for dotted in gate.get("required_fields", []):
        try:
            _field(payload, str(dotted))
        except KeyError:
            reasons.append(f"missing required field {dotted!r}")
    for dotted, expected in dict(gate.get("equals", {})).items():
        try:
            observed = _field(payload, str(dotted))
        except KeyError:
            reasons.append(f"missing equality field {dotted!r}")
            continue
        if observed != expected:
            reasons.append(f"field {dotted!r} expected {expected!r}, observed {observed!r}")
    for dotted, choices in dict(gate.get("one_of", {})).items():
        try:
            observed = _field(payload, str(dotted))
        except KeyError:
            reasons.append(f"missing one_of field {dotted!r}")
            continue
        if observed not in list(choices):
            reasons.append(f"field {dotted!r} value {observed!r} not in {list(choices)!r}")
    for contract_key, comparison in (("at_least", "minimum"), ("at_most", "maximum")):
        for dotted, threshold in dict(gate.get(contract_key, {})).items():
            try:
                observed = float(_field(payload, str(dotted)))
                bound = float(threshold)
            except KeyError:
                reasons.append(f"missing numeric field {dotted!r}")
                continue
            except (TypeError, ValueError):
                reasons.append(f"field {dotted!r} is not numeric")
                continue
            failed = observed < bound if contract_key == "at_least" else observed > bound
            if failed:
                reasons.append(
                    f"field {dotted!r} violates {comparison} {bound}: observed {observed}"
                )
    return {
        "met": not reasons,
        "reasons": reasons,
        "evidence_path": path.relative_to(root).as_posix(),
        "evidence_sha256": _sha256(path),
    }


def _formal_preference_dataset(
    root: Path, gate: Mapping[str, object]
) -> Dict[str, object]:
    manifest_path = _resolve(root, gate["manifest"])
    if not manifest_path.is_file():
        return {
            "met": False,
            "reasons": [
                "missing frozen preference manifest: "
                f"{manifest_path.relative_to(root).as_posix()}"
            ],
        }
    try:
        manifest = _load_object(manifest_path)
        records_path = _resolve(root, manifest["records_path"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        return {"met": False, "reasons": [f"invalid preference manifest: {exc}"]}
    reasons: list[str] = []
    if not records_path.is_file():
        reasons.append(
            "missing frozen preference records: "
            f"{records_path.relative_to(root).as_posix()}"
        )
    else:
        observed_sha = _sha256(records_path)
        if observed_sha != manifest.get("records_sha256"):
            reasons.append("preference records SHA-256 differs from frozen manifest")
        else:
            try:
                from preference_dataset import (
                    audit_formal_preference_dataset,
                    load_preference_jsonl,
                )

                audit = audit_formal_preference_dataset(
                    load_preference_jsonl(records_path),
                    min_records_per_class=int(gate.get("min_records_per_class", 50)),
                    min_writers_per_split=int(gate.get("min_writers_per_split", 5)),
                )
                embedded = manifest.get("audit")
                if not isinstance(embedded, Mapping):
                    reasons.append("frozen preference manifest lacks embedded audit")
                elif dict(embedded) != audit:
                    reasons.append("embedded preference audit differs from recomputed audit")
            except (OSError, ValueError, TypeError) as exc:
                reasons.append(f"formal preference audit failed: {exc}")
    for key in ("dataset_id", "consent_version", "validation_code_git_head", "test_access_contract"):
        if not str(manifest.get(key, "")).strip():
            reasons.append(f"preference manifest lacks non-empty {key}")
    return {
        "met": not reasons,
        "reasons": reasons,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "records_path": (
            records_path.relative_to(root).as_posix()
            if records_path.is_relative_to(root) else str(records_path)
        ),
    }


def _study_artifact(root: Path, gate: Mapping[str, object]) -> Dict[str, object]:
    config_path = _resolve(root, gate["config"])
    study_dir = _resolve(root, gate["study_dir"])
    reasons: list[str] = []
    if not config_path.is_file():
        reasons.append(
            "missing registered paper config: "
            f"{config_path.relative_to(root).as_posix()}"
        )
    if not study_dir.is_dir():
        reasons.append(
            "missing study artifact directory: "
            f"{study_dir.relative_to(root).as_posix()}"
        )
    if reasons:
        return {"met": False, "reasons": reasons}
    try:
        config = _load_object(config_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"met": False, "reasons": [f"invalid study config: {exc}"]}
    required_level = str(gate.get("required_level", "paper"))
    if config.get("level") != required_level:
        reasons.append(
            f"study level expected {required_level!r}, observed {config.get('level')!r}"
        )
    from research_artifact import validate_study_artifact

    try:
        report = validate_study_artifact(study_dir, config, verify_checksums=True)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return {
            "met": False,
            "reasons": [f"artifact validator raised an error: {exc}"],
            "config_path": config_path.relative_to(root).as_posix(),
            "study_dir": study_dir.relative_to(root).as_posix(),
            "artifact_status": "invalid",
            "completed_result_count": 0,
            "expected_result_count": (
                len(config.get("seeds", [])) * len(config.get("variants", []))
            ),
            "artifact_error_count": 1,
        }
    if report.get("status") != "valid":
        artifact_errors = list(report.get("errors", []))
        reasons.append(
            f"artifact status is {report.get('status')!r}, not 'valid' "
            f"({len(artifact_errors)} validation errors)"
        )
        reasons.extend(str(item) for item in artifact_errors[:12])
    expected_count = len(config.get("seeds", [])) * len(config.get("variants", []))
    minimum_count = int(gate.get("minimum_result_count", expected_count))
    observed_count = int(report.get("completed_result_count", 0))
    if observed_count < minimum_count:
        reasons.append(
            f"completed result count {observed_count} is below required {minimum_count}"
        )
    return {
        "met": not reasons,
        "reasons": reasons,
        "config_path": config_path.relative_to(root).as_posix(),
        "study_dir": study_dir.relative_to(root).as_posix(),
        "artifact_status": report.get("status"),
        "completed_result_count": observed_count,
        "expected_result_count": expected_count,
        "artifact_error_count": len(report.get("errors", [])),
    }


def _evaluate_gate(root: Path, gate: Mapping[str, object]) -> Dict[str, object]:
    kind = str(gate.get("kind", ""))
    if kind == "study_artifact":
        details = _study_artifact(root, gate)
    elif kind == "formal_preference_dataset":
        details = _formal_preference_dataset(root, gate)
    elif kind == "json_contract":
        details = _json_contract(root, gate)
    else:
        details = {"met": False, "reasons": [f"unsupported gate kind: {kind!r}"]}
    return {
        "key": str(gate.get("key", "")),
        "title": str(gate.get("title", "")),
        "kind": kind,
        "critical": bool(gate.get("critical", True)),
        "status": "met" if details.pop("met") else "unmet",
        **details,
    }


def audit_submission_readiness(
    repository_root: str | Path, spec: Mapping[str, object]
) -> Dict[str, object]:
    """Evaluate every registered evidence gate; readiness requires all critical gates."""

    root = Path(repository_root).resolve()
    if int(spec.get("schema_version", 0)) != 1:
        raise ValueError("submission readiness spec schema_version must be 1")
    gates = spec.get("gates")
    if not isinstance(gates, Sequence) or isinstance(gates, (str, bytes)) or not gates:
        raise ValueError("submission readiness spec requires a non-empty gates list")
    keys: list[str] = []
    for index, gate in enumerate(gates):
        if not isinstance(gate, Mapping):
            raise ValueError(f"gate {index} must be an object")
        key = str(gate.get("key", "")).strip()
        if not key:
            raise ValueError(f"gate {index} requires a non-empty key")
        if str(gate.get("kind", "")) not in SUPPORTED_GATE_KINDS:
            raise ValueError(f"gate {key!r} uses unsupported kind {gate.get('kind')!r}")
        kind = str(gate["kind"])
        missing_fields = sorted(REQUIRED_GATE_FIELDS[kind] - set(gate))
        if missing_fields:
            raise ValueError(f"gate {key!r} missing required fields: {missing_fields}")
        if kind == "json_contract":
            contract_keys = ("required_fields", "equals", "one_of", "at_least", "at_most")
            if not any(gate.get(contract_key) for contract_key in contract_keys):
                raise ValueError(
                    f"gate {key!r} needs semantic JSON conditions, not file existence alone"
                )
            for contract_key in ("equals", "one_of", "at_least", "at_most"):
                value = gate.get(contract_key, {})
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"gate {key!r} field {contract_key!r} must be an object"
                    )
        if kind == "study_artifact" and int(gate.get("minimum_result_count", 1)) <= 0:
            raise ValueError(f"gate {key!r} minimum_result_count must be positive")
        if kind == "formal_preference_dataset" and (
            int(gate.get("min_records_per_class", 50)) <= 0
            or int(gate.get("min_writers_per_split", 5)) <= 0
        ):
            raise ValueError(f"gate {key!r} preference minima must be positive")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("submission readiness gate keys must be unique")
    reports = [_evaluate_gate(root, gate) for gate in gates]
    critical = [report for report in reports if report["critical"]]
    blockers = [report["key"] for report in critical if report["status"] != "met"]
    return {
        "schema_version": 1,
        "audit_id": str(spec.get("audit_id", "submission_readiness")),
        "spec_sha256": hashlib.sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "status": "ready" if not blockers else "not_ready",
        "gate_count": len(reports),
        "critical_gate_count": len(critical),
        "met_gate_count": sum(report["status"] == "met" for report in reports),
        "unmet_critical_gate_count": len(blockers),
        "blocking_gates": blockers,
        "gates": reports,
    }
