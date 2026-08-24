"""Diagnostic OOD false-activation audit for external UAV navigation language."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from external_language_corpus import (  # noqa: E402
    load_jsonl,
    sha256_file,
    validate_external_corpus_records,
)
from intent_objectives import OBJECTIVE_KEYS  # noqa: E402
from intent_semantic_encoder import DEFAULT_INTENT_DESCRIPTIONS  # noqa: E402
from objective_semantic_adapter import ObjectiveSemanticAdapter  # noqa: E402


def summarize_ood_profiles(
    profiles: np.ndarray,
    *,
    thresholds: Sequence[float] = (0.05, 0.10, 0.20),
) -> Dict[str, object]:
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim != 2 or profiles.shape[1] != len(OBJECTIVE_KEYS):
        raise ValueError("OOD profiles must have one column per preference objective")
    if not np.isfinite(profiles).all():
        raise ValueError("OOD profiles contain non-finite values")
    deviations = np.abs(profiles - 1.0)
    maximum = deviations.max(axis=1)
    winners = deviations.argmax(axis=1)
    labels = [
        f"{OBJECTIVE_KEYS[index]}:{'high' if profiles[row, index] >= 1.0 else 'low'}"
        for row, index in enumerate(winners)
    ]
    return {
        "record_count": int(profiles.shape[0]),
        "preference_objectives": list(OBJECTIVE_KEYS),
        "collision_decoded": False,
        "max_abs_deviation": {
            "mean": float(maximum.mean()),
            "median": float(np.median(maximum)),
            "p95": float(np.quantile(maximum, 0.95)),
            "max": float(maximum.max()),
        },
        "activation_rate_by_uncalibrated_threshold": {
            str(float(threshold)): float(np.mean(maximum >= float(threshold)))
            for threshold in thresholds
        },
        "largest_deviation_class_counts": dict(sorted(Counter(labels).items())),
        "interpretation": (
            "Diagnostic false-activation proxy only; no threshold is accepted until "
            "independent preference dev data are frozen and calibrated."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=128)
    parser.add_argument("--sample-seed", type=int, default=20260820)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preference-relevance-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_jsonl(args.records)
    corpus_audit = validate_external_corpus_records(records)
    if args.max_records <= 0:
        raise ValueError("max-records must be positive")
    rng = np.random.default_rng(args.sample_seed)
    indices = np.sort(rng.choice(
        len(records), size=min(args.max_records, len(records)), replace=False
    ))
    sample = [records[int(index)] for index in indices]
    sample_ids = "\n".join(str(record["id"]) for record in sample).encode("utf-8")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    intent = config["intent"]
    adapter = ObjectiveSemanticAdapter.fit(
        DEFAULT_INTENT_DESCRIPTIONS,
        intent_dim=int(intent["dim"]),
        model_name=str(intent["encoder_model"]),
        model_revision=str(intent["encoder_revision"]),
        projection_seed=int(intent["projection_seed"]),
        ridge=float(intent.get("adapter_ridge", 0.01)),
        semantic_weight=float(intent.get("semantic_weight", 0.5)),
        objective_weight=float(intent.get("objective_weight", 1.0)),
        device=args.device,
        profile_decoder="nli_prototype_gated",
        nli_model_name=str(intent["nli_model"]),
        nli_model_revision=str(intent["nli_model_revision"]),
        nli_batch_size=int(intent.get("nli_batch_size", 32)),
        preference_relevance_gate_path=(
            str(args.preference_relevance_gate)
            if args.preference_relevance_gate else ""
        ),
    )
    entries = [(str(record["id"]), str(record["text"])) for record in sample]
    profiles = adapter.predict_profiles(entries)
    payload = {
        "schema_version": 1,
        "audit_id": "aerialvln_v8_val_unseen_ood_smoke",
        "evidence_level": "smoke",
        "source_usage": "ood_negative_control",
        "source_archive_sha256": sha256_file(args.source_archive),
        "records_sha256": sha256_file(args.records),
        "source_record_count": corpus_audit["record_count"],
        "sample_count": len(sample),
        "sample_seed": args.sample_seed,
        "sample_ids_sha256": hashlib.sha256(sample_ids).hexdigest(),
        "preference_relevance_gate_sha256": (
            sha256_file(args.preference_relevance_gate)
            if args.preference_relevance_gate else None
        ),
        "model": adapter.metadata([label for label, _ in DEFAULT_INTENT_DESCRIPTIONS]),
        "result": summarize_ood_profiles(profiles),
        "claim_boundary": (
            "Navigation instructions have no six-axis preference ground truth. This "
            "audit cannot estimate preference accuracy or select an operating threshold."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
