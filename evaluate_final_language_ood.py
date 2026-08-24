"""Run the preregistered, one-shot CityNav relevance-gate OOD evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from external_language_corpus import (  # noqa: E402
    import_citynav_records,
    normalise_external_text,
    sha256_file,
)
from final_ood_registration import (  # noqa: E402
    load_final_ood_registration,
    validate_final_ood_registration,
)
from objective_semantic_adapter import _load_sentence_transformer  # noqa: E402
from preference_relevance_gate import PreferenceRelevanceGate  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [float(max(0.0, center - half_width)), float(min(1.0, center + half_width))]


def summarize_relevance_scores(
    scores: Sequence[float],
    splits: Sequence[str],
    *,
    threshold: float,
) -> Dict[str, object]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    split_values = np.asarray(list(splits), dtype=object).reshape(-1)
    if values.size == 0 or values.size != split_values.size:
        raise ValueError("scores and splits must be non-empty and aligned")
    if not np.isfinite(values).all():
        raise ValueError("relevance scores must be finite")
    accepted = values >= float(threshold)

    def one(mask: np.ndarray) -> Dict[str, object]:
        selected = values[mask]
        selected_accepted = accepted[mask]
        count = int(selected.size)
        accepted_count = int(selected_accepted.sum())
        return {
            "record_count": count,
            "accepted_count": accepted_count,
            "false_accept_rate_at_frozen_threshold": float(accepted_count / count),
            "wilson_95_percent": wilson_interval(accepted_count, count),
            "score_quantiles": {
                key: float(value)
                for key, value in zip(
                    ("p05", "median", "p95"),
                    np.quantile(selected, (0.05, 0.5, 0.95)),
                )
            },
        }

    result = {"overall": one(np.ones(values.size, dtype=bool)), "by_split": {}}
    for split in sorted(set(map(str, split_values.tolist()))):
        result["by_split"][split] = one(split_values == split)
    return result


def decide(far: float, decision_rule: Mapping[str, object]) -> str:
    if far <= float(decision_rule["pass_if_far_at_most"]):
        return "pass"
    if far <= float(decision_rule["caution_if_far_at_most"]):
        return "caution"
    return "fail"


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempt_path = args.output.with_suffix(args.output.suffix + ".attempt.json")
    if args.output.exists() or attempt_path.exists():
        raise FileExistsError(
            "one-shot final OOD output or attempt marker already exists; refusing rerun"
        )
    registration = load_final_ood_registration(args.registration)
    validation = validate_final_ood_registration(registration, gate_path=args.gate)
    expected_archive_hash = str(registration["source"]["source_archive_sha256"])
    actual_archive_hash = sha256_file(args.source_archive)
    if actual_archive_hash != expected_archive_hash:
        raise ValueError(
            f"CityNav archive hash mismatch: expected {expected_archive_hash}, "
            f"got {actual_archive_hash}"
        )
    registered_paths = list(registration["evaluation_population"]["registered_paths"])
    source_paths = [args.source_root / relative for relative in registered_paths]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"registered CityNav source files are missing: {missing}")
    source_hashes = {
        relative: sha256_file(path)
        for relative, path in zip(registered_paths, source_paths)
    }
    attempt = {
        "schema_version": 1,
        "registration_id": validation["registration_id"],
        "status": "started",
        "started_at_utc": utc_now(),
        "registration_sha256": sha256_file(args.registration),
        "gate_sha256": sha256_file(args.gate),
        "source_archive_sha256": actual_archive_hash,
        "source_file_sha256": source_hashes,
    }
    atomic_json(attempt_path, attempt)

    records = []
    split_counts = Counter()
    for relative, path in zip(registered_paths, source_paths):
        split = path.stem.removeprefix("citynav_")
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        imported = import_citynav_records(
            payload,
            source_split=split,
            source_version=validation["source_revision"],
        )
        records.extend(imported)
        split_counts[split] += len(imported)
        del payload, imported
        gc.collect()

    texts = [str(record["text"]) for record in records]
    splits = [str(record["source_split"]) for record in records]
    normalized_locations = defaultdict(set)
    for text, split in zip(texts, splits):
        normalized_locations[normalise_external_text(text)].add(split)
    cross_split_unique = sum(
        1 for locations in normalized_locations.values() if len(locations) > 1
    )

    gate = PreferenceRelevanceGate.load(args.gate)
    model = _load_sentence_transformer(
        gate.encoder_model, gate.encoder_revision, args.device
    )
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    scores = gate.probabilities(embeddings)
    metrics = summarize_relevance_scores(scores, splits, threshold=gate.threshold)
    outcome = decide(
        float(metrics["overall"]["false_accept_rate_at_frozen_threshold"]),
        registration["decision_rule"],
    )
    completed_at = utc_now()
    result = {
        "schema_version": 1,
        "audit_id": "citynav_human_language_final_ood_v2",
        "evidence_level": "preregistered_one_shot_external_ood",
        "registration_id": validation["registration_id"],
        "registration_sha256": sha256_file(args.registration),
        "started_at_utc": attempt["started_at_utc"],
        "completed_at_utc": completed_at,
        "source": {
            "dataset_id": validation["dataset_id"],
            "source_revision": validation["source_revision"],
            "archive_sha256": actual_archive_hash,
            "archive_bytes": args.source_archive.stat().st_size,
            "source_file_sha256": source_hashes,
            "split_counts": dict(sorted(split_counts.items())),
            "record_count": len(records),
            "normalized_unique_text_count": len(normalized_locations),
            "normalized_texts_present_in_multiple_splits": cross_split_unique,
        },
        "frozen_gate": {
            "sha256": sha256_file(args.gate),
            "threshold": gate.threshold,
            "encoder_model": gate.encoder_model,
            "encoder_revision": gate.encoder_revision,
            "gate_evidence_level": gate.metadata.get("evidence_level"),
        },
        "metrics": metrics,
        "decision": {
            "outcome": outcome,
            "registered_rule": registration["decision_rule"],
        },
        "claim_boundary": registration["claim_boundary"],
        "no_retuning_declaration": (
            "This result was computed with the preregistered gate and threshold. "
            "CityNav may not be used to alter this gate or threshold."
        ),
    }
    atomic_json(args.output, result)
    attempt["status"] = "completed"
    attempt["completed_at_utc"] = completed_at
    attempt["output_sha256"] = sha256_file(args.output)
    atomic_json(attempt_path, attempt)
    print(json.dumps({
        "audit_id": result["audit_id"],
        "record_count": len(records),
        "false_accept_rate": metrics["overall"][
            "false_accept_rate_at_frozen_threshold"
        ],
        "wilson_95_percent": metrics["overall"]["wilson_95_percent"],
        "decision": outcome,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
