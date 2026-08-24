"""Schema and leakage checks for independently annotated language preferences."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


PREFERENCE_OBJECTIVES = (
    "distance", "energy", "safety", "task", "time", "threat"
)
PREFERENCE_CLASSES = tuple(
    f"{objective}:{polarity}"
    for objective in PREFERENCE_OBJECTIVES
    for polarity in ("low", "high")
) + ("neutral",)
DATASET_SPLITS = ("train", "dev", "test")


def normalise_preference_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def preference_class(record: Mapping[str, object]) -> str:
    objective = str(record["objective"])
    polarity = str(record["polarity"])
    if objective == "neutral" and polarity == "neutral":
        return "neutral"
    return f"{objective}:{polarity}"


def validate_preference_records(
    records: Iterable[Mapping[str, object]],
    *,
    require_all_classes: bool = False,
    require_annotator_disjoint: bool = True,
    require_independent_review: bool = False,
) -> Dict[str, object]:
    records = list(records)
    if not records:
        raise ValueError("preference dataset is empty")
    required = {
        "id", "text", "objective", "polarity", "annotator_id", "source", "split"
    }
    ids = set()
    text_splits: Dict[str, set[str]] = {}
    annotator_splits: Dict[str, set[str]] = {}
    agreement_pairs = []
    class_counts = Counter()
    split_counts = Counter()
    for index, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"record {index} missing fields: {missing}")
        record_id = str(record["id"]).strip()
        if not record_id or record_id in ids:
            raise ValueError(f"duplicate or empty record id: {record_id!r}")
        ids.add(record_id)
        text = normalise_preference_text(str(record["text"]))
        if len(text) < 8:
            raise ValueError(f"record {record_id} text is too short")
        objective = str(record["objective"])
        polarity = str(record["polarity"])
        if objective == "neutral":
            if polarity != "neutral":
                raise ValueError(f"neutral record {record_id} must use neutral polarity")
        elif objective not in PREFERENCE_OBJECTIVES or polarity not in {"low", "high"}:
            raise ValueError(f"invalid objective/polarity for record {record_id}")
        split = str(record["split"])
        if split not in DATASET_SPLITS:
            raise ValueError(f"invalid split for record {record_id}: {split}")
        source = str(record["source"]).strip()
        annotator = str(record["annotator_id"]).strip()
        if not source or not annotator:
            raise ValueError(f"record {record_id} requires source and annotator_id")
        text_splits.setdefault(text, set()).add(split)
        annotator_splits.setdefault(annotator, set()).add(split)
        label = preference_class(record)
        if label not in PREFERENCE_CLASSES:
            raise ValueError(f"invalid class for record {record_id}: {label}")
        class_counts[label] += 1
        split_counts[split] += 1
        if require_independent_review:
            _validate_independent_review(record, record_id, label)
            agreement_pairs.append((
                _class_from_values(
                    record["elicited_objective"], record["elicited_polarity"]
                ),
                _class_from_values(
                    record["reviewer_objective"], record["reviewer_polarity"]
                ),
            ))
    leaked_texts = sorted(text for text, splits in text_splits.items() if len(splits) > 1)
    if leaked_texts:
        raise ValueError(
            f"normalised texts cross dataset splits: {leaked_texts[:3]}"
        )
    if require_annotator_disjoint:
        leaked_annotators = sorted(
            annotator for annotator, splits in annotator_splits.items() if len(splits) > 1
        )
        if leaked_annotators:
            raise ValueError(
                f"annotators cross dataset splits: {leaked_annotators[:5]}"
            )
    if require_all_classes:
        missing_classes = sorted(set(PREFERENCE_CLASSES) - set(class_counts))
        if missing_classes:
            raise ValueError(f"dataset missing preference classes: {missing_classes}")
    result = {
        "record_count": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "annotator_count": len(annotator_splits),
        "source_count": len({str(record["source"]) for record in records}),
        "annotator_disjoint": require_annotator_disjoint,
        "text_split_leakage": 0,
    }
    if agreement_pairs:
        result["independent_review"] = inter_annotator_agreement(agreement_pairs)
    return result


def _class_from_values(objective: object, polarity: object) -> str:
    return "neutral" if str(objective) == "neutral" else f"{objective}:{polarity}"


def _validate_independent_review(
    record: Mapping[str, object], record_id: str, final_label: str
) -> None:
    required = {
        "elicited_objective", "elicited_polarity", "reviewer_id",
        "reviewer_objective", "reviewer_polarity", "decision",
        "collection_batch", "prompt_id", "language", "consent_version",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"record {record_id} missing review fields: {missing}")
    writer = str(record["annotator_id"]).strip()
    reviewer = str(record["reviewer_id"]).strip()
    if not reviewer or reviewer == writer:
        raise ValueError(f"record {record_id} needs an independent reviewer")
    for key in ("collection_batch", "prompt_id", "language", "consent_version"):
        if not str(record[key]).strip():
            raise ValueError(f"record {record_id} has empty provenance field {key}")
    elicited = _class_from_values(
        record["elicited_objective"], record["elicited_polarity"]
    )
    reviewed = _class_from_values(
        record["reviewer_objective"], record["reviewer_polarity"]
    )
    for label in (elicited, reviewed):
        if label not in PREFERENCE_CLASSES:
            raise ValueError(f"record {record_id} has invalid review class {label}")
    decision = str(record["decision"])
    adjudicator = str(record.get("adjudicator_id", "")).strip()
    if decision == "agreed":
        if len({elicited, reviewed, final_label}) != 1:
            raise ValueError(f"agreed record {record_id} has inconsistent labels")
        if adjudicator:
            raise ValueError(f"agreed record {record_id} must not name an adjudicator")
    elif decision == "adjudicated":
        if elicited == reviewed:
            raise ValueError(f"adjudicated record {record_id} has no disagreement")
        if not adjudicator or adjudicator in {writer, reviewer}:
            raise ValueError(f"record {record_id} needs an independent adjudicator")
    else:
        raise ValueError(f"record {record_id} has invalid decision {decision!r}")


def inter_annotator_agreement(pairs: Iterable[tuple[str, str]]) -> Dict[str, object]:
    """Return raw agreement and Cohen's kappa before adjudication."""
    pairs = list(pairs)
    if not pairs:
        raise ValueError("agreement requires at least one label pair")
    invalid = sorted({label for pair in pairs for label in pair} - set(PREFERENCE_CLASSES))
    if invalid:
        raise ValueError(f"agreement contains invalid classes: {invalid}")
    writer_counts = Counter(first for first, _ in pairs)
    reviewer_counts = Counter(second for _, second in pairs)
    observed = sum(first == second for first, second in pairs) / len(pairs)
    expected = sum(
        writer_counts[label] * reviewer_counts[label]
        for label in PREFERENCE_CLASSES
    ) / (len(pairs) ** 2)
    kappa = None if expected >= 1.0 - 1e-12 else (observed - expected) / (1.0 - expected)
    return {
        "n_records": len(pairs),
        "raw_agreement": float(observed),
        "cohen_kappa": None if kappa is None else float(kappa),
        "disagreement_count": sum(first != second for first, second in pairs),
    }


def audit_formal_preference_dataset(
    records: Iterable[Mapping[str, object]],
    *,
    min_records_per_class: int = 50,
    min_writers_per_split: int = 5,
) -> Dict[str, object]:
    """Apply the preregistered paper-scale human-data acceptance contract."""
    records = list(records)
    if min_records_per_class <= 0 or min_writers_per_split <= 0:
        raise ValueError("formal dataset minima must be positive")
    audit = validate_preference_records(
        records,
        require_all_classes=True,
        require_annotator_disjoint=True,
        require_independent_review=True,
    )
    class_counts = Counter(preference_class(record) for record in records)
    underfilled = {
        label: class_counts[label]
        for label in PREFERENCE_CLASSES
        if class_counts[label] < min_records_per_class
    }
    if underfilled:
        raise ValueError(f"preference classes below registered minimum: {underfilled}")
    writers_by_split = {
        split: sorted({
            str(record["annotator_id"])
            for record in records if str(record["split"]) == split
        })
        for split in DATASET_SPLITS
    }
    insufficient_splits = {
        split: len(writers)
        for split, writers in writers_by_split.items()
        if len(writers) < min_writers_per_split
    }
    if insufficient_splits:
        raise ValueError(
            f"dataset splits below registered writer minimum: {insufficient_splits}"
        )
    audit.update({
        "minimum_records_per_class": min_records_per_class,
        "minimum_writers_per_split": min_writers_per_split,
        "writers_by_split": writers_by_split,
        "formal_acceptance": "passed",
    })
    return audit


def load_preference_jsonl(path: Path) -> List[Dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"record at {path}:{line_number} is not an object")
        records.append(record)
    return records
