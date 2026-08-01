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
    return {
        "record_count": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "annotator_count": len(annotator_splits),
        "source_count": len({str(record["source"]) for record in records}),
        "annotator_disjoint": require_annotator_disjoint,
        "text_split_leakage": 0,
    }


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
