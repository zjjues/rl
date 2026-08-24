"""Auditable intake for external language corpora used by the paper protocol.

External UAV navigation instructions are useful as out-of-distribution negative
controls, but they are not labels for the project's six preference objectives.
This module keeps that distinction machine-checkable.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


EXTERNAL_CORPUS_USAGES = {
    "ood_negative_control",
    "preference_supervision",
    "preference_blind_test",
}
LABEL_COMPATIBILITY = {
    "direct_preference_labels",
    "navigation_instruction_not_preference",
}


def normalise_external_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_external_corpus_records(
    records: Iterable[Mapping[str, object]],
) -> Dict[str, object]:
    """Validate source-preserving, label-free external-language records."""
    records = list(records)
    if not records:
        raise ValueError("external language corpus is empty")
    required = {
        "id", "text", "source", "source_version", "source_split",
        "source_record_id", "domain", "usage",
    }
    ids = set()
    source_keys = set()
    texts: Dict[str, set[str]] = {}
    split_counts = Counter()
    for index, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"external record {index} missing fields: {missing}")
        record_id = str(record["id"]).strip()
        if not record_id or record_id in ids:
            raise ValueError(f"duplicate or empty external record id: {record_id!r}")
        ids.add(record_id)
        text = normalise_external_text(str(record["text"]))
        if len(text) < 8:
            raise ValueError(f"external record {record_id} text is too short")
        split = str(record["source_split"]).strip()
        source_key = (
            str(record["source"]).strip(),
            str(record["source_version"]).strip(),
            split,
            str(record["source_record_id"]).strip(),
        )
        if not all(source_key) or source_key in source_keys:
            raise ValueError(f"duplicate or incomplete source key for {record_id}")
        source_keys.add(source_key)
        usage = str(record["usage"])
        if usage not in EXTERNAL_CORPUS_USAGES:
            raise ValueError(f"external record {record_id} has invalid usage {usage!r}")
        if not str(record["domain"]).strip():
            raise ValueError(f"external record {record_id} requires a domain")
        texts.setdefault(text, set()).add(split)
        split_counts[split] += 1
    leaked = sorted(text for text, splits in texts.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"normalised external texts cross source splits: {leaked[:3]}")
    return {
        "record_count": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "unique_text_count": len(texts),
        "cross_split_text_leakage": 0,
    }


def validate_external_corpus_manifest(
    manifest: Mapping[str, object],
    *,
    records_path: str | Path | None = None,
) -> Dict[str, object]:
    """Validate provenance, licensing, semantic use, and optional file hash."""
    required = {
        "schema_version", "dataset_id", "source_url", "source_version",
        "license_name", "license_url", "retrieved_at_utc", "usage",
        "label_compatibility", "records_sha256", "split_policy",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"external corpus manifest missing fields: {missing}")
    for key in required - {"schema_version"}:
        if not str(manifest[key]).strip():
            raise ValueError(f"external corpus manifest field {key!r} is empty")
    usage = str(manifest["usage"])
    compatibility = str(manifest["label_compatibility"])
    if usage not in EXTERNAL_CORPUS_USAGES:
        raise ValueError(f"invalid external corpus usage: {usage!r}")
    if compatibility not in LABEL_COMPATIBILITY:
        raise ValueError(f"invalid label compatibility: {compatibility!r}")
    if (
        compatibility == "navigation_instruction_not_preference"
        and usage != "ood_negative_control"
    ):
        raise ValueError(
            "navigation instructions may only be used as an OOD negative control"
        )
    if (
        compatibility == "direct_preference_labels"
        and usage == "ood_negative_control"
    ):
        raise ValueError("direct preference labels need a preference evaluation usage")
    expected_hash = str(manifest["records_sha256"]).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("records_sha256 must be a lowercase SHA-256 digest")
    actual_hash = None
    if records_path is not None:
        actual_hash = sha256_file(records_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"external corpus hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
    return {
        "dataset_id": str(manifest["dataset_id"]),
        "source_version": str(manifest["source_version"]),
        "usage": usage,
        "label_compatibility": compatibility,
        "hash_verified": records_path is not None,
        "actual_sha256": actual_hash,
    }


def _instruction_texts(item: Mapping[str, object]) -> Sequence[str]:
    instruction = item.get("instruction")
    if isinstance(instruction, str):
        return [instruction]
    if isinstance(instruction, Mapping):
        for key in ("instruction_text", "text"):
            if isinstance(instruction.get(key), str):
                return [str(instruction[key])]
    instructions = item.get("instructions")
    if isinstance(instructions, list) and all(isinstance(value, str) for value in instructions):
        return [str(value) for value in instructions]
    return []


def import_aerialvln_records(
    payload: object,
    *,
    source_split: str,
    source_version: str,
    source: str = "AirVLN/AerialVLN",
) -> List[Dict[str, object]]:
    """Convert official AerialVLN episode JSON to label-free OOD records.

    The importer intentionally emits no preference target. Its records can test
    whether a decoder abstains on navigation-only instructions, not whether it
    predicts one of the six negotiable objectives.
    """
    if isinstance(payload, Mapping):
        items = payload.get("episodes")
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError("AerialVLN payload must be a list or contain an episodes list")
    records: List[Dict[str, object]] = []
    for item_index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"AerialVLN item {item_index} is not an object")
        source_id = next(
            (
                str(item[key]) for key in
                ("episode_id", "trajectory_id", "path_id", "id")
                if key in item and str(item[key]).strip()
            ),
            str(item_index),
        )
        texts = _instruction_texts(item)
        for text_index, instruction_text in enumerate(texts):
            records.append({
                "id": f"aerialvln:{source_split}:{source_id}:{text_index}",
                "text": instruction_text,
                "source": source,
                "source_version": source_version,
                "source_split": source_split,
                "source_record_id": f"{source_id}:{text_index}",
                "domain": "uav_visual_language_navigation",
                "usage": "ood_negative_control",
            })
    if not records:
        raise ValueError("AerialVLN payload contains no supported instruction text")
    validate_external_corpus_records(records)
    return records


def import_citynav_records(
    payload: object,
    *,
    source_split: str,
    source_version: str,
    source: str = "water-cookie/CityNav",
) -> List[Dict[str, object]]:
    """Convert one canonical CityNav split to label-free OOD records.

    CityNav stores one or more descriptions and annotation identifiers on each
    trajectory object. The importer preserves every description and never emits
    a preference target.
    """
    if not isinstance(payload, list):
        raise ValueError("CityNav payload must be a list")
    records: List[Dict[str, object]] = []
    for item_index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"CityNav item {item_index} is not an object")
        descriptions = item.get("descriptions")
        if not isinstance(descriptions, list) or not all(
            isinstance(value, str) for value in descriptions
        ):
            raise ValueError(
                f"CityNav item {item_index} descriptions must be a string list"
            )
        annotation_ids = item.get("ann_ids")
        if annotation_ids is not None and (
            not isinstance(annotation_ids, list)
            or len(annotation_ids) != len(descriptions)
        ):
            raise ValueError(
                f"CityNav item {item_index} ann_ids must align with descriptions"
            )
        for description_index, description in enumerate(descriptions):
            annotation_id = (
                str(annotation_ids[description_index])
                if annotation_ids is not None
                else f"{item_index}:{description_index}"
            )
            source_record_id = f"{annotation_id}:{item_index}:{description_index}"
            records.append({
                "id": f"citynav:{source_split}:{source_record_id}",
                "text": description,
                "source": source,
                "source_version": source_version,
                "source_split": source_split,
                "source_record_id": source_record_id,
                "domain": "real_city_uav_language_goal_navigation",
                "usage": "ood_negative_control",
            })
    if not records:
        raise ValueError("CityNav payload contains no descriptions")
    validate_external_corpus_records(records)
    return records


def load_jsonl(path: str | Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"record at {path}:{line_number} is not an object")
        records.append(value)
    return records
