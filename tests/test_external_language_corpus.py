import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_language_corpus import (
    import_aerialvln_records,
    import_citynav_records,
    validate_external_corpus_manifest,
    validate_external_corpus_records,
)


def test_import_aerialvln_is_label_free_negative_control():
    payload = [{
        "episode_id": "episode-1",
        "instruction": {"instruction_text": "Take off and fly past the tower."},
    }]
    records = import_aerialvln_records(
        payload, source_split="train", source_version="kaggle-v8"
    )
    assert records[0]["usage"] == "ood_negative_control"
    assert "objective" not in records[0]
    assert validate_external_corpus_records(records)["record_count"] == 1


def test_import_citynav_preserves_descriptions_without_preference_labels():
    payload = [{
        "ann_ids": [17, 18],
        "descriptions": [
            "Locate the building beside the station.",
            "Find the roof across from the park.",
        ],
        "trajectory": [[0.0, 0.0, 10.0]],
    }]
    records = import_citynav_records(
        payload, source_split="val_unseen", source_version="commit-abc"
    )
    assert len(records) == 2
    assert records[0]["source_record_id"].startswith("17:")
    assert records[0]["usage"] == "ood_negative_control"
    assert "objective" not in records[0]


def test_import_citynav_rejects_misaligned_annotation_ids():
    with pytest.raises(ValueError, match="align"):
        import_citynav_records(
            [{"ann_ids": [], "descriptions": ["A valid navigation goal."]}],
            source_split="test_unseen",
            source_version="commit-abc",
        )


def test_external_records_reject_cross_split_text_leakage():
    base = {
        "text": "Fly beside the river and land on the roof.",
        "source": "AirVLN/AerialVLN",
        "source_version": "kaggle-v8",
        "source_record_id": "one",
        "domain": "uav_visual_language_navigation",
        "usage": "ood_negative_control",
    }
    records = [
        {**base, "id": "a", "source_split": "train"},
        {**base, "id": "b", "source_split": "test", "source_record_id": "two"},
    ]
    with pytest.raises(ValueError, match="cross source splits"):
        validate_external_corpus_records(records)


def test_navigation_manifest_cannot_claim_preference_supervision(tmp_path: Path):
    data = tmp_path / "records.jsonl"
    data.write_text(json.dumps({"text": "example"}) + "\n", encoding="utf-8")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset_id": "aerialvln-kaggle-v8",
        "source_url": "https://www.kaggle.com/datasets/shuboliu/aerialvln",
        "source_version": "8",
        "license_name": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "retrieved_at_utc": "2026-08-20T00:00:00Z",
        "usage": "preference_supervision",
        "label_compatibility": "navigation_instruction_not_preference",
        "records_sha256": digest,
        "split_policy": "preserve official source split",
    }
    with pytest.raises(ValueError, match="only be used as an OOD negative control"):
        validate_external_corpus_manifest(manifest, records_path=data)


def test_external_manifest_verifies_frozen_hash(tmp_path: Path):
    data = tmp_path / "records.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset_id": "aerialvln-kaggle-v8",
        "source_url": "https://www.kaggle.com/datasets/shuboliu/aerialvln",
        "source_version": "8",
        "license_name": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "retrieved_at_utc": "2026-08-20T00:00:00Z",
        "usage": "ood_negative_control",
        "label_compatibility": "navigation_instruction_not_preference",
        "records_sha256": digest,
        "split_policy": "preserve official source split",
    }
    audit = validate_external_corpus_manifest(manifest, records_path=data)
    assert audit["hash_verified"] is True
    data.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_external_corpus_manifest(manifest, records_path=data)
