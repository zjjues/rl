from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preference_dataset import (  # noqa: E402
    PREFERENCE_CLASSES,
    audit_formal_preference_dataset,
    inter_annotator_agreement,
    validate_preference_records,
)


def record(record_id, text, objective, polarity, annotator, split):
    return {
        "id": record_id,
        "text": text,
        "objective": objective,
        "polarity": polarity,
        "annotator_id": annotator,
        "source": "human-elicitation-v1",
        "split": split,
    }


class PreferenceDatasetTests(unittest.TestCase):
    def test_valid_records_produce_audit(self):
        audit = validate_preference_records([
            record("a", "Conserve the battery during flight.", "energy", "high", "p1", "train"),
            record("b", "Power use does not matter today.", "energy", "low", "p2", "dev"),
            record("c", "Keep the ordinary mission balance.", "neutral", "neutral", "p3", "test"),
        ])
        self.assertEqual(audit["record_count"], 3)
        self.assertEqual(audit["annotator_count"], 3)

    def test_rejects_text_and_annotator_leakage(self):
        with self.assertRaisesRegex(ValueError, "texts cross"):
            validate_preference_records([
                record("a", "Conserve battery power.", "energy", "high", "p1", "train"),
                record("b", "  conserve BATTERY power. ", "energy", "high", "p2", "test"),
            ])
        with self.assertRaisesRegex(ValueError, "annotators cross"):
            validate_preference_records([
                record("a", "Conserve battery power.", "energy", "high", "p1", "train"),
                record("b", "Save electrical reserves.", "energy", "high", "p1", "test"),
            ])

    def test_collision_is_not_a_language_preference_class(self):
        with self.assertRaisesRegex(ValueError, "invalid objective"):
            validate_preference_records([
                record("a", "Allow aircraft contact.", "collision", "low", "p1", "train")
            ])

    def test_formal_records_require_independent_review_and_report_agreement(self):
        reviewed = record(
            "a", "Conserve the battery during flight.",
            "energy", "high", "writer-1", "train"
        )
        reviewed.update({
            "elicited_objective": "energy",
            "elicited_polarity": "high",
            "reviewer_id": "reviewer-1",
            "reviewer_objective": "energy",
            "reviewer_polarity": "high",
            "decision": "agreed",
            "collection_batch": "batch-1",
            "prompt_id": "prompt-1",
            "language": "en",
            "consent_version": "v1",
        })
        audit = validate_preference_records(
            [reviewed], require_independent_review=True
        )
        self.assertEqual(audit["independent_review"]["raw_agreement"], 1.0)

    def test_adjudication_must_be_independent(self):
        reviewed = record(
            "a", "Battery reserves are optional today.",
            "energy", "low", "writer-1", "train"
        )
        reviewed.update({
            "elicited_objective": "energy",
            "elicited_polarity": "low",
            "reviewer_id": "reviewer-1",
            "reviewer_objective": "time",
            "reviewer_polarity": "high",
            "decision": "adjudicated",
            "adjudicator_id": "writer-1",
            "collection_batch": "batch-1",
            "prompt_id": "prompt-1",
            "language": "en",
            "consent_version": "v1",
        })
        with self.assertRaisesRegex(ValueError, "independent adjudicator"):
            validate_preference_records([reviewed], require_independent_review=True)

    def test_agreement_uses_pre_adjudication_labels(self):
        audit = inter_annotator_agreement([
            ("energy:high", "energy:high"),
            ("time:low", "time:high"),
        ])
        self.assertEqual(audit["raw_agreement"], 0.5)
        self.assertEqual(audit["disagreement_count"], 1)

    def test_formal_dataset_contract_checks_class_and_writer_minima(self):
        records = []
        splits = ("train", "dev", "test")
        for index, label in enumerate(PREFERENCE_CLASSES):
            split = splits[index % len(splits)]
            objective, polarity = (
                ("neutral", "neutral") if label == "neutral"
                else tuple(label.split(":"))
            )
            item = record(
                f"r-{index}", f"Independent operator wording number {index}.",
                objective, polarity, f"writer-{split}-{index}", split,
            )
            item.update({
                "elicited_objective": objective,
                "elicited_polarity": polarity,
                "reviewer_id": f"reviewer-{index}",
                "reviewer_objective": objective,
                "reviewer_polarity": polarity,
                "decision": "agreed",
                "collection_batch": "batch-1",
                "prompt_id": f"prompt-{index}",
                "language": "en",
                "consent_version": "v1",
            })
            records.append(item)
        audit = audit_formal_preference_dataset(
            records, min_records_per_class=1, min_writers_per_split=1
        )
        self.assertEqual(audit["formal_acceptance"], "passed")
        with self.assertRaisesRegex(ValueError, "below registered minimum"):
            audit_formal_preference_dataset(
                records, min_records_per_class=2, min_writers_per_split=1
            )


if __name__ == "__main__":
    unittest.main()
