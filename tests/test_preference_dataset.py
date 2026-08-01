from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preference_dataset import validate_preference_records  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
