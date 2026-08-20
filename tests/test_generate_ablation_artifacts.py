from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_ablation_artifacts import build_comparison_rows  # noqa: E402


class GenerateAblationArtifactsTests(unittest.TestCase):
    def test_comparison_rows_preserve_registered_orientation(self):
        record = {
            "mean_difference": 0.2,
            "difference_ci": {"low": 0.1, "high": 0.3},
            "standardized_effect_dz": 1.2,
            "randomization_test": {"p_value": 0.03125},
            "holm_adjusted_p_value": 0.0625,
            "holm_reject_0_05": False,
        }
        summary = {
            "paired_comparisons": {
                "no_intent": {
                    "reference_key": "identity_oracle",
                    "factor": "intent_channel",
                    "changed_fields": ["intent_source"],
                    "primary_tiers": ["hard"],
                    "primary_metrics": ["collision_rate"],
                    "risk_tiers": {"hard": {"collision_rate": record}},
                }
            }
        }
        rows = build_comparison_rows(summary)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reference"], "identity_oracle")
        self.assertEqual(rows[0]["variant"], "no_intent")
        self.assertEqual(rows[0]["direction"], "variant_minus_reference")
        self.assertEqual(rows[0]["mean_difference"], 0.2)


if __name__ == "__main__":
    unittest.main()
