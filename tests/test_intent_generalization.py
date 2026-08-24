from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_generalization import (  # noqa: E402
    intent_behavior_controllability,
    load_generalization_suite,
    objective_profile_prediction_diagnostics,
    representation_retrieval_diagnostics,
    resolve_query_objective_profile,
    validate_generalization_suite,
)
from intent_semantic_encoder import IntentLibrary  # noqa: E402
from intent_objectives import OBJECTIVE_KEYS  # noqa: E402


class IntentGeneralizationTests(unittest.TestCase):
    def test_suite_inheritance_can_exclude_deprecated_queries(self):
        suite = load_generalization_suite(
            ROOT / "configs" / "research" / "uav_intent_generalization_suite.v8.json"
        )
        keys = {str(query["key"]) for query in suite["queries"]}
        self.assertFalse(any(key.startswith("cf_collision_") for key in keys))
        self.assertNotIn("collision", suite["preference_objectives"])
        self.assertEqual(suite["safety_constraints"], ["collision"])

    def test_counterfactual_profile_overrides_one_objective(self):
        query = {
            "canonical_label": "balanced",
            "objective_profile": {"energy": 1.8},
        }
        profile = resolve_query_objective_profile(query)
        self.assertEqual(profile["energy"], 1.8)
        self.assertEqual(profile["task"], 1.0)

    def test_profile_prediction_diagnostics_keep_split_and_objective_audits(self):
        queries = [
            {"key": "a", "split": "seen", "canonical_label": "balanced"},
            {"key": "b", "split": "unseen", "canonical_label": "energy_saving"},
            {"key": "c", "split": "unseen", "canonical_label": "rapid_response"},
        ]
        predictions = np.ones((3, len(OBJECTIVE_KEYS)), dtype=np.float32)
        diagnostic = objective_profile_prediction_diagnostics(queries, predictions)
        self.assertEqual(diagnostic["by_split"]["unseen"]["n_queries"], 2)
        self.assertEqual(len(diagnostic["queries"]), 3)
        self.assertIn("energy", diagnostic["objective_spearman"])

    def test_repository_suite_has_no_label_or_exact_text_leakage(self):
        suite = load_generalization_suite(
            ROOT / "configs" / "research" / "uav_intent_generalization_suite.v1.json"
        )
        self.assertEqual(
            {query["split"] for query in suite["queries"]},
            {"seen", "paraphrase", "unseen"},
        )

    def test_validator_rejects_unseen_training_label(self):
        suite = {
            "schema_version": 1,
            "suite_id": "bad",
            "train_labels": ["balanced"],
            "queries": [
                {
                    "key": "leak",
                    "split": "unseen",
                    "canonical_label": "balanced",
                    "description": "A different balanced wording",
                }
            ],
        }
        with self.assertRaises(ValueError):
            validate_generalization_suite(suite)

    def test_identity_oracle_retrieves_seen_paraphrase_but_not_heldout_label(self):
        full = IntentLibrary.create_onehot()
        train = full.subset_by_labels(["balanced", "safety_first"])
        queries = [
            {
                "key": "paraphrase_balanced",
                "split": "paraphrase",
                "canonical_label": "balanced",
                "posture": "neutral",
                "description": "Balance speed and safety.",
            },
            {
                "key": "unseen_rapid",
                "split": "unseen",
                "canonical_label": "rapid_response",
                "posture": "attack",
                "description": "Respond immediately.",
            },
        ]
        query_vectors = np.stack(
            [full.get_by_label(query["canonical_label"]) for query in queries]
        )
        diagnostics = representation_retrieval_diagnostics(train, query_vectors, queries)
        self.assertEqual(
            diagnostics["by_split"]["paraphrase"]["top1_retrieval_accuracy"], 1.0
        )
        self.assertIsNone(diagnostics["by_split"]["unseen"]["top1_retrieval_accuracy"])
        self.assertIn("objective_retrieval_accuracy", diagnostics["by_split"]["unseen"])
        self.assertGreaterEqual(
            diagnostics["by_split"]["unseen"]["mean_selected_objective_similarity"],
            0.0,
        )

    def test_controllability_detects_aligned_safety_efficiency_tradeoff(self):
        labels = ["safety_first", "balanced", "aggressive_pursuit"]
        queries = [
            {
                "key": label,
                "split": "unseen",
                "canonical_label": label,
                "description": label,
            }
            for label in labels
        ]
        behavior = {}
        for index, label in enumerate(labels):
            behavior[label] = {
                "risk_tiers": {
                    "hard": {
                        "collision_rate": 0.1 * index,
                        "task_completion": 0.4 + 0.2 * index,
                        "energy_remaining": 0.8 - 0.1 * index,
                        "distance_to_target": 3.0 - index,
                        "speed": 0.2 + 0.2 * index,
                        "min_neighbor_distance": 2.0 - 0.5 * index,
                    }
                }
            }
        diagnostic = intent_behavior_controllability(queries, behavior)
        unseen = diagnostic["hard"]["unseen"]
        self.assertAlmostEqual(unseen["safety_tradeoff_spearman"], 1.0)
        self.assertNotIn("collision_preference_spearman", unseen)
        self.assertAlmostEqual(unseen["task_preference_spearman"], 1.0)
        self.assertIn("energy_preference_spearman", unseen)
        self.assertIn("distance_preference_spearman", unseen)
        self.assertIn("time_preference_spearman", unseen)
        self.assertIn("safety_distance_spearman", unseen)

    def test_controllability_rejects_missing_behavior_query(self):
        query = {
            "key": "balanced",
            "split": "seen",
            "canonical_label": "balanced",
            "description": "balanced",
        }
        with self.assertRaises(ValueError):
            intent_behavior_controllability([query], {})


if __name__ == "__main__":
    unittest.main()
