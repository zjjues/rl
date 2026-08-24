from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from generalization_statistics import compute_generalization_statistics  # noqa: E402


class GeneralizationStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((
            ROOT / "configs" / "research" / "uav_intent_generalization.paper.json"
        ).read_text(encoding="utf-8"))

    def make_results(self):
        behavior_keys = self.spec["generalization"]["behavior_query_keys"]
        split_by_key = {}
        for key in behavior_keys:
            if key.startswith("seen_"):
                split_by_key[key] = "seen"
            elif key.startswith("paraphrase_"):
                split_by_key[key] = "paraphrase"
            else:
                split_by_key[key] = "unseen"
        variant_offsets = {
            "objective_grounded_semantic": 0.20,
            "pretrained_semantic": 0.10,
            "legacy_hash": 0.05,
            "random_dense_oracle": 0.15,
            "identity_oracle": 0.18,
            "no_intent": 0.0,
        }
        results = {}
        for variant, offset in variant_offsets.items():
            results[variant] = {}
            for seed_index, seed in enumerate(self.spec["seeds"]):
                behavior = {}
                for query_index, key in enumerate(behavior_keys):
                    base = 0.4 + 0.001 * seed_index + 0.0001 * query_index
                    behavior[key] = {
                        "split": split_by_key[key],
                        "risk_tiers": {"hard": {
                            "task_completion": base + offset,
                            "episode_return": 10.0 * (base + offset),
                        }},
                    }
                results[variant][seed] = {
                    "intent_generalization": {"behavior": behavior}
                }
        return results

    def test_queries_are_aggregated_within_each_of_ten_seeds(self):
        report = compute_generalization_statistics(self.make_results(), self.spec)
        self.assertEqual(report["status"], "valid")
        self.assertFalse(report["query_pseudoreplication"])
        self.assertEqual(report["seed_count"], 10)
        self.assertEqual(report["family_size"], 12)
        for hypothesis in report["hypotheses"].values():
            self.assertEqual(hypothesis["n_paired_seeds"], 10)
            self.assertEqual(
                hypothesis["paired"]["randomization_test"]["method"],
                "exact_paired_sign_flip",
            )

    def test_holm_family_contains_only_confirmatory_nonoracle_baselines(self):
        report = compute_generalization_statistics(self.make_results(), self.spec)
        self.assertEqual(report["multiple_testing"]["family_size"], 12)
        self.assertTrue(all(
            "identity_oracle" not in key and "random_dense_oracle" not in key
            for key in report["hypotheses"]
        ))
        self.assertTrue(all(report["multiple_testing"]["reject"].values()))

    def test_missing_registered_seed_is_rejected(self):
        results = self.make_results()
        del results["legacy_hash"][self.spec["seeds"][-1]]
        with self.assertRaisesRegex(ValueError, "exact registered seeds"):
            compute_generalization_statistics(results, self.spec)

    def test_missing_split_query_is_rejected(self):
        results = self.make_results()
        for seed in self.spec["seeds"]:
            behavior = results["pretrained_semantic"][seed]["intent_generalization"]["behavior"]
            for key in list(behavior):
                if behavior[key]["split"] == "unseen":
                    del behavior[key]
        with self.assertRaisesRegex(ValueError, "no behavior queries"):
            compute_generalization_statistics(results, self.spec)

    def test_protocol_family_size_tampering_is_rejected(self):
        spec = copy.deepcopy(self.spec)
        spec["reporting"]["generalization_contract"]["family_size"] = 11
        with self.assertRaisesRegex(ValueError, "family size"):
            compute_generalization_statistics(self.make_results(), spec)


if __name__ == "__main__":
    unittest.main()
