from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from research_ablation import validate_ablation_contract  # noqa: E402
from run_research_study import (  # noqa: E402
    annotate_primary_holm,
    summarize_ablation_comparisons,
)


def contracted_spec() -> dict:
    common = {
        "algorithm": "imappo",
        "intent_source": "onehot",
        "use_action_mask": True,
    }
    return {
        "treatment_key": "full",
        "variants": [
            {"key": "full", **common},
            {"key": "no_mask", **common, "use_action_mask": False},
            {
                "key": "zero_intent",
                **common,
                "use_action_mask": False,
                "intent_source": "none",
            },
        ],
        "evaluation": {"risk_tiers": {"easy": {}, "hard": {}}},
        "ablation_contract": {
            "version": 1,
            "treatment_key": "full",
            "comparisons": [
                {
                    "reference": "full",
                    "variant": "no_mask",
                    "factor": "action_mask",
                    "changed_fields": ["use_action_mask"],
                    "primary_metrics": ["collision_rate", "task_completion"],
                    "primary_tiers": ["hard"],
                    "hypothesis": "Removing the mask changes hard-tier safety or utility.",
                },
                {
                    "reference": "no_mask",
                    "variant": "zero_intent",
                    "factor": "intent_conditioning",
                    "changed_fields": ["intent_source"],
                    "primary_metrics": ["collision_rate"],
                    "primary_tiers": ["hard"],
                    "hypothesis": "Removing intent conditioning changes hard-tier safety.",
                },
            ],
        },
    }


def seed_result(seed: int, collision: float, task: float) -> dict:
    return {
        "seed": seed,
        "tier_metrics": {
            "hard": {
                "hard_collision_rate": collision,
                "hard_task_completion": task,
            }
        },
    }


class ResearchAblationContractTests(unittest.TestCase):
    def test_valid_chained_contract_is_auditable(self):
        audit = validate_ablation_contract(contracted_spec())
        self.assertEqual(audit["status"], "valid")
        self.assertEqual(audit["comparison_count"], 2)
        self.assertEqual(
            audit["comparisons"][1]["changed_fields"], ["intent_source"]
        )

    def test_contract_rejects_undeclared_factor_drift(self):
        spec = contracted_spec()
        spec["variants"][1]["intent_source"] = "none"
        with self.assertRaisesRegex(ValueError, "observed"):
            validate_ablation_contract(spec)

    def test_contract_rejects_uncovered_variant(self):
        spec = contracted_spec()
        spec["ablation_contract"]["comparisons"].pop()
        with self.assertRaisesRegex(ValueError, "does not cover"):
            validate_ablation_contract(spec)

    def test_chained_summary_uses_variant_minus_reference(self):
        spec = contracted_spec()
        audit = validate_ablation_contract(spec)
        results = {
            "full": [seed_result(7, 0.10, 0.80), seed_result(11, 0.20, 0.70)],
            "no_mask": [seed_result(7, 0.30, 0.70), seed_result(11, 0.40, 0.60)],
            "zero_intent": [
                seed_result(7, 0.35, 0.65),
                seed_result(11, 0.45, 0.55),
            ],
        }
        comparisons = summarize_ablation_comparisons(results, audit, 17)
        self.assertAlmostEqual(
            comparisons["no_mask"]["risk_tiers"]["hard"]
            ["collision_rate"]["mean_difference"],
            0.20,
        )
        self.assertEqual(
            comparisons["zero_intent"]["reference_key"], "no_mask"
        )
        family = annotate_primary_holm(comparisons, audit)
        self.assertEqual(family["family_size"], 3)
        self.assertEqual(
            family["family_definition"],
            "ablation_contract primary_metrics × primary_tiers",
        )

    def test_contract_requires_explicit_fields_on_both_variants(self):
        spec = copy.deepcopy(contracted_spec())
        del spec["variants"][0]["use_action_mask"]
        with self.assertRaisesRegex(ValueError, "explicitly set"):
            validate_ablation_contract(spec)


if __name__ == "__main__":
    unittest.main()
