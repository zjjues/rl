from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from generalization_protocol import (  # noqa: E402
    build_generalization_protocol_audit,
    validate_calibration_compatibility,
    validate_generalization_paper_protocol,
)
from intent_generalization import load_generalization_suite  # noqa: E402


class GeneralizationPaperProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paper = json.loads((
            ROOT / "configs" / "research" / "uav_intent_generalization.paper.json"
        ).read_text(encoding="utf-8"))
        cls.calibration = json.loads((
            ROOT / "configs" / "research" /
            "uav_intent_generalization.calibration.json"
        ).read_text(encoding="utf-8"))
        cls.suite = load_generalization_suite(
            ROOT / "configs" / "research" /
            "uav_intent_generalization_suite.v8.json"
        )

    def test_repository_protocol_is_valid_and_seed_is_statistical_unit(self):
        audit = validate_generalization_paper_protocol(self.paper, self.suite)
        self.assertEqual(audit["status"], "valid")
        self.assertEqual(audit["expected_result_count"], 60)
        self.assertEqual(audit["behavior_query_count"], 12)
        self.assertEqual(audit["representation_query_count"], 30)
        self.assertEqual(audit["confirmatory_unit"], "seed")

    def test_action_mask_label_oracle_is_rejected(self):
        spec = copy.deepcopy(self.paper)
        spec["variants"][0]["use_action_mask"] = True
        with self.assertRaisesRegex(ValueError, "use_action_mask"):
            validate_generalization_paper_protocol(spec, self.suite)

    def test_counterfactual_queries_cannot_enter_behavior_family(self):
        spec = copy.deepcopy(self.paper)
        spec["generalization"]["behavior_query_keys"].append("cf_energy_high")
        with self.assertRaisesRegex(ValueError, "counterfactual"):
            validate_generalization_paper_protocol(spec, self.suite)

    def test_identity_oracle_cannot_be_confirmatory_baseline(self):
        spec = copy.deepcopy(self.paper)
        contract = spec["reporting"]["generalization_contract"]
        contract["confirmatory_baselines"].append("identity_oracle")
        contract["family_size"] = 16
        with self.assertRaisesRegex(ValueError, "frozen contract"):
            validate_generalization_paper_protocol(spec, self.suite)

    def test_variant_specific_input_dimension_is_rejected(self):
        spec = copy.deepcopy(self.paper)
        spec["variants"][4]["intent_dim"] = 25
        with self.assertRaisesRegex(ValueError, "shared 64-D"):
            validate_generalization_paper_protocol(spec, self.suite)

    def test_calibration_must_use_exact_variants(self):
        calibration = copy.deepcopy(self.calibration)
        calibration["variants"][0]["cbf_iterations"] = 3
        with self.assertRaisesRegex(ValueError, "variants"):
            validate_calibration_compatibility(self.paper, calibration)

    def test_repository_configs_produce_checksummed_protocol_audit(self):
        audit = build_generalization_protocol_audit(
            ROOT,
            "configs/research/uav_intent_generalization.paper.json",
            "configs/research/uav_intent_generalization.calibration.json",
        )
        self.assertTrue(audit["calibration_compatible"])
        self.assertEqual(len(audit["paper_config_sha256"]), 64)
        self.assertEqual(len(audit["suite_resolved_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
