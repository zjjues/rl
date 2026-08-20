from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_protocol import validate_variant_protocol  # noqa: E402


class ResearchVariantProtocolTests(unittest.TestCase):
    def test_legacy_concat_mode_is_rejected(self):
        spec = {
            "variants": [
                {"key": "baseline", "algorithm": "imappo", "critic_mode": "concat"}
            ]
        }
        with self.assertRaisesRegex(ValueError, "was not implemented"):
            validate_variant_protocol(spec)

    def test_reserved_baseline_name_must_match_algorithm(self):
        spec = {
            "variants": [
                {"key": "mappo", "algorithm": "imappo", "critic_mode": "attention"}
            ]
        }
        with self.assertRaisesRegex(ValueError, "reserved baseline key"):
            validate_variant_protocol(spec)

    def test_ippo_must_declare_effective_local_critic(self):
        spec = {
            "variants": [
                {"key": "ippo", "algorithm": "ippo", "critic_mode": "mlp"}
            ]
        }
        with self.assertRaisesRegex(ValueError, "must explicitly declare"):
            validate_variant_protocol(spec)

    def test_valid_strong_baselines_are_audited(self):
        spec = {
            "variants": [
                {"key": "imappo", "algorithm": "imappo", "critic_mode": "attention"},
                {"key": "mappo", "algorithm": "mappo", "critic_mode": "mlp"},
                {"key": "ippo", "algorithm": "ippo", "critic_mode": "local"},
                {"key": "matd3", "algorithm": "matd3", "critic_mode": "mlp"},
            ]
        }
        audit = validate_variant_protocol(spec)
        self.assertEqual(audit["status"], "valid")
        self.assertEqual(audit["variant_count"], 4)
        self.assertEqual(
            audit["variants"][-1]["critic_mode_effective"],
            "centralized_twin_critics",
        )


if __name__ == "__main__":
    unittest.main()
