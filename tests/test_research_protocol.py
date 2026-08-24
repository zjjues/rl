from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from research_protocol import validate_variant_protocol  # noqa: E402
from run_research_study import validate_spec  # noqa: E402


class ResearchVariantProtocolTests(unittest.TestCase):
    def test_paper_rejects_development_only_relevance_gate(self):
        spec = json.loads((
            ROOT / "configs" / "research" /
            "uav_language_relevance_gate.smoke.json"
        ).read_text(encoding="utf-8"))
        spec["level"] = "paper"
        spec["seeds"] = list(range(10))
        spec["evaluation"]["episodes"] = 100
        with self.assertRaisesRegex(ValueError, "development-only"):
            validate_spec(spec, allow_dirty=True)

    def test_relevance_gate_hash_mismatch_is_rejected(self):
        spec = json.loads((
            ROOT / "configs" / "research" /
            "uav_language_relevance_gate.smoke.json"
        ).read_text(encoding="utf-8"))
        spec["variants"][0]["preference_relevance_gate_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_spec(spec, allow_dirty=True)

    def test_relevance_gate_rejects_baseline_or_nonsemantic_variant(self):
        spec = {
            "variants": [{
                "key": "bad_gate",
                "algorithm": "mappo",
                "critic_mode": "mlp",
                "intent_source": "onehot",
                "intent_profile_decoder": "none",
                "preference_relevance_gate_path": "gate.json",
            }]
        }
        with self.assertRaisesRegex(ValueError, "relevance gate only"):
            validate_variant_protocol(spec)

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
                {
                    "key": "happo",
                    "algorithm": "happo",
                    "critic_mode": "mlp",
                    "intent_source": "none",
                    "use_action_mask": False,
                    "policy_mode": "direct",
                    "safety_filter_mode": "none",
                    "actor_parameter_sharing": "independent",
                    "update_scheme": "random_sequential_likelihood_factor",
                },
                {"key": "matd3", "algorithm": "matd3", "critic_mode": "mlp"},
            ]
        }
        audit = validate_variant_protocol(spec)
        self.assertEqual(audit["status"], "valid")
        self.assertEqual(audit["variant_count"], 5)
        self.assertEqual(
            audit["variants"][-1]["critic_mode_effective"],
            "centralized_twin_critics",
        )

    def test_happo_rejects_shared_actor_alias(self):
        spec = {
            "variants": [
                {
                    "key": "happo",
                    "algorithm": "happo",
                    "critic_mode": "mlp",
                    "intent_source": "none",
                    "use_action_mask": False,
                    "policy_mode": "direct",
                    "safety_filter_mode": "none",
                    "actor_parameter_sharing": "shared",
                    "update_scheme": "random_sequential_likelihood_factor",
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "independent sequential"):
            validate_variant_protocol(spec)

    def test_vmas_rejects_uav_language_or_mask_semantics(self):
        spec = {
            "environment": {"name": "vmas:navigation"},
            "variants": [{
                "key": "imappo",
                "algorithm": "imappo",
                "critic_mode": "attention",
                "intent_source": "onehot",
                "intent_profile_decoder": "none",
                "disable_intent_reward": True,
                "use_action_mask": False,
                "policy_mode": "direct",
                "safety_filter_mode": "none",
            }],
        }
        with self.assertRaisesRegex(ValueError, "architecture-only"):
            validate_variant_protocol(spec)

    def test_vmas_neutral_architecture_variant_is_audited_without_language_claim(self):
        spec = {
            "environment": {"name": "vmas:dispersion"},
            "variants": [{
                "key": "imappo",
                "algorithm": "imappo",
                "critic_mode": "attention",
                "intent_source": "none",
                "intent_profile_decoder": "none",
                "disable_intent_reward": True,
                "use_action_mask": False,
                "policy_mode": "direct",
                "safety_filter_mode": "none",
            }],
        }
        audit = validate_variant_protocol(spec)
        self.assertEqual(
            audit["variants"][0]["evaluation_scope"],
            "architecture_only_no_language_claim",
        )


if __name__ == "__main__":
    unittest.main()
