from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

if torch is not None:
    from imappo import IMAPPO, IMAPPOConfig  # noqa: E402


@unittest.skipUnless(torch is not None, "PyTorch research dependency is not installed")
class IMAPPOIntentAlignmentTests(unittest.TestCase):
    def make_algorithm(self, use_action_mask: bool = True) -> IMAPPO:
        return IMAPPO(
            IMAPPOConfig(
                n_agents=4,
                n_targets=4,
                obs_dim=30,
                state_dim=120,
                intent_dim=8,
                intent_source="legacy_hash",
                use_action_mask=use_action_mask,
                device="cpu",
                seed=7,
            )
        )

    def test_stealth_sampling_is_aligned_and_masked(self):
        algo = self.make_algorithm()
        _, mask, label = algo.sample_episode_intent_and_mask("stealth")
        self.assertIn(algo.intent_library.posture_for_label(label), {"stealth", "neutral"})
        self.assertTrue((mask[:, 2] == 0).all().item())

    def test_no_masking_ablation_returns_full_mask(self):
        algo = self.make_algorithm(use_action_mask=False)
        _, mask, _ = algo.sample_episode_intent_and_mask("stealth")
        self.assertTrue((mask == 1).all().item())

    def test_zero_intent_control_hides_vector_but_preserves_task_protocol(self):
        algo = IMAPPO(
            IMAPPOConfig(
                algorithm="imappo",
                n_agents=4,
                n_targets=4,
                obs_dim=30,
                state_dim=120,
                intent_dim=25,
                intent_source="none",
                intent_profile_decoder="none",
                use_action_mask=True,
                device="cpu",
            )
        )
        intent, mask, label = algo.sample_episode_intent_and_mask("stealth")
        self.assertTrue((intent == 0).all().item())
        self.assertTrue(label)
        self.assertIn(
            algo.task_intent_library.posture_for_label(label), {"stealth", "neutral"}
        )
        self.assertTrue((mask[:, 2] == 0).all().item())
        metadata = algo.intent_representation_metadata()
        self.assertFalse(metadata["intent_conditioning"])
        self.assertTrue(metadata["task_posture_exposed_via_action_mask"])

    def test_zero_intent_query_encoder_never_uses_canonical_identity(self):
        algo = IMAPPO(
            IMAPPOConfig(
                algorithm="imappo",
                n_agents=4,
                n_targets=4,
                obs_dim=30,
                state_dim=120,
                intent_dim=25,
                intent_source="none",
                intent_profile_decoder="none",
                device="cpu",
            )
        )
        encoded = algo.encode_intent_queries(
            [("balanced", "Maintain balanced mission progress.")]
        )
        self.assertEqual(tuple(encoded.shape), (1, 25))
        self.assertTrue((encoded == 0).all().item())

    def test_unknown_critic_mode_is_rejected_instead_of_using_attention(self):
        with self.assertRaisesRegex(ValueError, "unsupported critic_mode"):
            IMAPPO(
                IMAPPOConfig(
                    n_agents=4,
                    n_targets=4,
                    obs_dim=30,
                    state_dim=120,
                    intent_dim=8,
                    intent_source="legacy_hash",
                    critic_mode="concat",
                    device="cpu",
                )
            )

    def test_posture_alignment_ablation_can_sample_conflicting_intent(self):
        algo = IMAPPO(
            IMAPPOConfig(
                n_agents=4,
                n_targets=4,
                obs_dim=30,
                state_dim=120,
                intent_dim=8,
                intent_source="legacy_hash",
                align_intent_posture=False,
                device="cpu",
                seed=7,
            )
        )
        observed = set()
        for _ in range(200):
            _, _, label = algo.sample_episode_intent_and_mask("attack")
            observed.add(algo.intent_library.posture_for_label(label))
        self.assertIn("stealth", observed)

    def test_onehot_uses_catalog_identity_and_training_subset(self):
        algo = IMAPPO(
            IMAPPOConfig(
                n_agents=4,
                n_targets=4,
                obs_dim=30,
                state_dim=120,
                intent_dim=25,
                intent_source="onehot",
                intent_train_labels=("balanced", "safety_first"),
                device="cpu",
                seed=7,
            )
        )
        self.assertEqual(algo.intent_library.labels, ["balanced", "safety_first"])
        queries = algo.encode_intent_queries(
            [("balanced", "A reworded balanced mission"), ("rapid_response", "Move immediately")]
        )
        self.assertEqual(tuple(queries.shape), (2, 25))
        self.assertEqual(float(queries[0].sum().item()), 1.0)
        self.assertEqual(float(queries[1].sum().item()), 1.0)
        self.assertFalse(bool((queries[0] == queries[1]).all().item()))

    def test_ippo_has_local_per_agent_critic_and_no_intent_conditioning(self):
        algo = IMAPPO(
            IMAPPOConfig(
                algorithm="ippo",
                critic_mode="local",
                n_agents=4,
                n_targets=4,
                obs_dim=30,
                state_dim=120,
                intent_dim=8,
                intent_source="onehot",
                use_action_mask=False,
                eta=0.0,
                eta_end=0.0,
                potential_update_mode="frozen",
                device="cpu",
                seed=7,
            )
        )
        self.assertIsNone(algo.intent_library)
        self.assertIsNotNone(algo.task_intent_library)
        intent, mask, label = algo.sample_episode_intent_and_mask("attack")
        self.assertTrue((intent == 0).all().item())
        self.assertTrue((mask == 1).all().item())
        self.assertIn(
            algo.task_intent_library.posture_for_label(label),
            {"attack", "neutral"},
        )
        states = torch.zeros(2, 120)
        intents = torch.zeros(2, 8)
        observations = torch.zeros(2, 4, 30)
        values, local_weights = algo.critic(states, intents, observations)
        self.assertEqual(tuple(values.shape), (2, 4))
        self.assertEqual(tuple(local_weights.shape), (2, 4, 4))
        self.assertEqual(algo.intent_representation_metadata()["representation_type"], "none")

    def test_private_rng_makes_intent_sequence_seed_deterministic(self):
        def build(seed: int) -> IMAPPO:
            return IMAPPO(
                IMAPPOConfig(
                    algorithm="imappo",
                    n_agents=4,
                    n_targets=4,
                    obs_dim=30,
                    state_dim=120,
                    intent_dim=25,
                    intent_source="none",
                    intent_profile_decoder="none",
                    device="cpu",
                    seed=seed,
                )
            )

        first = build(7)
        second = build(7)
        different = build(11)
        first_labels = [first.sample_episode_intent_and_mask()[2] for _ in range(16)]
        second_labels = [second.sample_episode_intent_and_mask()[2] for _ in range(16)]
        different_labels = [
            different.sample_episode_intent_and_mask()[2] for _ in range(16)
        ]
        self.assertEqual(first_labels, second_labels)
        self.assertNotEqual(first_labels, different_labels)

    def test_checkpoint_restores_next_private_rng_sample(self):
        algo = IMAPPO(
            IMAPPOConfig(
                algorithm="imappo",
                n_agents=4,
                n_targets=4,
                obs_dim=30,
                state_dim=120,
                intent_dim=25,
                intent_source="none",
                intent_profile_decoder="none",
                device="cpu",
                seed=7,
            )
        )
        for _ in range(5):
            algo.sample_episode_intent_and_mask()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "imappo.pt")
            algo.save_checkpoint(path)
            expected = [algo.sample_episode_intent_and_mask()[2] for _ in range(12)]
            restored = IMAPPO.load_checkpoint(path)
        actual = [restored.sample_episode_intent_and_mask()[2] for _ in range(12)]
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
