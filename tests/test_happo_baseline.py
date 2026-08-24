from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from happo_baseline import HAPPOBaseline  # noqa: E402
from imappo import IMAPPOConfig, RolloutBuffer  # noqa: E402


class HAPPOBaselineTests(unittest.TestCase):
    def config(self) -> IMAPPOConfig:
        return IMAPPOConfig(
            algorithm="happo",
            critic_mode="mlp",
            intent_source="none",
            intent_profile_decoder="none",
            use_action_mask=False,
            policy_mode="direct",
            safety_filter_mode="none",
            n_agents=3,
            n_targets=3,
            obs_dim=18,
            state_dim=54,
            intent_dim=8,
            ppo_epochs=2,
            minibatch_size=4,
            eta=0.0,
            eta_end=0.0,
            potential_update_mode="frozen",
            device="cpu",
            seed=7,
        )

    def test_has_independent_actors_and_auditable_metadata(self):
        algo = HAPPOBaseline(self.config())
        self.assertEqual(len(algo.actor), 3)
        first_parameters = [next(actor.parameters()) for actor in algo.actor]
        self.assertEqual(len({parameter.data_ptr() for parameter in first_parameters}), 3)
        metadata = algo.algorithm_metadata()
        self.assertEqual(metadata["actor_parameter_sharing"], "independent")
        self.assertEqual(
            metadata["update_scheme"], "random_sequential_likelihood_factor"
        )

    def test_select_actions_uses_one_actor_per_registered_agent(self):
        algo = HAPPOBaseline(self.config())
        actions, log_probs = algo.select_actions(
            torch.zeros(3, 18), torch.zeros(8), torch.ones(3, 3), deterministic=True
        )
        self.assertEqual(tuple(actions.shape), (3, 3))
        self.assertEqual(tuple(log_probs.shape), (3,))
        with self.assertRaisesRegex(ValueError, "registered training agent count"):
            algo.select_actions(
                torch.zeros(4, 18),
                torch.zeros(8),
                torch.ones(4, 3),
                deterministic=True,
            )

    def test_sequential_update_changes_each_actor_and_reports_factor(self):
        algo = HAPPOBaseline(self.config())
        buffer = RolloutBuffer()
        generator = torch.Generator().manual_seed(23)
        intent = torch.zeros(8)
        mask = torch.ones(3, 3)
        for step in range(8):
            obs = torch.randn(3, 18, generator=generator)
            state = obs.reshape(-1)
            actions, log_probs = algo.select_actions(obs, intent, mask)
            next_obs = torch.randn(3, 18, generator=generator)
            buffer.add(
                state=state,
                obs=obs,
                action=actions,
                base_action=algo._last_base_actions,
                policy_latent=algo._last_policy_latents,
                action_mask=mask,
                intent=intent,
                reward=torch.randn(3, generator=generator),
                done=torch.tensor(float(step == 7)),
                log_prob=log_probs,
                next_state=next_obs.reshape(-1),
                next_obs=next_obs,
            )
        before = [
            [parameter.detach().clone() for parameter in actor.parameters()]
            for actor in algo.actor
        ]
        log = algo.update(buffer)
        for actor_id, actor in enumerate(algo.actor):
            self.assertTrue(
                any(
                    not torch.allclose(old, new)
                    for old, new in zip(before[actor_id], actor.parameters())
                )
            )
        self.assertEqual(sorted(algo.last_agent_order), [0, 1, 2])
        self.assertTrue(math.isfinite(log["happo_factor_mean"]))
        self.assertTrue(math.isfinite(log["happo_factor_abs_max"]))
        self.assertEqual(len(buffer.storage["states"]), 0)

    def test_checkpoint_preserves_all_independent_actors(self):
        algo = HAPPOBaseline(self.config())
        for _ in range(4):
            algo.rng.permutation(algo.config.n_agents)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "happo.pt")
            algo.save_checkpoint(path)
            expected_order = algo.rng.permutation(algo.config.n_agents).tolist()
            restored = HAPPOBaseline.load_checkpoint(path)
        for expected_actor, actual_actor in zip(algo.actor, restored.actor):
            for expected, actual in zip(
                expected_actor.parameters(), actual_actor.parameters()
            ):
                self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(
            expected_order,
            restored.rng.permutation(restored.config.n_agents).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
