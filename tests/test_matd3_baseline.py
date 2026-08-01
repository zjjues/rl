from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imappo import IMAPPOConfig  # noqa: E402
from matd3_baseline import MATD3Baseline  # noqa: E402


class MATD3BaselineTests(unittest.TestCase):
    def config(self):
        return IMAPPOConfig(
            algorithm="matd3",
            n_agents=2,
            n_targets=2,
            obs_dim=18,
            state_dim=36,
            action_dim=3,
            intent_dim=25,
            minibatch_size=4,
            matd3_policy_delay=1,
            device="cpu",
            seed=7,
        )

    def test_policy_has_no_intent_input_and_continuous_actions(self):
        algo = MATD3Baseline(self.config())
        intent, mask, _ = algo.evaluation_intent_and_mask("dense")
        actions, _ = algo.select_actions(
            torch.zeros(2, 18), intent, mask, deterministic=True
        )
        self.assertEqual(tuple(actions.shape), (2, 3))
        self.assertTrue((actions.abs() <= 1.0).all().item())
        metadata = algo.intent_representation_metadata()
        self.assertFalse(metadata["intent_conditioning"])
        self.assertTrue(metadata["centralized_twin_critics"])

    def test_twin_critic_and_delayed_actor_update_are_finite(self):
        cfg = self.config()
        algo = MATD3Baseline(cfg)
        rng = np.random.default_rng(3)
        for _ in range(8):
            algo.replay.add(
                rng.normal(size=cfg.state_dim),
                rng.normal(size=(cfg.n_agents, cfg.obs_dim)),
                rng.uniform(-1, 1, size=(cfg.n_agents, cfg.action_dim)),
                float(rng.normal()),
                rng.normal(size=cfg.state_dim),
                rng.normal(size=(cfg.n_agents, cfg.obs_dim)),
                False,
            )
        before = [parameter.detach().clone() for parameter in algo.actor.parameters()]
        metrics = algo.update()
        self.assertTrue(np.isfinite(metrics["critic_loss"]))
        self.assertTrue(np.isfinite(metrics["actor_loss"]))
        self.assertEqual(metrics["actor_updates"], 1.0)
        self.assertEqual(metrics["actor_updated"], 1.0)
        self.assertTrue(
            any(not torch.allclose(old, new) for old, new in zip(before, algo.actor.parameters()))
        )


if __name__ == "__main__":
    unittest.main()
