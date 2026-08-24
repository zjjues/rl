from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from envs.vmas_adapter import VMASAdapter, infer_vmas_dims  # noqa: E402
from run_research_study import build_config  # noqa: E402


class VMASAdapterTests(unittest.TestCase):
    def test_navigation_continuous_api_and_dimensions(self):
        obs_dim, state_dim, action_dim = infer_vmas_dims("navigation", 3)
        self.assertEqual((obs_dim, state_dim, action_dim), (18, 54, 2))
        env = VMASAdapter("navigation", n_agents=3, max_steps=5)
        try:
            observations, info = env.reset(seed=7)
            self.assertEqual(len(observations), 3)
            self.assertTrue(all(obs.shape == (18,) for obs in observations))
            actions = [np.zeros(2, dtype=np.float32) for _ in range(3)]
            next_obs, rewards, done, truncated, next_info = env.step(actions)
            self.assertEqual(len(next_obs), 3)
            self.assertEqual(len(rewards), 3)
            self.assertIsInstance(done, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)
            self.assertIsInstance(next_info, dict)
            self.assertIn("collision", next_info)
            self.assertIn("uav_0", next_info)
        finally:
            env.close()

    def test_adapter_enforces_registered_episode_horizon(self):
        env = VMASAdapter("dispersion", n_agents=3, max_steps=1)
        try:
            env.reset(seed=9)
            actions = [np.zeros(2, dtype=np.float32) for _ in range(3)]
            _, _, _, truncated, _ = env.step(actions)
            self.assertTrue(truncated)
        finally:
            env.close()

    def test_info_normalisation_does_not_invent_uav_objectives(self):
        info = VMASAdapter._normalise_info({
            "agent_0": {"agent_collisions": np.asarray(1.0), "final_rew": -2.0},
            "agent_1": {"agent_collisions": np.asarray(0.0), "final_rew": -1.0},
        })
        self.assertTrue(info["collision"])
        self.assertEqual(info["uav_0"]["reward_env"], -2.0)
        self.assertNotIn("task_completion", info["uav_0"])

    def test_formal_config_rejects_uav_rule_residual_on_vmas(self):
        spec = {
            "environment": {"name": "vmas:navigation", "n_agents": 3},
            "training": {
                "episodes": 2,
                "steps": 5,
                "rollout_length": 4,
                "minibatch_size": 4,
                "eval_interval": 2,
                "device": "cpu",
            },
            "intent": {
                "dim": 25,
                "encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "evaluation": {"episodes": 1, "risk_tiers": {"default": {}}},
        }
        variant = {
            "key": "invalid",
            "algorithm": "imappo",
            "intent_source": "onehot",
            "policy_mode": "residual_rule",
        }
        with self.assertRaises(ValueError):
            build_config(spec, variant, 7)


if __name__ == "__main__":
    unittest.main()
